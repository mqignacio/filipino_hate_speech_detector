import argparse
import copy
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models.transformer import SmallTransformerClassifier
from utils.data_utils import SimpleTokenizer


class EncodedTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: SimpleTokenizer):
        self.labels = [int(l) for l in labels]
        self.encoded = [tokenizer.encode(t) for t in texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {description} at {path}, but it was not found.")


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"File {path} must contain 'text' and 'label' columns.")
    return df[['text', 'label']]


def build_tokenizer(train_texts: List[str], extra_texts: Optional[List[str]], max_len: int) -> SimpleTokenizer:
    all_texts = list(train_texts)
    if extra_texts:
        all_texts.extend(extra_texts)
    return SimpleTokenizer(all_texts, max_len=max_len)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'accuracy': correct / total if total > 0 else 0.0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    keys = metric_list[0].keys()
    return {k: float(np.mean([metrics[k] for metrics in metric_list])) for k in keys}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    grad_clip: float,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    metrics_accumulator: List[Dict[str, float]] = []

    autocast_enabled = scaler is not None and device.type == 'cuda'

    for batch_idx, (inputs, targets) in enumerate(loader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=autocast_enabled):
            logits = model(inputs)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        metrics_accumulator.append(compute_metrics(logits.detach().cpu(), targets.detach().cpu()))

    return total_loss / max(len(loader), 1), aggregate_metrics(metrics_accumulator)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    metrics_accumulator: List[Dict[str, float]] = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        metrics_accumulator.append(compute_metrics(logits.cpu(), targets.cpu()))

    return total_loss / max(len(loader), 1), aggregate_metrics(metrics_accumulator)


def create_dataloader(texts: List[str], labels: List[int], tokenizer: SimpleTokenizer, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = EncodedTextDataset(texts, labels, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment transformer on processed datasets")
    parser.add_argument('--train-csv', type=Path, default=Path('data/combined/processed/train.csv'))
    parser.add_argument('--val-csv', type=Path, default=Path('data/combined/processed/validation.csv'))
    parser.add_argument('--test-csv', type=Path, default=Path('data/combined/processed/test.csv'))
    parser.add_argument('--max-len', type=int, default=128)
    parser.add_argument('--embed-dim', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=2)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--mixed-precision', action='store_true', help='Enable automatic mixed precision (CUDA only).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('models/processed'))
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience based on validation accuracy.')
    parser.add_argument('--min-delta', type=float, default=1e-3, help='Minimum accuracy improvement to reset patience.')
    parser.add_argument('--log-dir', type=Path, default=Path('runs/processed'), help='Base directory for TensorBoard logs.')
    parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard logging.')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / 'sentiment_transformer.pt'
    last_model_path = args.output_dir / 'sentiment_transformer_last.pt'
    tokenizer_path = args.output_dir / 'tokenizer.json'
    metrics_path = args.output_dir / 'metrics.json'

    ensure_file(args.train_csv, 'training CSV')
    ensure_file(args.val_csv, 'validation CSV')
    ensure_file(args.test_csv, 'test CSV')

    train_df = load_dataframe(args.train_csv)
    val_df = load_dataframe(args.val_csv)
    test_df = load_dataframe(args.test_csv)

    extra_texts: Optional[List[str]] = None
    extra_labels: Optional[List[int]] = None

    tokenizer = build_tokenizer(
        train_df['text'].astype(str).tolist(),
        extra_texts,
        max_len=args.max_len,
    )

    train_texts = train_df['text'].astype(str).tolist()
    train_labels = train_df['label'].astype(int).tolist()
    if extra_texts and extra_labels:
        train_texts.extend(extra_texts)
        train_labels.extend(extra_labels)

    val_texts = val_df['text'].astype(str).tolist()
    val_labels = val_df['label'].astype(int).tolist()

    test_texts = test_df['text'].astype(str).tolist()
    test_labels = test_df['label'].astype(int).tolist()

    train_loader = create_dataloader(train_texts, train_labels, tokenizer, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_texts, val_labels, tokenizer, args.batch_size, shuffle=False)
    test_loader = create_dataloader(test_texts, test_labels, tokenizer, args.batch_size, shuffle=False)

    model = SmallTransformerClassifier(
        vocab_size=tokenizer.vocab_size(),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)

    model_hparams = {
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'max_len': args.max_len,
    }

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision and device.type == 'cuda')

    history: List[Dict[str, Dict[str, float]]] = []
    best_val_accuracy = float('-inf')
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    writer: Optional[SummaryWriter] = None
    run_log_dir: Optional[Path] = None
    if not args.no_tensorboard:
        run_log_dir = args.log_dir / datetime.now().strftime('%Y%m%d-%H%M%S')
        run_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_log_dir))

    def snapshot_best_state(epoch: int, val_loss: float, val_metrics: Dict[str, float]) -> Dict[str, object]:
        return {
            'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'model_hyperparams': copy.deepcopy(model_hparams),
            'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        }

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.grad_clip,
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_metrics': train_metrics,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
        })

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            for metric_name, metric_value in train_metrics.items():
                writer.add_scalar(f'Metrics/train/{metric_name}', metric_value, epoch)
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f'Metrics/validation/{metric_name}', metric_value, epoch)
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LearningRate', lr, epoch)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}"
        )

        val_accuracy = val_metrics.get('accuracy', 0.0)
        improved = val_accuracy > best_val_accuracy + args.min_delta

        if improved:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            patience_counter = 0
            best_state = snapshot_best_state(epoch, val_loss, val_metrics)
            torch.save(best_state, model_path)
            print(f"New best model saved at epoch {epoch} to {model_path}")
            if writer is not None:
                writer.add_scalar('Metrics/validation/best_accuracy', best_val_accuracy, epoch)
        else:
            patience_counter += 1
            if patience_counter > args.patience:
                print('Early stopping triggered.')
                break

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, last_model_path)

    if best_state is None:
        best_state = snapshot_best_state(args.epochs, val_loss, val_metrics)
        best_val_accuracy = best_state['val_metrics'].get('accuracy', best_val_accuracy)
        best_val_loss = best_state['val_loss']
        torch.save(best_state, model_path)

    model.load_state_dict(best_state['model_state_dict'])
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['f1']:.4f}")

    tokenizer.save(tokenizer_path)

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    if writer is not None:
        writer.add_text('config/json', json.dumps(cfg, indent=2))
        writer.add_scalar('Loss/test', test_loss, best_state['epoch'])
        for metric_name, metric_value in test_metrics.items():
            writer.add_scalar(f'Metrics/test/{metric_name}', metric_value, best_state['epoch'])
        writer.add_text('metrics/best_epoch', str(best_state['epoch']))
        writer.flush()
        writer.close()

    summary = {
        'train_history': history,
        'best_epoch': best_state['epoch'],
        'best_val_accuracy': best_val_accuracy,
        'best_val_loss': best_state['val_loss'],
        'best_val_metrics': best_state['val_metrics'],
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'config': cfg,
        'model_hyperparams': model_hparams,
        'device': str(device),
        'artifacts': {
            'best_model': str(model_path),
            'last_checkpoint': str(last_model_path),
            'tokenizer': str(tokenizer_path),
        },
        'tensorboard_run_dir': str(run_log_dir) if run_log_dir is not None else None,
    }

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Last checkpoint saved to {last_model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Metrics saved to {metrics_path}")
    if run_log_dir is not None:
        print(f"TensorBoard logs written to {run_log_dir}")


if __name__ == '__main__':
    main()
