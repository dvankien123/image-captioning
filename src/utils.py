"""Shared persistence, configuration, and training utilities."""

import os
import json
from pathlib import Path
import pickle
import yaml

def save_checkpoint(model, optimizer, epoch, val_loss, vocab, model_config, path):
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "vocab": vocab,
        "model_config": model_config,
    }, path)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True   # cải thiện -> đáng lưu checkpoint
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # không cải thiện


def save_vocab(vocab):
    config, config_dir = load_config()

    vocab_path = config_dir / config["paths"]["vocab"]

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    
def load_vocab(vocab_path):
    vocab_path = Path(vocab_path).expanduser()

    if not Path(vocab_path).is_file():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    
    with open(vocab_path, "rb") as f:
        return pickle.load(f)

def save_splits(train, val, test):
    config, config_dir = load_config()

    train_path = config_dir / config["paths"]["train_json"]
    val_path = config_dir / config["paths"]["val_json"]
    test_path = config_dir / config["paths"]["test_json"]

    for path in (train_path, val_path, test_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train, f)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val, f)
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test, f)

def load_splits(data_path):
    data_path = Path(data_path).expanduser()

    if not data_path.is_file():
        raise FileNotFoundError(f"One or more split files not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_config(config_path=None):
    """Load config, defaulting to the project-root file independent of cwd."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    else:
        # Explicit relative paths intentionally follow the caller's working directory.
        config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found {config_path}")
    
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config, config_path.parent
        
