"""
Training script for the Contextual Sub-Model (DistilBERT + Multi-Task Heads).

- Reads versioned dataset from context_dataset.csv.
- Strictly uses 'train' split for gradient descent and 'val' split for model checkpoint selection.
- Optimizes multi-task objective with primary emphasis on cyberbullying classification.
- Saves best checkpoint to backend/models/weights/context_submodel.pt.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.context_model import MultiTaskContextualTransformer
from models.tokenizer import get_context_tokenizer, format_context_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ContextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.texts = []
        for _, row in df.iterrows():
            formatted = format_context_input(str(row["text"]), str(row.get("context_snippet", "")))
            self.texts.append(formatted)

        self.y_cb = df["cyberbullying"].values.astype(np.float32)
        self.y_targ = df["is_targeted"].values.astype(np.float32)
        self.y_rep = df["is_repeated"].values.astype(np.float32)
        self.y_esc = df["is_escalating"].values.astype(np.float32)
        self.y_sarc = df["is_sarcastic"].values.astype(np.float32)
        self.y_threat = df["is_threatening"].values.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "y_cb": torch.tensor(self.y_cb[idx]),
            "y_targ": torch.tensor(self.y_targ[idx]),
            "y_rep": torch.tensor(self.y_rep[idx]),
            "y_esc": torch.tensor(self.y_esc[idx]),
            "y_sarc": torch.tensor(self.y_sarc[idx]),
            "y_threat": torch.tensor(self.y_threat[idx])
        }

def train_context_model(
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    seed: int = 42
):
    print("=======================================================")
    print("TRAINING CONTEXTUAL SUB-MODEL (Phase 2)")
    print("=======================================================")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    csv_path = os.path.join(os.path.dirname(__file__), "context_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Context dataset missing at {csv_path}. Run generate_context_dataset.py first.")

    df = pd.read_csv(csv_path)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)

    print(f"Loaded {len(df_train)} training samples and {len(df_val)} validation samples.")

    tokenizer = get_context_tokenizer("distilbert-base-uncased")
    train_dataset = ContextDataset(df_train, tokenizer)
    val_dataset = ContextDataset(df_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MultiTaskContextualTransformer()
    model.to(device)

    # Multi-task loss functions
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    weights_dir = os.path.join(os.path.dirname(__file__), "..", "models", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_weights_path = os.path.join(weights_dir, "context_submodel.pt")

    best_val_f1 = -1.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            y_cb = batch["y_cb"].to(device)
            y_targ = batch["y_targ"].to(device)
            y_rep = batch["y_rep"].to(device)
            y_esc = batch["y_esc"].to(device)
            y_sarc = batch["y_sarc"].to(device)
            y_threat = batch["y_threat"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            loss_cb = bce(logits["logit_cyberbullying"], y_cb)
            loss_targ = bce(logits["logit_targeted"], y_targ)
            loss_rep = bce(logits["logit_repeated"], y_rep)
            loss_esc = bce(logits["logit_escalating"], y_esc)
            loss_sarc = bce(logits["logit_sarcastic"], y_sarc)
            loss_threat = bce(logits["logit_threatening"], y_threat)

            # Combined multi-task objective
            loss = loss_cb + 0.25 * (loss_targ + loss_rep + loss_esc + loss_sarc + loss_threat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation evaluation
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                y_cb = batch["y_cb"].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_cb = bce(logits["logit_cyberbullying"], y_cb)
                val_loss += loss_cb.item()

                probs = torch.sigmoid(logits["logit_cyberbullying"]).cpu().numpy()
                val_preds.extend((probs >= 0.50).astype(int))
                val_targets.extend(y_cb.cpu().numpy().astype(int))

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)
        val_rec = recall_score(val_targets, val_preds, zero_division=0)
        val_prec = precision_score(val_targets, val_preds, zero_division=0)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and avg_val_loss < best_metrics.get("val_loss", 999)):
            best_val_f1 = val_f1
            best_metrics = {
                "epoch": epoch,
                "val_loss": round(avg_val_loss, 4),
                "val_f1": round(val_f1 * 100, 2),
                "val_accuracy": round(val_acc * 100, 2),
                "val_recall": round(val_rec * 100, 2),
                "val_precision": round(val_prec * 100, 2)
            }
            torch.save(model.state_dict(), best_weights_path)
            print(f"  -> Checkpoint saved: {best_weights_path} (F1: {best_val_f1*100:.1f}%)")

    # Save training metadata
    meta_path = os.path.join(weights_dir, "context_model_metadata.json")
    metadata = {
        "model_name": "distilbert-base-uncased",
        "version": "context-distilbert-v1",
        "training_samples": len(df_train),
        "validation_samples": len(df_val),
        "best_epoch": best_metrics["epoch"],
        "metrics": best_metrics,
        "saved_at": best_weights_path
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nContextual Sub-Model Training Complete!")
    print(f"Best Validation Performance: F1={best_metrics['val_f1']}%, Acc={best_metrics['val_accuracy']}%")
    return best_weights_path

if __name__ == "__main__":
    train_context_model()
