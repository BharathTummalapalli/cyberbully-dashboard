"""
Training script for the Learned Context-Aware Fusion MLP.

- Computes P_toxic and P_context signals across train/val/test splits.
- Fits feature normalization parameters STRICTLY on the training set (Zero Test-Leakage).
- Exports feature_scaler.json.
- Trains ContextAwareFusionMLP using BCEWithLogitsLoss and AdamW.
- Saves best model weights to backend/models/weights/fusion_mlp.pt.
- Evaluates on the held-out test split.
"""

import os
import sys
import json
import logging
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.toxic_model import ToxicModel
from models.context_model import ContextModelWrapper
from models.fusion_model import ContextAwareFusionMLP, FEATURE_SCHEMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def prepare_feature_matrix(
    df: pd.DataFrame,
    toxic_model: ToxicModel,
    context_model: ContextModelWrapper
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes model signals and extracts ordered feature vectors based on FEATURE_SCHEMA.
    """
    X_list = []
    y_list = []

    for _, row in df.iterrows():
        text = str(row["text"])
        ctx_snippet = str(row.get("context_snippet", ""))

        # Model signals
        p_tox = toxic_model.predict_p_toxic(text, use_real_model=toxic_model.ml_available)
        ctx_preds = context_model.predict(text, ctx_snippet)

        row_dict = dict(row)
        row_dict["p_toxic"] = p_tox
        row_dict["p_context"] = ctx_preds["p_context"]
        row_dict["is_targeted"] = ctx_preds["is_targeted"]
        row_dict["is_repeated"] = ctx_preds["is_repeated"]
        row_dict["is_escalating"] = ctx_preds["is_escalating"]
        row_dict["is_sarcastic"] = ctx_preds["is_sarcastic"]
        row_dict["is_threatening"] = ctx_preds["is_threatening"]

        vector = [float(row_dict.get(fname, 0.0) or 0.0) for fname in FEATURE_SCHEMA]
        X_list.append(vector)
        y_list.append(float(row["cyberbullying"]))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def train_fusion_model(
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 0.005,
    seed: int = 42
):
    print("=======================================================")
    print("TRAINING LEARNED CONTEXT-AWARE FUSION NETWORK (Phase 2)")
    print("=======================================================")

    torch.manual_seed(seed)
    np.random.seed(seed)

    csv_path = os.path.join(os.path.dirname(__file__), "context_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Context dataset missing at {csv_path}")

    df = pd.read_csv(csv_path)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Dataset partition: {len(df_train)} train, {len(df_val)} val, {len(df_test)} test.")

    weights_dir = os.path.join(os.path.dirname(__file__), "..", "models", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    context_weights = os.path.join(weights_dir, "context_submodel.pt")

    toxic_model = ToxicModel()
    context_model = ContextModelWrapper(weights_path=context_weights if os.path.exists(context_weights) else None)

    print("Extracting feature matrices for Train split...")
    X_train_raw, y_train = prepare_feature_matrix(df_train, toxic_model, context_model)
    print("Extracting feature matrices for Val split...")
    X_val_raw, y_val = prepare_feature_matrix(df_val, toxic_model, context_model)
    print("Extracting feature matrices for Test split...")
    X_test_raw, y_test = prepare_feature_matrix(df_test, toxic_model, context_model)

    # -------------------------------------------------------------
    # Feature Scaling - STRICTLY FIT ON TRAINING DATA ONLY (NO LEAKAGE)
    # -------------------------------------------------------------
    print("Fitting feature scaler strictly on training set...")
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    std = np.where(std < 1e-6, 1.0, std) # Prevent division by zero

    scaler_dict = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_names": FEATURE_SCHEMA,
        "input_dim": len(FEATURE_SCHEMA)
    }

    scaler_path = os.path.join(weights_dir, "feature_scaler.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"Saved feature scaler to: {scaler_path}")

    # Standardize matrices
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    # DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(FEATURE_SCHEMA)
    model = ContextAwareFusionMLP(input_dim=input_dim, dropout_rate=0.2).to(device)

    # Balanced class weights for BCE loss
    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / max(1.0, sum(y_train))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    best_val_f1 = -1.0
    best_weights_path = os.path.join(weights_dir, "fusion_mlp.pt")
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(batch_y.cpu().numpy())

        val_preds_binary = (np.array(val_preds) >= 0.50).astype(int)
        val_f1 = f1_score(val_targets, val_preds_binary, zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds_binary)

        if epoch % 5 == 0 or epoch == epochs or val_f1 > best_val_f1:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = {
                "epoch": epoch,
                "val_f1": round(float(val_f1) * 100, 2),
                "val_accuracy": round(float(val_acc) * 100, 2)
            }
            torch.save(model.state_dict(), best_weights_path)

    # -------------------------------------------------------------
    # Held-Out Test Evaluation
    # -------------------------------------------------------------
    print("\nEvaluating best checkpoint on held-out test split...")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    with torch.no_grad():
        test_logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    test_preds_binary = (test_probs >= 0.50).astype(int)
    test_acc = accuracy_score(y_test, test_preds_binary)
    test_f1 = f1_score(y_test, test_preds_binary, zero_division=0)
    test_prec = precision_score(y_test, test_preds_binary, zero_division=0)
    test_rec = recall_score(y_test, test_preds_binary, zero_division=0)
    test_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.5
    test_pr_auc = average_precision_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.5

    test_results = {
        "accuracy": round(float(test_acc) * 100, 2),
        "f1_score": round(float(test_f1) * 100, 2),
        "precision": round(float(test_prec) * 100, 2),
        "recall": round(float(test_rec) * 100, 2),
        "auroc": round(float(test_auc), 4),
        "auprc": round(float(test_pr_auc), 4)
    }

    meta_path = os.path.join(weights_dir, "fusion_model_metadata.json")
    metadata = {
        "version": "fusion-mlp-v1",
        "input_dim": input_dim,
        "feature_schema": FEATURE_SCHEMA,
        "best_epoch": best_metrics["epoch"],
        "validation_metrics": best_metrics,
        "held_out_test_metrics": test_results,
        "weights_file": best_weights_path,
        "scaler_file": scaler_path
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Test Set Performance:")
    print(f"  Accuracy:  {test_results['accuracy']}%")
    print(f"  F1-Score:  {test_results['f1_score']}%")
    print(f"  Precision: {test_results['precision']}%")
    print(f"  Recall:    {test_results['recall']}%")
    print(f"  AUROC:     {test_results['auroc']}")
    print(f"  AUPRC:     {test_results['auprc']}")
    print(f"\nFusion MLP weights saved: {best_weights_path}")
    return best_weights_path

if __name__ == "__main__":
    train_fusion_model()
