import os
import time
import math
import copy
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Importdata import LoadData
from model import STMultiHorizonNet

class SeqDataset(Dataset):
    def __init__(self, X_all, Y_all, input_len, out_len, start_idx, end_idx):
        super().__init__()
        self.X = X_all
        self.Y = Y_all
        self.in_len = input_len
        self.out_len = out_len
        self.start = start_idx
        self.end = end_idx

    def __len__(self):
        return max(0, self.end - self.start - self.in_len - self.out_len + 1)

    def __getitem__(self, idx):
        s = self.start + idx
        e = s + self.in_len
        y_e = e + self.out_len
        # X: (T, N, F) -> (in_len, N, F)
        # Y: (T, N)    -> (out_len, N)
        x = self.X[s:e]                  # (in_len, N, F)
        y = self.Y[e:y_e]                # (out_len, N)
        x = torch.from_numpy(x).float().unsqueeze(0).squeeze(0)
        y = torch.from_numpy(y).float().unsqueeze(0).squeeze(0)
        return x, y


def build_loaders(X_all, Y_all, input_len, out_len,
                  train_ratio=0.6, valid_ratio=0.2,
                  batch_size=8, num_workers=0, shuffle=True):
    T_total = X_all.shape[0]
    n_train = int(T_total * train_ratio)
    n_valid = int(T_total * (train_ratio + valid_ratio))
    ds_train = SeqDataset(X_all, Y_all, input_len, out_len, 0, n_train)
    ds_valid = SeqDataset(X_all, Y_all, input_len, out_len,
                          n_train - input_len - out_len + 1, n_valid)
    ds_test = SeqDataset(X_all, Y_all, input_len, out_len,
                         n_valid - input_len - out_len + 1, T_total)

    dl_train = DataLoader(ds_train, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          drop_last=False)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers,
                         drop_last=False)
    return dl_train, dl_valid, dl_test

def _pick_field(obj, names):
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
    else:
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
    raise KeyError(f"Missing fields {names} in LoadData")


def extract_from_loaddata(data):
    """
    X_all: (T, N, F)   Y_all: (T, N)   adj: (N, N)   coordinates: (N, 2)
    """
    X_all = _pick_field(data, ["X", "x", "X_all", "x_all"])
    Y_all = _pick_field(data, ["Y", "y", "Y_all", "y_all", "target"])
    adj = _pick_field(data, ["num", "adj", "A", "geo_adj", "distance", "adjacency"])
    coordinates = _pick_field(data, ["coordinates", "coors", "coords", "sites"])

    # to numpy
    if torch.is_tensor(X_all):
        X_all = X_all.cpu().numpy()
    if torch.is_tensor(Y_all):
        Y_all = Y_all.cpu().numpy()
    if torch.is_tensor(adj):
        adj = adj.cpu().numpy()
    if torch.is_tensor(coordinates):
        coordinates = coordinates.cpu().numpy()

    assert X_all.ndim == 3, f"X_all shape should be (T,N,F), got {X_all.shape}"
    assert Y_all.ndim == 2, f"Y_all shape should be (T,N), got {Y_all.shape}"
    return X_all, Y_all, adj, coordinates

def compute_metrics(y_true, y_pred):
    if y_true.ndim == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0
    return rmse, mae, r2, mape


def save_prediction_csv(y_true_flat, y_pred_flat,
                        path="pred_vs_true_mhop_gat.csv"):
    df = pd.DataFrame({
        "y_true": y_true_flat.reshape(-1),
        "y_pred": y_pred_flat.reshape(-1),
    })
    df.to_csv(path, index=False)
    print(f"CSV saved to: {os.path.abspath(path)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--input_len", type=int, default=48)
    parser.add_argument("--out_len", type=int, default=24)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_short_blocks", type=int, default=2)
    parser.add_argument("--num_long_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_trend_residual_split",action="store_true")

    parser.add_argument("--use_critical_paths", action="store_true")
    parser.add_argument("--correlation_threshold", type=float, default=0.6)
    parser.add_argument("--distance_threshold", type=float, default=100.0)
    parser.add_argument("--max_hops", type=int, default=4)
    parser.add_argument("--max_paths_to_expand", type=int,default=4)
    parser.add_argument("--fusion_alpha", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--tri_loss",action="store_true" )

    parser.add_argument("--device",type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    print("Using device:", device)

    data = LoadData(args.train_ratio, args.valid_ratio,
                    args.input_len, args.out_len)

    X_all = data.X if hasattr(data, "X") else data.X_all
    Y_all = data.Y if hasattr(data, "Y") else data.Y_all
    coordinates = data.coordinates
    adj = data._getAdj()

    if not isinstance(X_all, np.ndarray):
        X_all = np.asarray(X_all)
    if not isinstance(Y_all, np.ndarray):
        Y_all = np.asarray(Y_all)
    if isinstance(adj, torch.Tensor):
        adj = adj.detach().cpu().numpy()
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.asarray(coordinates)

    T_total, N, F = X_all.shape
    print(f"Loaded: X={X_all.shape}, Y={Y_all.shape}, N={N}, F={F}")

    train_n = int(T_total * args.train_ratio)
    pollution_data_train = Y_all[:train_n]  # (train_n, N)

    dl_train, dl_valid, dl_test = build_loaders(
        X_all, Y_all, args.input_len, args.out_len,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True
    )
    print("train/valid/test sizes:",
          len(dl_train.dataset),
          len(dl_valid.dataset),
          len(dl_test.dataset))

    geo_adj_tensor = torch.from_numpy(adj).float().to(device)
    model = STMultiHorizonNet(
        input_dim=F,
        hidden_feats=args.hidden_dim,
        out_len=args.out_len,
        geo_adj=geo_adj_tensor,
        input_seq_len=args.input_len,
        num_short_blocks=args.num_short_blocks,
        num_long_blocks=args.num_long_blocks,
        dropout=args.dropout,
        use_critical_paths=args.use_critical_paths,
        coordinates=coordinates,
        pollution_data=pollution_data_train,
        correlation_threshold=args.correlation_threshold,
        distance_threshold=args.distance_threshold,
        max_hops=args.max_hops,
        fusion_alpha=args.fusion_alpha,
        device=args.device,
        use_trend_residual_split=args.use_trend_residual_split,
        max_paths_to_expand=args.max_paths_to_expand
    ).to(device)

    if args.use_critical_paths:
        print(
            f"corr_thresh={args.correlation_threshold}, "
            f"dist_thresh={args.distance_threshold}, "
            f"max_hops={args.max_hops}, "
            f"max_paths_to_expand={args.max_paths_to_expand}"
        )
    print(f"use_trend_residual_split = {args.use_trend_residual_split}")

    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss(beta=1.0)

    best_val = float("inf")
    best_state = None
    patience_cnt = 0

    def run_one_epoch(loader, train=True, current_epoch=None, val_loader=None):
        model.train(train)
        total_loss = 0.0
        y_true_all, y_pred_all = [], []

        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)  # (B,T,N,F)
            Y_batch = Y_batch.to(device)  # (B,out_len,N)

            if train:
                optimizer.zero_grad()

            out_short, out_long, out_fused, alpha = model(
                X_batch,
                current_epoch=current_epoch,
                val_loader=val_loader if train else None
            )

            if args.tri_loss:
                loss = (0.3 * criterion(out_short, Y_batch)
                        + 0.3 * criterion(out_long, Y_batch)
                        + 0.4 * criterion(out_fused, Y_batch))
            else:
                loss = criterion(out_fused, Y_batch)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            y_true_all.append(Y_batch.detach().cpu())
            y_pred_all.append(out_fused.detach().cpu())

        y_true_all = torch.cat(y_true_all, dim=0).numpy()
        y_pred_all = torch.cat(y_pred_all, dim=0).numpy()
        rmse, mae, r2, mape = compute_metrics(y_true_all, y_pred_all)
        return total_loss / len(loader.dataset), (rmse, mae, r2, mape)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = run_one_epoch(
            dl_train, train=True,
            current_epoch=epoch,
            val_loader=dl_valid
        )
        val_loss, val_metrics = run_one_epoch(
            dl_valid, train=False,
            current_epoch=epoch
        )

        model.fusion.optimize_once(model, dl_valid, device)
        dt = time.time() - t0

        tr_rmse, tr_mae, tr_r2, tr_mape = train_metrics
        va_rmse, va_mae, va_r2, va_mape = val_metrics
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"RMSE={tr_rmse:.4f} MAE={tr_mae:.4f} "
            f"R2={tr_r2:.4f} MAPE={tr_mape:.2f}% | "
            f"val_loss={val_loss:.4f} "
            f"RMSE={va_rmse:.4f} MAE={va_mae:.4f} "
            f"R2={va_r2:.4f} MAPE={va_mape:.2f}% | "
            f"{dt:.1f}s"
        )

        monitor = val_loss
        if monitor + 1e-12 < best_val:
            best_val = monitor
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch}. "
                      f"Best val loss={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), "best_mhop_gat_24.pt")
        print("Saved best model to best_mhop_gat_24.pt")

    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for X_batch, Y_batch in dl_test:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            out_short, out_long, out_fused, alpha = model(X_batch)
            y_true_all.append(Y_batch.cpu())
            y_pred_all.append(out_fused.cpu())

    y_true_all = torch.cat(y_true_all, dim=0).numpy()
    y_pred_all = torch.cat(y_pred_all, dim=0).numpy()
    rmse, mae, r2, mape = compute_metrics(y_true_all, y_pred_all)
    print(f"[TEST] RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  MAPE={mape:.2f}%")

    save_prediction_csv(
        y_true_all.reshape(-1, y_true_all.shape[-1]),
        y_pred_all.reshape(-1, y_pred_all.shape[-1]),
        path="pred_vs_true_mhop_gat.csv"
    )


if __name__ == "__main__":
    main()
