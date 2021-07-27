import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import accuracy_score, f1_score


from dataset import Dataset
from model import Model


def get_data(path, min_label_count):
    df = pd.read_parquet(path, engine="pyarrow")
    features = df.url.values
    targets = df.target

    labels_count = targets.explode().value_counts().loc[lambda x: x > min_label_count]
    labels = labels_count.index
    print(f"{len(labels)} labels")

    targets_sparse = torch.zeros(size=(len(features), len(labels)), dtype=torch.float)

    to_keep = []
    label_to_index = {label: index for index, label in enumerate(labels)}
    for i, target in enumerate(targets):
        for label in set(target) & label_to_index.keys():
            targets_sparse[i][label_to_index[label]] = 1.0
            if not to_keep or to_keep[-1] != i:
                to_keep.append(i)

    features = features[to_keep]
    targets_sparse = targets_sparse[to_keep]

    total_count = labels_count.sum()
    pos_weight = [np.sqrt((total_count - count) / count) for count in labels_count]

    return features, targets_sparse, pos_weight


def get_optimizer(named_parameters, weight_decay, learning_rate):
    param_optimizer = list(named_parameters)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_parameters, lr=learning_rate)


def train(
    path,
    token_len,
    min_label_count,
    batch_size,
    epochs,
    dropout,
    learning_rate,
    weight_decay,
    early_stopping_patience,
    model_path,
):
    X, y, pos_weight = get_data(path, min_label_count)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)

    train = Dataset(X_train, y_train, token_len)
    valid = Dataset(X_valid, y_valid, token_len)
    train_loader = DataLoader(train, batch_size=batch_size, pin_memory=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, pin_memory=True)

    model = Model(dropout=dropout, num_classes=y.shape[1])
    model = nn.DataParallel(model)  # batch_size > num of GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(device)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight)
    optimizer = get_optimizer(model.named_parameters(), weight_decay=weight_decay, learning_rate=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_training_steps=len(train_loader) * epochs)

    valid_loss_best = float("inf")
    es_patience = early_stopping_patience
    for epoch in range(epochs):

        for stage in ["train", "valid"]:
            model.train() if stage == "train" else model.eval()
            torch.set_grad_enabled(stage == "train")

            loader = tqdm(iter(train_loader if stage == "train" else valid_loader), desc=stage)
            preds = []
            targets = []
            total_loss = 0

            for batch in loader:
                for key in batch:
                    batch[key] = batch[key].to(device)
                if stage == "train":
                    optimizer.zero_grad()

                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = criterion(logits, batch["target"])
                if stage == "train":
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                pred = torch.sigmoid(logits).detach().cpu().numpy().flatten() >= 0.5
                target = batch["target"].detach().cpu().numpy().flatten()

                total_loss += loss.item() * batch["input_ids"].shape[0]
                preds.extend(pred.tolist())
                targets.extend(target.tolist())
                metrics = dict()
                metrics["Loss"] = loss.item()
                metrics["Hamming"] = accuracy_score(target, pred)
                metrics["F1-micro"] = f1_score(target, pred, zero_division=0)
                loader.set_postfix(metrics)

            total_loss = total_loss / len(train if stage == "train" else valid)

            print(
                f"{epoch}/{epochs} stage: {stage}\t Loss: {total_loss:.5f}\t"
                f"Hamming score: {accuracy_score(targets, preds):.5f}\t"
                f"F1 Micro: {f1_score(targets, preds, zero_division=0):.5f}"
            )

            if stage == "valid":
                if total_loss < valid_loss_best:
                    print("valid loss decreasing, saving model...")
                    torch.save(model.state_dict(), model_path)
                    es_patience = early_stopping_patience
                    valid_loss_best = total_loss
                else:
                    es_patience -= 1
                    if es_patience == 0:
                        print("early stopping")
                        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="data")
    parser.add_argument("-t", "--token-len", type=int, default=128)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-d", "--dropout", type=float, default=0.2)
    parser.add_argument("-l", "--learning-rate", type=float, default=5e-5)
    parser.add_argument("-w", "--weight-decay", type=float, default=1e-2)
    parser.add_argument("-m", "--model_path", type=str, default="model.pth")
    parser.add_argument("--min-label-count", type=int, default=100)  # prune labels
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    train(**vars(parser.parse_args()))
