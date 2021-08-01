import torch
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import WeightedRandomSampler

from dataset import Dataset


def get_data(path, min_label_count, token_len, batch_size):
    df = pd.read_parquet(path, engine="pyarrow")
    features = df.url.values
    targets = df.target
    labels_count = targets.explode().value_counts().loc[lambda x: x > min_label_count]
    labels = labels_count.index
    print(f"{len(labels)} labels")

    targets_sparse = torch.zeros(size=(len(features), len(labels)), dtype=torch.float)

    # one hot encoding and remove data with no labels
    to_keep = []
    label_to_index = {label: index for index, label in enumerate(labels)}
    for i, target in enumerate(targets):
        for label in set(target) & label_to_index.keys():
            targets_sparse[i][label_to_index[label]] = 1.0
            if not to_keep or to_keep[-1] != i:
                to_keep.append(i)
    features = features[to_keep]
    targets_sparse = targets_sparse[to_keep]

    split_index = int(len(features) * 0.8)
    X_train, X_valid = features[:split_index], features[split_index:]
    y_train, y_valid = targets_sparse[:split_index], targets_sparse[split_index:]

    # over-sample rare labels
    weights = torch.ones(len(y_train), dtype=torch.float)
    for i, target_sparse in enumerate(y_train):
        weights[i] = 1 / labels_count[target_sparse.nonzero().flatten().numpy()].min()
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train = Dataset(X_train, y_train, token_len)
    valid = Dataset(X_valid, y_valid, token_len)
    train_loader = DataLoader(train, batch_size=batch_size, pin_memory=True, sampler=sampler)
    valid_loader = DataLoader(valid, batch_size=batch_size, pin_memory=True)

    return train_loader, valid_loader, len(labels)


def get_metrics(targets, preds):
    # vectorized implementation faster than scikit-learn
    metrics = dict()
    metrics["accuracy"] = (targets == preds).mean()
    P = targets == 1
    N = targets == 0
    PP = preds == 1
    PN = preds == 0
    TP = (P & PP).sum()
    FP = (N & PP).sum()
    FN = (P & PN).sum()
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["F-measure"] = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    return metrics
