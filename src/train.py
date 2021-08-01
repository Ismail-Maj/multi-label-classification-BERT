import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AdamW

from utils import get_data, get_metrics
from model import Model


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


def train(args):
    train_loader, valid_loader, num_labels = get_data(args.path, args.min_label_count, args.token_len, args.batch_size)

    model = Model(args.dropout, num_labels)
    model = nn.DataParallel(model)  # batch_size > num of GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model.named_parameters(), args.weight_decay, args.learning_rate)
    scheduler = (get_linear_schedule_with_warmup if args.scheduler == "linear" else get_cosine_schedule_with_warmup)(
        optimizer, 0, len(train_loader) * args.epochs
    )

    for epoch in range(1, args.epochs + 1):
        for stage in ["train", "valid"]:
            if stage == "valid" and epoch % args.validation_every != 0:
                continue

            model.train() if stage == "train" else model.eval()
            torch.set_grad_enabled(stage == "train")

            loader = tqdm(iter(train_loader if stage == "train" else valid_loader), desc=stage)
            preds = np.zeros(0, dtype=bool)
            targets = np.zeros(0, dtype=bool)
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

                total_loss += loss.item() * batch["input_ids"].shape[0]

                pred = torch.sigmoid(logits).detach().cpu().numpy().flatten() >= 0.5
                target = batch["target"].detach().cpu().numpy().flatten().astype(bool)

                preds = np.hstack([preds, pred])
                targets = np.hstack([targets, target])

                metrics = {"Loss": loss.item()}
                metrics.update(get_metrics(target, pred))

                loader.set_postfix(metrics)

            total_loss = total_loss / len(preds) * num_labels
            metrics = {"Loss": total_loss}
            metrics.update(get_metrics(targets, preds))
            print(
                " \t".join(
                    [f"{epoch}/{args.epochs}", stage] + [f"{key}:{value:.5f}" for (key, value) in metrics.items()]
                )
            )
        torch.save(model.state_dict(), args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="data")
    parser.add_argument("-t", "--token-len", type=int, default=128)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-e", "--epochs", type=int, default=120)
    parser.add_argument("-v", "--validation-every", type=int, default=5)
    parser.add_argument("-d", "--dropout", type=float, default=0.2)
    parser.add_argument("-l", "--learning-rate", type=float, default=5e-5)
    parser.add_argument("-w", "--weight-decay", type=float, default=1e-3)
    parser.add_argument("-m", "--model_path", type=str, default="model.pth")
    parser.add_argument("-s", "--scheduler", type=str, default="cosine")
    parser.add_argument("--min-label-count", type=int, default=10)
    print(vars(parser.parse_args()))
    train(parser.parse_args())
