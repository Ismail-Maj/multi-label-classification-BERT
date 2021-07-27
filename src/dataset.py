import torch
from transformers import CamembertTokenizer


class Dataset:

    tokenizer = CamembertTokenizer.from_pretrained(
        "camembert-base", do_lower_case=False
    )

    def __init__(self, X, y=None, token_len=256):
        self.X = X
        self.y = y
        self.token_len = token_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        tokens = Dataset.tokenizer.encode_plus(
            self.X[index],
            None,
            add_special_tokens=True,
            max_length=self.token_len,
            padding="max_length",
            truncation=True,
        )
        item = {
            "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokens["attention_mask"], dtype=torch.long),
        }
        if self.y is not None:
            item["target"] = self.y[index]
        return item
