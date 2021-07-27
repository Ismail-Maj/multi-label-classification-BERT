import torch.nn as nn
from transformers import CamembertModel


class Model(nn.Module):
    def __init__(self, dropout, num_classes):
        super(Model, self).__init__()
        self.bert = CamembertModel.from_pretrained("camembert-base", return_dict=False)
        self.dropout = nn.Dropout(dropout)
        self.FC = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        _, x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(x)
        x = self.FC(x)
        return x
