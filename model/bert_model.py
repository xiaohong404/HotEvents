import torch.cuda
import torch.nn as nn
from transformers import BertForSequenceClassification


class BertHotModel(nn.Module):
    """
    BERT NER模型定义类
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = BertForSequenceClassification.from_pretrained(
            self.args.pretrain_model_path,
            num_labels=2,
            output_hidden_states=True
        )

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

