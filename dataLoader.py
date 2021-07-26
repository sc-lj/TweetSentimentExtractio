# -*- encoding: utf-8 -*-

from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import torch


class SentiSegDataLoader(Dataset):
    def __init__(self, data, label, span_label, gold_span, tokenizer: RobertaTokenizer):
        super().__init__()

        self.data = data
        self.label = label
        self.span_label = span_label
        self.gold_span = gold_span
        self.tokenizer = tokenizer
        self.cls_special_id = tokenizer.cls_token_id
        self.cls_special = tokenizer.cls_token
        self.sep_special_id = tokenizer.sep_token_id
        self.sep_special = tokenizer.sep_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        sent_label = self.label[index]
        span_label = self.span_label[index]
        token_ids, convert_span_label, length, tokens, gold_token = self.tokenizer_encode(line, span_label)
        return token_ids, convert_span_label, sent_label, length, tokens, gold_token

    def tokenizer_encode(self, text, span_label):
        token_ids = [self.cls_special_id]
        convert_span_label = []
        tokens = [self.cls_special]

        start_span_index, end_span_index = span_label

        first_span_tokens = self.tokenizer.tokenize(text[:start_span_index])
        first_span_token_id = self.tokenizer.convert_tokens_to_ids(first_span_tokens)
        tokens += first_span_tokens
        token_ids += first_span_token_id
        convert_span_label.append(len(token_ids))

        middle_span_token = self.tokenizer.tokenize(text[start_span_index:end_span_index])
        middle_span_token_id = self.tokenizer.convert_tokens_to_ids(middle_span_token)
        tokens += middle_span_token
        token_ids += middle_span_token_id
        convert_span_label.append(len(token_ids))

        last_span_token = self.tokenizer.tokenize(text[end_span_index:])
        last_span_token_id = self.tokenizer.convert_tokens_to_ids(last_span_token)
        tokens += last_span_token
        token_ids += last_span_token_id
        token_ids += [self.sep_special_id]
        tokens += [self.sep_special]

        new_span_label = [0]*len(token_ids)
        for i in convert_span_label:
            new_span_label[i] = 1
        return token_ids, new_span_label, len(token_ids), tokens, middle_span_token


def collate_fn(batch):
    """[对输入的batch，转换成整个batch向量,并转换为统一的文本长度]

    Args:
        batch ([type]): [description]
    """

    token_ids, span_label, sent_label, lengthes, tokens, gold_tokens = zip(*(batch))
    max_length = max(map(len, token_ids))
    if max_length > 512:
        max_length = 512
    batch_size = len(token_ids)

    input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    mask_ids = torch.zeros((batch_size, max_length))
    span_labels = torch.zeros((batch_size, max_length), dtype=torch.float)
    sent_labels = torch.zeros(batch_size, dtype=torch.long)

    for i in range(batch_size):
        lenth = len(token_ids[i])
        input_ids[i, :lenth] = torch.tensor(token_ids[i], dtype=torch.long)
        mask_ids[i, :lenth] = 1
        span_labels[i, :lenth] = torch.tensor(span_label[i], dtype=torch.float32)
        sent_labels[i] = sent_label[i]

    return {"input_id": input_ids,
            "mask_id": mask_ids,
            "span_label": span_labels,
            "sent_label": sent_labels,
            "length": lengthes,
            "token": tokens,
            "gold_token": gold_tokens}
