# -*- encoding: utf-8 -*-

from transformers import RobertaModel
from transformers import RobertaConfig
import torch.nn.functional as F
import torch.nn as nn
import torch
import random


class SentSegModel(RobertaModel):
    def __init__(self, config: RobertaConfig):
        super(SentSegModel, self).__init__(config, add_pooling_layer=False)
        self.init_weights()
        hidden_size = config.hidden_size
        self.senti_linear = nn.Linear(hidden_size, 3)  # 标签向量为3
        self.senti_vector = nn.Embedding(3, hidden_size)
        self.span_linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.5)

    def get_encoded_text(self, input_ids, mask_ids):
        output = super().forward(input_ids, attention_mask=mask_ids, output_hidden_states=True, return_dict=True)
        all_hidden_states = output['hidden_states'][1:]
        return all_hidden_states

    def get_sent_predict(self, hidden_states):
        first_hidden, last_hidden = hidden_states[0], hidden_states[-1]
        # hidden = (first_hidden+last_hidden)/2
        # [batch_size,hidden_size]
        hidden_mean = last_hidden.mean(1)
        # hidden_cls = last_hidden[:,0,:]
        hidden = self.dropout(hidden_mean)
        sent_logit = self.senti_linear(hidden)
        sent_logit = torch.softmax(sent_logit, -1)
        return sent_logit

    def forward(self, data):
        input_ids = data['input_id']
        mask_ids = data['mask_id']
        all_hidden_states = self.get_encoded_text(input_ids, mask_ids)
        sent_logit = self.get_sent_predict(all_hidden_states)

        sent_gold_label = data['sent_label']

        if random.random() < 0.5:
            sent_label = sent_logit.argmax(-1)
        else:
            sent_label = sent_gold_label

        span_logits = self.extract_span(sent_label, all_hidden_states)
        span_gold_label = data['span_label']

        sent_loss = F.cross_entropy(sent_logit, sent_gold_label)
        span_loss = F.binary_cross_entropy(span_logits, span_gold_label, reduce="none")
        span_loss = torch.sum(span_loss*mask_ids) / torch.sum(mask_ids)

        return sent_loss+span_loss

    def extract_span(self, label_ids, hidden_states):
        first_hidden, last_hidden = hidden_states[0], hidden_states[-1]
        text_encode = (first_hidden+last_hidden)/2
        # [batch_size,hidden_size]
        label_vector = self.senti_vector(label_ids)
        # [batch_size,1,hidden_size]
        label_vector = label_vector.unsqueeze(1)
        # [batch_size,1,hidden_size]
        # encode_label = torch.matmul(text_encode, label_vector)
        # [batch_size,seq_len,hidden_size]
        span_encode = text_encode + label_vector
        # [batch_size,seq_len,1]
        span_encode = self.span_linear(span_encode)
        span_encode = span_encode.squeeze(-1)
        span_logit = torch.sigmoid(span_encode)
        return span_logit
