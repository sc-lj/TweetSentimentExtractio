# -*- encoding: utf-8 -*-

import torch
from models import SentSegModel
from dataLoader import SentiSegDataLoader, collate_fn
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import logging

logging.basicConfig(datefmt="%Y-%m-%d %H:%M:%S",
                    format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO,
                    filename="./segmodel.log")

sentiment = {'neutral': 0, 'negative': 1, 'positive': 2}


def get_span_index(text, span):
    if text.lower().find(span.lower().strip()) == -1:
        print(text+"\t"+span)
        raise
    start_index = text.lower().find(span.lower().strip())
    end_index = start_index + len(span)
    return start_index, end_index


def read_data(files):
    train_data = pd.read_csv(files)
    train_data = train_data.dropna()
    textes = train_data["text"].values
    spanes = train_data['selected_text'].values
    senti_label = train_data['sentiment'].values
    senti_label = [sentiment[i.strip()] for i in senti_label]
    span_label = []
    for text, span in zip(*(textes, spanes)):
        start_index, end_index = get_span_index(text, span)
        span_label.append((start_index, end_index))
    return textes, senti_label, span_label


def get_argparse():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--train_files", default="./data/train_data.csv", help="训练样本文件")
    args.add_argument(
        "--val_files", default="./data/val_data.csv", help="验证样本文件")
    args.add_argument("--epochs", default=100, help="训练轮数")
    args.add_argument("--learning_rate", default=1e-5, help="学习率")
    args.add_argument("--batch_size", default=10,
                      help="the number of every batch")
    args.add_argument("--pretrain_path",
                      default="./roberta_english", help="预训练模型路径")
    args.add_argument("--val_step", default=300, help="间隔多少步进行模型验证")
    args.add_argument(
        "--save_name", default="./models/best_seg_model", help="保存的模型文件名")

    return args.parse_args()


def test_predict(model, data):
    input_ids = data['input_id']
    mask_ids = data['mask_id']
    lengthes = data['length']
    tokens = data['token']
    all_hidden_states = model.get_encoded_text(input_ids, mask_ids)
    senti_logit = model.get_sent_predict(all_hidden_states)
    sent_label = senti_logit.argmax(-1)
    span_logits = model.extract_span(sent_label, all_hidden_states)
    batch_size = span_logits.shape[0]
    batch_span_token = []
    for i in range(batch_size):
        token = tokens[i]
        span_index = np.where(span_logits[i, :].cpu() > 0.5)[0]
        if len(span_index) <= 1:
            pred_token = token[1:lengthes[i]-1]
        else:
            pred_token = token[span_index[0]:span_index[1]]
        out_string = " ".join(pred_token).replace(" ##", "").strip()
        batch_span_token.append(out_string)
    sent_label = sent_label.cpu().numpy().tolist()
    return sent_label, batch_span_token


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def validation(val_dataloader, model, device):
    model.eval()
    sent_pred_result = []
    sent_target_result = []
    span_pred_result = []
    span_target_result = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="validation"):
            new_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    new_batch[k] = v.to(device)
                else:
                    new_batch[k] = v
            batch = new_batch
            sent_label, batch_span_indexes = test_predict(model, batch)
            sent_target_result.extend(
                batch['sent_label'].cpu().numpy().tolist())
            sent_pred_result.extend(sent_label)

            span_target_result.extend(batch['gold_token'])
            span_pred_result.extend(batch_span_indexes)

    sentiment_acc = accuracy_score(sent_target_result, sent_pred_result)
    sentiment_recall = recall_score(sent_target_result, sent_pred_result)
    sentiment_f1 = f1_score(sent_target_result, sent_pred_result)

    span_correct = set(span_target_result) & set(span_pred_result)
    span_acc = span_correct/len(span_pred_result)

    jaccard_result = []
    for pred, target in zip(*(span_pred_result, span_target_result)):
        jaccard_result.append(jaccard(pred, target))
    jaccardcorr = sum(jaccard_result)/len(jaccard_result)
    return {"sentiment_acc": sentiment_acc, "sentiment_recall": sentiment_recall, "sentiment_f1": sentiment_f1, "span_acc": span_acc, "jaccard": jaccardcorr}


def train():
    args = get_argparse()
    train_data, train_senti_label, train_span_label = read_data(
        args.train_files)
    val_data, val_senti_label, val_span_label = read_data(args.val_files)

    tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_path)
    train_dataset = SentiSegDataLoader(
        train_data, train_senti_label, train_span_label, tokenizer)
    val_dataset = SentiSegDataLoader(
        val_data, val_senti_label, val_span_label, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(
        val_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    device = torch.device("cuda")
    model = SentSegModel.from_pretrained(args.pretrain_path)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.8},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=1e-8)
    step = 0
    best_metrics = 0
    all_metrics = []

    for epoch in range(args.epochs):
        model.train()
        logging.info(f"epoch {epoch}")
        pbar = tqdm(train_dataloader, desc="training")
        for batch in pbar:
            new_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    new_batch[k] = v.to(device)
                else:
                    new_batch[k] = v
            batch = new_batch
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix(loss="%.5f" % loss.cpu().detach().numpy())
            step += 1
            if (step) % int(args.val_step) == 0:
                metrics = validation(val_dataloader, model, device)
                logging.info("模型的评估指标：%s" % ("%s" % metrics))
                if metrics['jaccard'] > best_metrics:
                    torch.save(model, args.save_name+".pt")
                    best_metrics = metrics['jaccard']
                    logging.info("save model to %s,metrics :%s" %
                                 (args.save_name, "%s" % metrics))

                all_metrics.append(metrics)

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv("./models/metrics.csv", index=False)


if __name__ == "__main__":
    train()
