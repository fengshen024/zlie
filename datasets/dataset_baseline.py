import json
import tqdm
import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class BaselineDataset(Dataset):

    def __init__(self, config, data_path):
        super().__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        self.data = self.load_data(data_path)
        self.max_len = self.config.max_len
        self.num_rels = self.config.num_rels
        self.pred2idx = self.config.pred2idx

    def __len__(self):
        return len(self.data)

    def load_data(self, path):
        data = json.load(open(path, 'r'))
        out = []
        for i in tqdm.tqdm(data):
            input_ids = self.tokenizer.encode(i['text'])
            if len(input_ids) <= self.config.max_len - 2:
                out.append(i)
        print("precess done~……~")
        return out

    def __getitem__(self, idx):
        data = self.data[idx]
        text = data["text"]
        inputs = self.tokenizer.encode_plus(text,
                                            max_length=self.max_len,
                                            padding='max_length',
                                            truncation=True)
        spo_list = set()
        spo_span_list = set()
        # [CLS] texts [SEP] rels
        head_matrix = np.zeros(
            [self.num_rels,self.max_len, self.max_len])
        tail_matrix = np.zeros(
            [self.num_rels,self.max_len, self.max_len])
        span_matrix = np.zeros(
            [self.max_len, self.max_len])

        for spo in data["relation_list"]:
            pred = spo["predicate"]
            sub = spo["subject"]
            obj = spo["object"]
            spo_list.add((sub, pred, obj))
            sub_span = spo["subj_tok_span"]
            obj_span = spo["obj_tok_span"]
            plus_token_pred_idx = self.pred2idx[pred]
            spo_span_list.add((tuple(sub_span), plus_token_pred_idx, tuple(obj_span)))

            h_s, h_e = sub_span
            t_s, t_e = obj_span
            # Interaction
            head_matrix[plus_token_pred_idx][h_s + 1][t_s + 1] = 1
            head_matrix[plus_token_pred_idx][t_s + 1][h_s + 1] = 1
            tail_matrix[plus_token_pred_idx][h_e][t_e] = 1
            tail_matrix[plus_token_pred_idx][t_e][h_e] = 1
            span_matrix[h_s + 1][h_e] = 1
            span_matrix[h_e][h_s + 1] = 1
            span_matrix[t_s + 1][t_e] = 1
            span_matrix[t_e][t_s + 1] = 1

        # head_label = torch.tensor(head_matrix, dtype=torch.long)
        # tail_label = torch.tensor(tail_matrix, dtype=torch.long)
        # span_label = torch.tensor(span_matrix, dtype=torch.long)
        # output = {
        #     'text': text,
        #     "spo_list": spo_list,
        #     "spo_span_list": spo_span_list,
        #     "head_label": head_matrix,
        #     "tail_label": tail_matrix,
        #     "span_label": span_matrix
        # }
        output = {
            'text': text,
            'input_ids': inputs["input_ids"],
            'attention_mask': inputs["attention_mask"],
            'token_type_ids': inputs["token_type_ids"],
            "head_label": head_matrix,
            "tail_label": tail_matrix,
            "span_label": span_matrix,
        }
        return output

    @staticmethod
    def collate_fn(batch):
        # batch 是一个列表，包含了从数据集中获取的所有样本
        outputs = {
            # 'text': [],
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            "head_label": [],
            "tail_label": [],
            "span_label": [],
        }
        for o in batch:
            # outputs["text"].append(o["text"])
            outputs["input_ids"].append(o["input_ids"])
            outputs["attention_mask"].append(o["attention_mask"])
            outputs["token_type_ids"].append(o["token_type_ids"])
            outputs["head_label"].append(o["head_label"])
            outputs["tail_label"].append(o["tail_label"])
            outputs["span_label"].append(o["span_label"])

        # for o in batch:
        #     outputs["text"].append(o["text"])
        #     outputs["input_ids"].append(torch.tensor(o["input_ids"], dtype=torch.long))
        #     outputs["attention_mask"].append(torch.tensor(o["attention_mask"], dtype=torch.long))
        #     outputs["token_type_ids"].append(torch.tensor(o["token_type_ids"], dtype=torch.long))
        #     outputs["head_label"].append(torch.tensor(o["head_label"], dtype=torch.long))
        #     outputs["tail_label"].append(torch.tensor(o["tail_label"], dtype=torch.long))
        #     outputs["span_label"].append(torch.tensor(o["span_label"], dtype=torch.long))
        for key in outputs.keys():
            if key == 'text':
                continue
            outputs[key] = torch.tensor(outputs[key], dtype=torch.long)
        return outputs

class BaselineDataModule(pl.LightningDataModule):
    def __init__(self, config):

        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.config = config
        self.batch_size = self.config.batch_size
        self.setup()

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            # 假设我们有一个函数来加载数据集
            self.train_dataset = BaselineDataset(self.config, os.path.join(self.config.data_path, "train_data.json"))
            self.val_dataset = BaselineDataset(self.config, os.path.join(self.config.data_path, "valid_data.json"))
        # 如果是测试阶段，加载测试数据集
        elif stage == 'test':
            self.test_dataset = BaselineDataset(self.config, os.path.join(self.config.data_path, "test_data.json"))

    # def train_dataloader(self):
    #     return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
    #                       num_workers=0)
    #
    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
    #                       num_workers=0)
    #
    # def test_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
    #                       num_workers=0)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=0, collate_fn=BaselineDataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=0, collate_fn=BaselineDataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=0, collate_fn=BaselineDataset.collate_fn)