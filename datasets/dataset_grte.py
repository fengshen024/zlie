import pytorch_lightning as pl
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

import json
import numpy as np

def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def mat_padding(inputs, length=None, padding=0):

    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[0] for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        pad_width[0] = (0, length - x.shape[0])
        pad_width[1] = (0, length - x.shape[0])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

class GRTEDataset(Dataset):
    def __init__(self, hparams, data_path):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.bert_path)
        self.data = json.load(open(data_path, 'r'))
        self.id2predicate = hparams.id2predicate
        self.predicate2id = hparams.predicate2id
        self.label2id = hparams.label2id
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        _ = self.tokenizer.encode_plus(data['text'])
        token_ids, mask = _['input_ids'], _['attention_mask']
        if 'triple_list' in data.keys():
            spoes = {}
            for s, p, o in data['triple_list']:
                s = self.tokenizer.encode_plus(s)['input_ids'][1:-1]
                p = self.predicate2id[p]
                o = self.tokenizer.encode_plus(o)['input_ids'][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
        if spoes:
            label = np.zeros([len(token_ids), len(token_ids), len(self.id2predicate)])  # LLR
            # label = ["N/A", "SMH", "SMT", "SS", "MMH", "MMT", "MSH","MST"]
            for s in spoes:
                s1, s2 = s
                for o1, o2, p in spoes[s]:
                    if s1 == s2 and o1 == o2:
                        label[s1, o1, p] = self.label2id["SS"]
                    elif s1 != s2 and o1 == o2:
                        label[s1, o1, p] = self.label2id["MSH"]
                        label[s2, o1, p] = self.label2id["MST"]
                    elif s1 == s2 and o1 != o2:
                        label[s1, o1, p] = self.label2id["SMH"]
                        label[s1, o2, p] = self.label2id["SMT"]
                    elif s1 != s2 and o1 != o2:
                        label[s1, o1, p] = self.label2id["MMH"]
                        label[s2, o2, p] = self.label2id["MMT"]

            mask_label = np.ones(label.shape)
            mask_label[0, :, :] = 0
            mask_label[-1, :, :] = 0
            mask_label[:, 0, :] = 0
            mask_label[:, -1, :] = 0
            return token_ids, mask, label, mask_label, data
        return token_ids, mask, None, None,  data
    
    @staticmethod
    def collate_fn(batch):
        # batch 是一个列表，包含了从数据集中获取的所有样本
        # 假设每个样本是一个字典，包含 'input_sequences' 和 'labels' 键
        # 没错，我太聪明了
        # 提取序列和标签
        batch_token_ids, batch_mask, batch_label, batch_mask_label, batch_ex = [], [], [], [], []
        for token_ids, mask, label, mask_label, d in batch:
            batch_token_ids.append(torch.tensor(token_ids))
            batch_mask.append(torch.tensor(mask))
            batch_label.append(torch.tensor(label))
            batch_mask_label.append(torch.tensor(mask_label))
            batch_ex.append(d)

        # 对序列进行填充，使它们具有相同的长度
        batch_token_ids = pad_sequence(batch_token_ids, batch_first=True, padding_value=0)
        batch_mask = pad_sequence(batch_mask, batch_first=True, padding_value=0)
        if batch_label is not None:
            batch_label, batch_mask_label = mat_padding(batch_label), mat_padding(batch_mask_label)
            batch_label = torch.tensor(batch_label)
            batch_mask_label = torch.tensor(batch_mask_label)
        return batch_token_ids, batch_mask, batch_label, batch_mask_label, batch_ex

class GRTEDataModule(pl.LightningDataModule):
    def __init__(self, hparams,batch_size=4):

        super().__init__()
        self.args = hparams
        self.batch_size = batch_size


    def setup(self, stage=None):
        if stage in (None, 'fit'):
        # 假设我们有一个函数来加载数据集
            self.train_dataset = GRTEDataset(self.args, self.args.data_path+ "train.json")
            self.val_dataset = GRTEDataset(self.args, self.args.data_path+ "dev.json")
        # 如果是测试阶段，加载测试数据集
        elif stage == 'test':
            self.test_dataset = GRTEDataset(self.args, self.args.data_path+ "test.json")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=4, collate_fn=GRTEDataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=4, collate_fn=GRTEDataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=4, collate_fn=GRTEDataset.collate_fn)