import argparse
import json
from transformers import BertConfig

from datasets.dataset_grte import GRTEDataModule


def get_labels():
    # label
    label_list = ["N/A", "SMH", "SMT", "SS", "MMH", "MMT", "MSH", "MST"]

    id2label, label2id = {}, {}
    for i, l in enumerate(label_list):
        id2label[str(i)] = l
        label2id[l] = i
    return id2label, label2id

def get_arguments():
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--bert_path', default='./weights/bert-base-cased', type=str, help="")
    parser.add_argument('--data_path', default='data/NYT24/')
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--warmup', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--num_train_epochs', default=20, type=int)
    parser.add_argument('--min_num', default=1e-7, type=float)
    args = parser.parse_args()
    args.id2predicate, args.predicate2id = json.load(open(args.data_path + 'rel2id.json'))
    args.id2label, args.label2id = get_labels()
    config = BertConfig.from_pretrained(args.bert_path)
    config.num_p = len(args.id2predicate)
    config.num_label = len(args.id2label)
    config.rounds = 4
    args.config = config
    dataModule = GRTEDataModule(args)
    dataModule.setup()
    args.t_total = len(dataModule.train_dataloader()) * args.num_train_epochs
    print(args)
    return args, dataModule