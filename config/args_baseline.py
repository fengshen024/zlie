import argparse
import json
import os


def get_arguments():
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--bert_path', default='./weights/bert-base-cased', type=str, help="")
    parser.add_argument('--data_path', default='./data/tplinker/nyt')
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--min_num', default=1e-7, type=float)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--warmup', default=0.0, type=float)
    parser.add_argument('--max_len', default=102, type=int)
    parser.add_argument('--rounds', default=4, type=int)
    parser.add_argument('--num_train_epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--bertconfig', default=None, type=str)
    parser.add_argument('--hidden_dropout_prob', default=0.1, type=str)
    args = parser.parse_args()
    args.pred2idx = json.load(open(os.path.join(args.data_path, 'rel2id.json'), 'r'))
    args.num_rels = len(args.pred2idx)
    return args