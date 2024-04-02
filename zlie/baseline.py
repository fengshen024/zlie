from abc import ABC
from typing import Optional
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, EPOCH_OUTPUT, STEP_OUTPUT
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig, get_linear_schedule_with_warmup, AdamW
from transformers.modeling_bert import BertAttention, BertIntermediate, BertOutput

def span_metric(label_ids, predictions):
    head_labels, tail_labels, span_labels = label_ids
    head_preds, tail_preds, span_preds = predictions
    head_acc, head_recall, head_f1, _ = precision_recall_fscore_support(
        y_pred=head_preds.reshape(-1),
        y_true=head_labels.reshape(-1),
        labels=[1],
        zero_division=1,
        average='micro')
    tail_acc, tail_recall, tail_f1, _ = precision_recall_fscore_support(
        y_pred=tail_preds.reshape(-1),
        y_true=tail_labels.reshape(-1),
        labels=[1],
        zero_division=1,
        average='micro')
    span_acc, span_recall, span_f1, _ = precision_recall_fscore_support(
        y_pred=span_preds.reshape(-1),
        y_true=span_labels.reshape(-1),
        labels=[1],
        zero_division=1,
        average='micro')
    return {
        "head_acc": head_acc,
        "head_recall": head_recall,
        "head_f1": head_f1,
        "tail_acc": tail_acc,
        "tail_recall": tail_recall,
        "tail_f1": tail_f1,
        "span_acc": span_acc,
        "span_recall": span_recall,
        "span_f1": span_f1,
    }


class Baseline(pl.LightningModule, ABC):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        if self.config.bertconfig is not None:
            self.bert = BertModel.from_pretrained(config=self.config.bertconfig)
        else:
            self.bert = BertModel.from_pretrained(self.config.bert_path)
        
        self.rounds = self.config.rounds
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.elu=nn.LeakyReLU()


        hidden_size = self.bert.config.hidden_size
        self.hr = nn.Linear(hidden_size, hidden_size)
        self.tr = nn.Linear(hidden_size, hidden_size)
        self.Cr = nn.Linear(hidden_size, config.num_rels * 2)
        self.hr_rev = nn.Linear(config.num_rels * 2, hidden_size)
        self.tr_rev = nn.Linear(config.num_rels * 2, hidden_size)
        self.e_layer=DecoderLayer(self.bert.config)
        ### init weights
        torch.nn.init.orthogonal_(self.hr.weight, gain=1)
        torch.nn.init.orthogonal_(self.tr.weight, gain=1)
        torch.nn.init.orthogonal_(self.Cr.weight, gain=1)
        torch.nn.init.orthogonal_(self.hr_rev.weight, gain=1)
        torch.nn.init.orthogonal_(self.tr_rev.weight, gain=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, output_attentions=True,
                            return_dict=True)  # last_hidden_state, pooler_output, hidden_states, attentions, cross_attentions
        attentions_scores = outputs.attentions[-1]  # (batch_size, num_heads, sequence_length, sequence_length)
        last_hidden_state = self.dropout(outputs.last_hidden_state)
        span_logits = attentions_scores[:, :, :, :].mean(1)  # (batch_size, sequence_length, sequence_length)
        b, s, h  = last_hidden_state.shape
        head_logits, tail_logits, = self.hr(last_hidden_state), self.tr(last_hidden_state)

        for i in range(self.rounds):
            h = self.elu(head_logits.unsqueeze(2).repeat(1, 1, s, 1) * tail_logits.unsqueeze(1).repeat(1, s, 1, 1))  # BLL 2H
            B, L = h.shape[0], h.shape[1]

            table_logist = self.Cr(h)  # B L L R

            if i!=self.rounds-1:

                table_h = table_logist.max(dim=2).values
                table_t = table_logist.max(dim=1).values
                h_ = self.hr_rev(table_h)
                t_ = self.tr_rev(table_t)

                head_logits=head_logits+self.e_layer(h_,last_hidden_state,attention_mask)[0] # B S H
                tail_logits=tail_logits+self.e_layer(t_,last_hidden_state,attention_mask)[0]   

        table_logist = table_logist.permute(0, 3, 1, 2)  # (batch_size, rel_num, sequence_length, hidden_size)
        head_logits = table_logist[:, :self.config.num_rels, :, :]
        tail_logits = table_logist[:, self.config.num_rels:, :, :]

        return head_logits, tail_logits, span_logits  # (batch_size, rel_num, sequence_length, sequence_length)

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, token_type_ids = batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        head_label, tail_label, span_label = batch["head_label"], batch["tail_label"], batch["span_label"],
        head_logits, tail_logits, span_logits = self.forward(input_ids, attention_mask, token_type_ids)
        loss_fun = nn.BCEWithLogitsLoss()
        """
        在Pytorch进行BCELoss的时候，需要输入值都在[0, 1]之间，如果你的网络的最后一层不是sigmoid，你需要把BCELoss换成BCEWithLogitsLoss，这样损失函数会替你做Sigmoid的操作。
        """
        head_loss = loss_fun(head_logits.float().reshape(-1), head_label.reshape(-1).float())
        tail_loss = loss_fun(tail_logits.float().reshape(-1), tail_label.reshape(-1).float())
        span_loss = loss_fun(span_logits.float().reshape(-1), span_label.reshape(-1).float())
        # self.log("head_loss", head_loss, on_epoch=True, prog_bar=True, batch_size=4)
        # self.log("tail_loss", tail_loss, on_epoch=True, prog_bar=True, batch_size=4)
        # self.log("span_loss", span_loss, on_epoch=True, prog_bar=True, batch_size=4)
        loss = head_loss + tail_loss + span_loss
        # self.log("total_loss", loss, prog_bar=True)
        self.log('loss', loss, on_epoch=True, logger=True, batch_size=self.config.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        head_label, tail_label, span_label = batch["head_label"].cpu().numpy(), batch["tail_label"].cpu().numpy(), \
        batch["span_label"].cpu().numpy(),
        head_logits, tail_logits, span_logits = self.forward(batch["input_ids"], batch["attention_mask"],
                                                             batch["token_type_ids"])
        head_logits, tail_logits, span_logits = head_logits.cpu().numpy() > self.config.threshold, tail_logits.cpu().numpy() > self.config.threshold, span_logits.cpu().numpy() > self.config.threshold
        out = span_metric((head_label, tail_label, span_label), (head_logits, tail_logits, span_logits))
        return out

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        out = {key: 1e-10 for key in keys}
        for output in outputs:
            for key in keys:
                out[key] += output[key]

        length = len(outputs)
        for key in keys:
            out[key] /= length
        self.log('total_evaluate_result', out, prog_bar=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.min_num)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.warmup * self.config.t_total, num_training_steps=self.config.t_total
        )
        return ([optimizer, ], [scheduler, ])


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,  #B num_generated_triples H
        encoder_hidden_states,
        encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0] #hidden_states.shape
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :] # B 1 1 H
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0 #1 1 0 0 -> 0 0 -1000 -1000
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0] #B m H
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output) #B m H
        outputs = (layer_output,) + outputs
        return outputs
