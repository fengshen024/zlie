import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
import numpy as np
from transformers.modeling_bert import BertModel, BertAttention, BertIntermediate, BertOutput
from transformers import BertTokenizer, BertConfig

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


class Grte(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.args = hparams
        self.tokenizer = BertTokenizer.from_pretrained(hparams.bert_path)
        self.config = hparams.config

        self.bert=BertModel(config=self.config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.Lr_e1=nn.Linear(self.config.hidden_size,self.config.hidden_size)
        self.Lr_e2=nn.Linear(self.config.hidden_size,self.config.hidden_size)

        self.elu=nn.ELU()
        self.Cr = nn.Linear(self.config.hidden_size, self.config.num_p*self.config.num_label)

        self.Lr_e1_rev=nn.Linear(self.config.num_p*self.config.num_label,self.config.hidden_size)
        self.Lr_e2_rev=nn.Linear(self.config.num_p*self.config.num_label,self.config.hidden_size)

        self.rounds=self.config.rounds

        self.e_layer=DecoderLayer(self.config)

        torch.nn.init.orthogonal_(self.Lr_e1.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e2.weight, gain=1)
        torch.nn.init.orthogonal_(self.Cr.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e1_rev.weight, gain=1) # 初始化神经网络的权重，使得初始化后的权重矩阵是正交的, 帮助避免梯度消失或梯度爆炸的问题
        torch.nn.init.orthogonal_(self.Lr_e2_rev.weight, gain=1)
    
    def forward(self, token_ids, mask_token_ids):
        embed=self.get_embed(token_ids, mask_token_ids)
        #embed:BLH
        L=embed.shape[1]

        e1 = self.Lr_e1(embed) # BLL H 主体特征，
        e2 = self.Lr_e2(embed) #  客体特征

        for i in range(self.rounds):
            h = self.elu(e1.unsqueeze(2).repeat(1, 1, L, 1) * e2.unsqueeze(1).repeat(1, L, 1, 1))  # BLL 2H
            B, L = h.shape[0], h.shape[1]

            table_logist = self.Cr(h)  # BLL RM

            if i!=self.rounds-1:

                table_e1 = table_logist.max(dim=2).values
                table_e2 = table_logist.max(dim=1).values
                e1_ = self.Lr_e1_rev(table_e1)
                e2_ = self.Lr_e2_rev(table_e2)

                e1=e1+self.e_layer(e1_,embed,mask_token_ids)[0]
                e2=e2+self.e_layer(e2_,embed,mask_token_ids)[0]

        return table_logist.reshape([B,L,L,self.config.num_p,self.config.num_label])

    def get_pred_id(self, table, all_tokens, label2id):

        B, L, _, R, _ = table.shape

        res = []
        for i in range(B):
            res.append([])

        table = table.argmax(axis=-1)  # BLLR

        all_loc = np.where(table != label2id["N/A"])

        res_dict = []
        for i in range(B):
            res_dict.append([])

        for i in range(len(all_loc[0])):
            token_n = len(all_tokens[all_loc[0][i]])

            if token_n - 1 <= all_loc[1][i] \
                    or token_n - 1 <= all_loc[2][i] \
                    or 0 in [all_loc[1][i], all_loc[2][i]]:
                continue

            res_dict[all_loc[0][i]].append([all_loc[1][i], all_loc[2][i], all_loc[3][i]])

        for i in range(B):
            for l1, l2, r in res_dict[i]:
                if table[i, l1, l2, r] == label2id["SS"]:
                    res[i].append([l1, l1, r, l2, l2])
                elif table[i, l1, l2, r] == label2id["SMH"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "SMT"] and l1_ == l1 and l2_ > l2:
                            res[i].append([l1, l1, r, l2, l2_])
                            break
                elif table[i, l1, l2, r] == label2id["MMH"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "MMT"] and l1_ > l1 and l2_ > l2:
                            res[i].append([l1, l1_, r, l2, l2_])
                            break
                elif table[i, l1, l2, r] == label2id["MSH"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "MST"] and l1_ > l1 and l2_ == l2:
                            res[i].append([l1, l1_, r, l2, l2_])
                            break
        return res

    def get_embed(self,token_ids, mask_token_ids):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long())
        embed=bert_out[0]
        embed=self.dropout(embed)
        return embed #(sequence_output, pooled_output) + encoder_outputs[1:]
        
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        batch_token_ids, batch_mask, batch_label, batch_mask_label, batch_ex = batch
        output = self.forward(batch_token_ids, batch_mask)
        crossentropy = nn.CrossEntropyLoss(reduction="none")
        output = output.reshape(-1, self.config.num_label)
        label = batch_label.reshape([-1])
        loss = crossentropy(output, label.long())
        loss = (loss * batch_mask_label.reshape([-1])).sum()
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        batch_token_ids, batch_mask, batch_label, batch_mask_label, batch_ex = batch
        output = self.forward(batch_token_ids, batch_mask)
        output = output.cpu().detach().numpy()
        

        all_tokens = []
        for ex in batch_ex:
            tokens = self.tokenizer.tokenize(ex["text"])
            all_tokens.append(tokens)

        # res_id = self.get_pred_id(output, all_tokens, self.args.label2id)
        res_id = []
        batch_spo = [[] for _ in range(len(batch))]

        for b, ex in enumerate(batch_ex):
            text = ex["text"]
            tokens = all_tokens[b]
            mapping = self.tokenizer.rematch(text, tokens)
            for sh, st, r, oh, ot in res_id[b]:
                s = (mapping[sh][0], mapping[st][-1])
                o = (mapping[oh][0], mapping[ot][-1])

                batch_spo[b].append(
                    (text[s[0]:s[1] + 1], self.id2predicate[str(r)], text[o[0]:o[1] + 1])
                )
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for i, ex in enumerate(batch_ex):
            R = set(batch_spo[i])
            T = set([(item[0], item[1], item[2]) for item in ex['triple_list']])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
        
        return {'X':X, 'Y':Y, 'Z':Z}
    
    def validation_epoch_end(self, outputs):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for item in outputs:
            X += item['X']
            Y += item['Y']
            Z += item['Z']
        precision, recall, f1 = 2 * X / (Y + Z), X / Y, X / Z
        self.log('precision', precision, prog_bar=True)
        self.log('recall', recall, prog_bar=True)
        self.log('f1', f1, prog_bar=True)
        
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.min_num)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup * self.args.t_total, num_training_steps=self.args.t_total
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


