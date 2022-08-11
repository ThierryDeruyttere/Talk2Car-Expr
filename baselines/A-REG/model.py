import operator  # allows us to make the queue of BeamSearchNode Nodes comparable
import argparse
import os
import sys
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet152
from expr_dataset import VGDataset, T2CDataset
from attention import Attention
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from collections import defaultdict
import numpy as np
from queue import PriorityQueue
import time
from transformers import BertTokenizer, BertModel, BertConfig
from reinforcer import Reinforcer

class BeamSearchNode(object):
    def __init__(self, hidden, previous_node, word_id, log_prob, length, is_start_node=False):
        """
        :param hidden:
        :param previous_node:
        :param word_id:
        :param log_prob:
        :param length:
        """
        self.hidden = hidden
        self.prev_node = previous_node
        self.word_id = word_id
        self.logp = log_prob
        self.length = length
        self.is_start_node = is_start_node

        if self.is_start_node:
            self.all_past_ids = self.word_id
        else:
            self.all_past_ids = torch.cat([self.prev_node.all_past_ids, self.word_id], dim=-1)

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.length + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.eval() < other.eval()


class HiddenOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def sort_batch(batch):
    if isinstance(batch, dict):
        if 'expression_length' not in batch:
            return batch
        return_dict = {}
        expr_len, sorting = batch['expression_length'][:, 0].sort(descending=True)
        for d_key, d_val in batch.items():
            if d_key == 'raw_sentence':
                return_dict[d_key] = [d_val[i] for i in sorting]
            else:
                if type(d_val) is list:

                    new_val = []
                    for v in d_val:
                        new_val.append(v[sorting])
                    return_dict[d_key] = new_val
                else:
                    return_dict[d_key] = d_val

        return return_dict
    else:
        gt_expr, expr_len, obj, img, box, attr_prob, pred_class, class_hot, image_id, region_id, switch_target = batch
        expr_len, sorting = expr_len.sort(descending=True)
        gt_expr = gt_expr[sorting]
        obj = obj[sorting]
        img = img[sorting]
        box = box[sorting]
        attr_prob = attr_prob[sorting]
        pred_class = pred_class[sorting]
        class_hot = class_hot[sorting]
        image_id = image_id[sorting]
        region_id = region_id[sorting]
        if (switch_target > 0).any():
            switch_target = switch_target[sorting]
        return gt_expr, expr_len, obj, img, box, attr_prob, pred_class, class_hot, image_id, region_id, switch_target


class AttributePredictors(nn.Module):

    def __init__(self, cls_embedding: nn.Module = None):
        super(AttributePredictors, self).__init__()
        self._net = resnet152(pretrained=True)
        self._net.fc = nn.Identity()
        hidden_dim = 2048
        self.n_actions = 11
        self.n_colors = 12
        if cls_embedding is None:
            self.obj_cls_emb_size = 300
            self.cls_embedding = cls_embedding
        else:
            self.obj_cls_emb_size = 512
            self.cls_embedding = nn.Embedding(23,  # Number of classes in nuScenes
                                              self.obj_cls_emb_size)

        # ACTION
        action_hidden_dim = hidden_dim
        # We will concat the image to the object
        action_hidden_dim *= 2
        # We will also use the object class
        action_hidden_dim += self.obj_cls_emb_size
        action_layers = [nn.Linear(action_hidden_dim, 1024),
                         nn.Linear(1024, self.n_actions)]
        self._action_fc = nn.Sequential(*action_layers)

        # COLOR
        color_layers = [nn.Linear(hidden_dim, 1024),
                        nn.Linear(1024, self.n_colors)]
        self._color_fc = nn.Sequential(*color_layers)

        # LOCATION
        self._location_fc = nn.Sequential(nn.Linear(3, 100),
                                          nn.ReLU(),
                                          nn.Linear(100, 3))

    def __extract_img_obj_feats(self, img, obj):
        obj_feats = self._net(obj)
        img_feats = self._net(img)
        concat_feats = torch.cat([obj_feats, img_feats], dim=1)
        return obj_feats, img_feats, concat_feats

    def forward(self, obj, img, box, cls=None):
        obj_feats, img_feats, concat_feats = self.__extract_img_obj_feats(img, obj)
        out = {'img_feats': img_feats,
               'obj_feats': obj_feats}

        # ACTION
        feats = obj_feats if concat_feats is None else concat_feats
        cls_feats = self.cls_embedding(cls).squeeze(1)
        feats = torch.cat([feats, cls_feats], dim=1)
        action_output = self._action_fc(feats)
        out["action"] = action_output

        # COLOR
        color_output = self._color_fc(obj_feats)
        out["color"] = color_output

        # LOCATION
        location_output = self._location_fc(box)
        out["location"] = location_output
        return out


class AttrExprGen(pl.LightningModule):
    def __init__(self, vocab, attr_vocab, class_vocab, hparams, rootdir=None):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters(self.hparams)
        self.best = defaultdict(lambda: 0)
        # self.best['val_loss'] = 99
        self.vocab = vocab
        self.attr_vocab = attr_vocab
        self.class_vocab = class_vocab
        self.rootdir = rootdir

        if hasattr(self.hparams, 'use_bert') and self.hparams.use_bert:
            with torch.no_grad():
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                # config = BertConfig.from_pretrained("bert-base-uncased")
                # config.is_decoder = True
                # self.bert_model = BertModelLMHeadModel.from_pretrained('bert-base-uncased', config=config)
                self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        # embeddings for words, attributes, and classes
        self.embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=self.hparams.embedding_size)
        # attribute and class embeddings only when needed
        if self.hparams.use_attr_att:
            self.attr_embs = nn.Embedding(num_embeddings=len(attr_vocab), embedding_dim=self.hparams.embedding_size)
        if self.hparams.use_class_mod or self.hparams.use_class_hot \
                or self.hparams.use_class_att or self.hparams.multi_task:
            self.class_embs = nn.EmbeddingBag(num_embeddings=len(class_vocab),
                                              embedding_dim=self.hparams.embedding_size)

        if self.hparams.multi_task:
            self.encoder = AttributePredictors(cls_embedding=self.class_embs)

        # fc for object features
        self.obj_feature_fc = nn.Linear(in_features=self.hparams.feature_size,
                                        out_features=self.hparams.feature_hidden)
        if self.hparams.use_img:
            self.img_feature_fc = nn.Linear(in_features=self.hparams.feature_size,
                                            out_features=self.hparams.feature_hidden)

        # create the attention layer if requested for the attributes
        if self.hparams.use_attr_att:
            self.attention = Attention(self.hparams.embedding_size, self.hparams.lstm_hidden,
                                       self.hparams.attention_hidden)

        # if a mapping is requested for the attribute/class (hot) vector
        if self.hparams.use_attr_hot == 'map':
            self.attr_map_fc = nn.Linear(in_features=len(self.attr_vocab),
                                         out_features=self.hparams.keywordmap_hidden)
        if self.hparams.use_class_hot == 'map':
            self.class_map_fc = nn.Linear(in_features=len(self.class_vocab),
                                          out_features=self.hparams.keywordmap_hidden)

        if hasattr(self.hparams, 'use_diff') and self.hparams.use_diff:
            self.diff_fc = nn.Linear(in_features=self.hparams.feature_size,
                                     out_features=self.hparams.feature_hidden)

        # create the first lstm layer
        input_size1 = self.hparams.lstm_hidden  # output h2
        input_size1 += self.hparams.embedding_size  # previous predicted word
        input_size1 += self.hparams.feature_hidden  # object feature vector
        input_size1 += self.hparams.feature_hidden if self.hparams.use_img else 0  # image feature
        input_size1 += len(self.attr_vocab) if self.hparams.use_attr_hot == 'True' else 0  # attr boolean hot vector
        input_size1 += self.hparams.keywordmap_hidden if self.hparams.use_attr_hot == 'map' else 0  # cls mapped vector
        input_size1 += len(self.class_vocab) if self.hparams.use_class_hot == 'True' else 0  # class one hot vector
        input_size1 += self.hparams.keywordmap_hidden if self.hparams.use_class_hot == 'map' else 0  # cls mapped vector
        input_size1 += (self.hparams.feature_hidden + 25) if hasattr(self.hparams, 'use_diff') and self.hparams.use_diff else 0  # diff feats + diff boxes
        self.lstm1 = nn.LSTMCell(input_size=input_size1, hidden_size=self.hparams.lstm_hidden)

        # Create the second lstm layer
        input_size2 = self.hparams.lstm_hidden  # output h1
        input_size2 += 4 if self.hparams.use_box else 0  # box location
        input_size2 += 6 if self.hparams.use_count else 0  # count vector, fixed size of 6
        input_size2 += self.hparams.embedding_size if self.hparams.use_attr_att else 0  # attribute_attention
        input_size2 += self.hparams.embedding_size if self.hparams.use_class_mod else 0  # class_embedding in model
        self.lstm2 = nn.LSTMCell(input_size=input_size2, hidden_size=self.hparams.lstm_hidden)

        # a gate for forcing attributes directly into the caption
        if self.hparams.use_force_attr:
            self.gate_layer_switch = nn.Sequential(nn.Linear(in_features=self.hparams.lstm_hidden, out_features=1),
                                                   nn.Sigmoid())
            self.gate_attr_out_layer = nn.Linear(in_features=self.hparams.lstm_hidden,
                                                 out_features=len(self.attr_vocab))
            # create a mapping vector, to go from the attribute indices to the vocab indices
            self.attr2voc_map = torch.tensor(
                [self.vocab.index(self.attr_vocab[idx]) for idx in range(len(self.attr_vocab))],
                dtype=torch.long, device=self.device)

        if hasattr(self.hparams, 'use_bert') and self.hparams.use_bert:
            self.vs = self.tokenizer.vocab_size
        else:
            self.vs = len(vocab)
        self.out_layer = nn.Linear(in_features=self.hparams.lstm_hidden, out_features=self.vs)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        if self.hparams.use_force_attr:
            if self.hparams.switch_loss == 'mse':
                self.switch_loss = nn.MSELoss(reduction='sum')
            elif self.hparams.switch_loss == 'l1':
                self.switch_loss = nn.L1Loss(reduction='sum')
            elif self.hparams.switch_loss == 'smooth_l1':
                self.switch_loss = nn.SmoothL1Loss(reduction='sum')
            elif self.hparams.switch_loss == 'ce':
                self.switch_loss = nn.CrossEntropyLoss(reduction='sum')
        if self.hparams.mmi_loss_weight > 0:
            self.logsoftmax = nn.LogSoftmax(dim=-1)
            self.mmi_loss = nn.MarginRankingLoss(margin=self.hparams.mmi_loss_margin, reduction='none')
        if self.hparams.multi_task:
            self.action_loss = nn.CrossEntropyLoss()
            self.color_loss = nn.CrossEntropyLoss()
            self.location_loss = nn.CrossEntropyLoss()
        # for quicker validation
        self.coco = None
        self.init_weights()

        if self.hparams.use_reinforcer:
            self.reinforcer = Reinforcer.load_from_checkpoint(self.hparams.use_reinforcer,
                                                              vocab=vocab,
                                                              attr_vocab=attr_vocab,
                                                              class_vocab=class_vocab,
                                                              hparams=self.hparams)
            self.reinforcer_ce_loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, reduction="none")
        # self.example_input_array = {
        #     'expr_ids': torch.zeros((2, 15), dtype=torch.long, device=self.device),
        #     'target': torch.zeros((2, 15), dtype=torch.long, device=self.device),
        #     'obj': torch.zeros((2, self.hparams.feature_hidden), dtype=torch.float, device=self.device),
        #     'box': torch.zeros((2, 4), dtype=torch.float, device=self.device),
        #     'attr_scr': torch.zeros((2, len(self.attr_vocab)), dtype=torch.float, device=self.device),
        #     'attr_ids': torch.zeros((2, 3), dtype=torch.long, device=self.device),
        #     'cls': torch.zeros((2, 1), dtype=torch.long, device=self.device),
        #     'cls_hot': torch.zeros((2, len(self.class_vocab)), dtype=torch.float, device=self.device),
        #     'img': torch.zeros((2, self.hparams.feature_hidden), dtype=torch.float, device=self.device),
        #     'expr_lens': torch.tensor([[15], [15]], dtype=torch.float, device=self.device),
        #     'switch_target': torch.zeros((2, 15), dtype=torch.float, device=self.device),
        #     'count': torch.zeros((2, 6), dtype=torch.float, device=self.device),
        #     'full_image': torch.zeros((2, 3, 224, 224), dtype=torch.float, device=self.device),
        #     'full_object': torch.zeros((2, 3, 224, 224), dtype=torch.float, device=self.device),
        #     'encoder_box': torch.zeros([2, 3], dtype=torch.float, device=self.device),
        #     'action_target': torch.tensor([[0], [0]], dtype=torch.long, device=self.device),
        #     'color_target': torch.tensor([[0], [0]], dtype=torch.long, device=self.device),
        #     'location_target': torch.tensor([[0], [0]], dtype=torch.long, device=self.device),
        #     'raw_sentence': [' '.join(['test']*15), ' '.join(['test']*15)]
        # }

    def init_weights(self):
        self.embedding.from_pretrained(embeddings=torch.tensor(self.vocab.weights),
                                       padding_idx=self.vocab.word2idx[self.vocab.pad_token])
        if self.hparams.use_attr_att:
            self.attr_embs.from_pretrained(embeddings=torch.tensor(self.attr_vocab.weights), freeze=True)
        if self.hparams.use_class_mod or self.hparams.use_class_att:
            self.class_embs.from_pretrained(embeddings=torch.tensor(self.class_vocab.weights), freeze=True)

    def init_hidden(self, bsz):
        h = torch.zeros(bsz, self.hparams.lstm_hidden, device=self.device)
        c = torch.zeros(bsz, self.hparams.lstm_hidden, device=self.device)
        return h, c

    def forward(self, expr_ids, target, obj, box, attr_scr, attr_ids, cls, cls_hot, img, count,
                expr_lens=None, start=None, switch_target=None, full_image=None, full_object=None,
                encoder_box=None, action_target=None, color_target=None, location_target=None,
                raw_sentence=None, diff_feats=None, diff_boxes=None, neg_sent_target=None):
        # prepare the input by collecting some numbers
        bsz = obj.size(0)
        bsz_divide = 3 if self.hparams.mmi_loss_weight > 0 else 1
        #bsz = self.hparams.train_batch_size*2

        # prepare expr length
        if self.training:
            assert expr_ids is not None, "when the model is training, the epxr_ids should be assigned"
            max_len = expr_ids.size(1)
        else:
            # bsz = obj.shape[0]
            max_len = self.hparams.max_length + 1  # add the 1 for the <EOS> token
            if hasattr(self.hparams, 'use_bert') and self.hparams.use_bert:
                raw_sentence = [self.tokenizer.cls_token]
            else:
                if start is not None:
                    expr_ids = start
                else:
                    expr_ids = torch.full((1,), fill_value=self.vocab.sos_idx,
                                          dtype=torch.long, device=self.device)

        # run attribute predictor
        if self.hparams.multi_task:
            assert full_image is not None and full_object is not None, \
                "When doing multi_task, the full image tensor should be given"
            assert encoder_box is not None, "When doing multi_task, the encoder box tensor should be given"
            assert self.hparams.max_attrs == 1, "It doesn't work with more than 1 attribute"
            encoded = self.encoder(obj=full_object, img=full_image, box=encoder_box, cls=cls)
            img = encoded['img_feats']
            obj = encoded['obj_feats']

            loc_idx = encoded['location'].argmax(dim=-1)
            col_idx = encoded['color'].argmax(dim=-1)
            act_idx = encoded['action'].argmax(dim=-1)
            voc_loc_idx = [self.attr_vocab.loc_id2vocab_id[idx.item()] for idx in loc_idx]
            voc_col_idx = [self.attr_vocab.col_id2vocab_id[idx.item()] for idx in col_idx]
            voc_act_idx = [self.attr_vocab.act_id2vocab_id[idx.item()] for idx in act_idx]
            attr_ids = torch.cat([torch.tensor(voc_loc_idx, dtype=torch.long, device=self.device).unsqueeze(1),
                                  torch.tensor(voc_col_idx, dtype=torch.long, device=self.device).unsqueeze(1),
                                  torch.tensor(voc_act_idx, dtype=torch.long, device=self.device).unsqueeze(1)], dim=-1)
            if self.hparams.attribute_prob2hot:
                loc_prob = torch.zeros((encoded['location'].shape[0], encoded['location'].shape[1]),
                                       dtype=torch.float, device=self.device)
                loc_prob[torch.arange(bsz), loc_idx] = 1
                col_prob = torch.zeros((encoded['color'].shape[0], encoded['color'].shape[1]),
                                       dtype=torch.float, device=self.device)
                col_prob[torch.arange(bsz), col_idx] = 1
                act_prob = torch.zeros((encoded['action'].shape[0], encoded['action'].shape[1]),
                                       dtype=torch.float, device=self.device)
                act_prob[torch.arange(bsz), act_idx] = 1
                attr_scr = torch.cat([loc_prob, col_prob, act_prob], dim=-1)
            else:
                attr_scr = torch.cat([torch.tensor(encoded['location'], dtype=torch.float, device=self.device),
                                      torch.tensor(encoded['color'], dtype=torch.float, device=self.device),
                                      torch.tensor(encoded['action'], dtype=torch.float, device=self.device)], dim=-1)

        if hasattr(self.hparams, 'use_bert') and self.hparams.use_bert:
            if self.training:
                tok_out = self.tokenizer(raw_sentence, padding=True, truncation=True,
                                         return_tensors="pt").to(self.device)
            else:
                tok_out = self.tokenizer(raw_sentence, padding=True, truncation=True, return_tensors="pt",
                                         is_split_into_words=True, add_special_tokens=False).to(self.device)
                expr_ids = tok_out['input_ids']
            bert_out = self.bert_model(tok_out['input_ids'])
            expr_embs = bert_out.last_hidden_state
        else:
            # cast the expression, attributes, and class to embeddings
            expr_embs = self.embedding(expr_ids)
            expr_embs = torch.dropout(expr_embs, p=self.hparams.dropout, train=self.training)

        # prepare attribute and class embeddings
        attr_embs = None
        attr_mask = None
        if self.hparams.use_attr_att:
            if self.hparams.old_attr:
                k = torch.max(torch.sum(attr_scr, dim=1)).to(torch.long).item()
                attr_vals, attr_ids = torch.topk(attr_scr, k=k, dim=1)
                attr_mask = attr_vals > 0
                if self.hparams.use_class_att:
                    attr_mask = torch.cat([attr_mask,
                                           torch.ones((bsz, 1), device=self.device, dtype=attr_mask.dtype)],
                                          dim=1)
            attr_embs = self.attr_embs(attr_ids)
            attr_embs = torch.dropout(attr_embs, p=self.hparams.dropout, train=self.training)
        class_embs = None
        if self.hparams.use_class_mod or self.hparams.use_class_att:
            class_embs = self.class_embs(cls)
            class_embs = torch.dropout(class_embs, p=self.hparams.dropout, train=self.training)
            if self.hparams.use_class_att:
                attr_embs = torch.cat([attr_embs, class_embs.unsqueeze(1)], dim=1)

        # cast input to correct size
        const_in1 = []
        obj = self.obj_feature_fc(obj)
        obj = torch.dropout(obj, p=self.hparams.dropout, train=self.training)
        const_in1.append(obj)
        if self.hparams.use_img:
            img = self.img_feature_fc(img)
            img = torch.dropout(img, p=self.hparams.dropout, train=self.training)
            const_in1.append(img)
        if self.hparams.use_attr_hot != 'False':
            if self.hparams.use_attr_hot == 'map':
                attr_hot = self.attr_map_fc(attr_scr)
                attr_hot = torch.dropout(attr_hot, p=self.hparams.dropout, train=self.training)
                const_in1.append(attr_hot)
            else:
                const_in1.append(attr_scr)
        if self.hparams.use_class_hot != 'False':
            if self.hparams.use_class_hot == 'map':
                cls_hot = self.class_map_fc(cls_hot)
                cls_hot = torch.dropout(cls_hot, p=self.hparams.dropout, train=self.training)
            const_in1.append(cls_hot)
        if hasattr(self.hparams, 'use_diff') and self.hparams.use_diff:
            assert diff_feats is not None and diff_boxes is not None, "when using diff, the features should be assigned"
            diff_feats = self.diff_fc(diff_feats)
            const_in1.append(diff_feats)
            const_in1.append(diff_boxes)
        const_in1 = torch.cat(const_in1, dim=-1)

        const_in2 = []
        if self.hparams.use_box:
            const_in2.append(box)
        if self.hparams.use_count:
            const_in2.append(count)
        if self.hparams.use_class_mod:
            const_in2.append(class_embs)
        if const_in2:
            const_in2 = torch.cat(const_in2, dim=1)
        else:
            const_in2 = None

        reinforcer_obj_input = None
        if self.hparams.use_reinforcer:
            reinforcer_obj_input = torch.cat([const_in1, const_in2], dim=1)

        # language generation loop
        ce_loss = None
        mmi_loss = None
        switch_loss = None
        switch_loss_elems = None
        if self.training:
            # Initialize LSTM state
            h1, c1 = self.init_hidden(bsz)
            h2, c2 = self.init_hidden(bsz)
            hidden = (h1, c1, h2, c2)

            # Create tensors to hold word prediction scores
            pre_out = torch.zeros((bsz, max_len, self.vs), device=self.device)
            if self.hparams.use_force_attr and self.training:
                switch_loss = torch.tensor(0, dtype=torch.float, device=self.device)
                switch_loss_elems = 0

            for t in range(max_len):
                # prepare embeddings and adaptive batch size
                h1, c1, h2, c2 = hidden
                if self.hparams.mmi_loss_weight > 0:
                    #true_bsz = int(bsz / 2)
                    #hid_half = int(h1.shape[0]/2)
                    #bs_t = sum([l > t for l in expr_lens[:hid_half]])
                    #neg_bs_t = sum([l > t for l in expr_lens[true_bsz*2:]])

                    # hidden = (torch.cat((h1[:bs_t], h1[hid_half:hid_half + bs_t]), dim=0),
                    #           torch.cat((c1[:bs_t], c1[hid_half:hid_half + bs_t]), dim=0),
                    #           torch.cat((h2[:bs_t], h2[hid_half:hid_half + bs_t]), dim=0),
                    #           torch.cat((c2[:bs_t], c2[hid_half:hid_half + bs_t]), dim=0))
                    hidden = (h1, c1, h2, c2)

                    exp = expr_embs[:, t, :].squeeze(1)
                    con1 = const_in1

                    con2 = const_in2
                    atemb = attr_embs #torch.cat((attr_embs[:bs_t], attr_embs[true_bsz:true_bsz + bs_t]), dim=0)
                    atmask = attr_mask

                    if not self.hparams.use_force_attr:
                        st = None
                    else:
                        st = switch_target[mask] #torch.cat((switch_target[:bs_t, t], switch_target[true_bsz:true_bsz + bs_t, t]), dim=0)

                else:
                    bs_t = sum([l > t for l in expr_lens])
                    hidden = (h1[:bs_t], c1[:bs_t], h2[:bs_t], c2[:bs_t])
                    exp = expr_embs[:bs_t, t, :].squeeze(1)
                    con1 = const_in1[:bs_t]
                    con2 = const_in2 if const_in2 is None else const_in2[:bs_t]
                    atemb = attr_embs if attr_embs is None else attr_embs[:bs_t]
                    atmask = attr_mask if attr_mask is None else attr_mask[:bs_t]
                    st = switch_target[:bs_t, t] if self.hparams.use_force_attr else None

                out, hidden, switch = self.decode_step(const_in1=con1, const_in2=con2,
                                                       embs=exp,
                                                       attr_att=atemb, att_mask=atmask,
                                                       hidden=hidden, attr_scr=attr_scr,
                                                       switch_target=st)
                if self.hparams.mmi_loss_weight > 0:
                    #pre_out[:bs_t, t, :] = out[:bs_t]
                    #pre_out[true_bsz:true_bsz + bs_t, t, :] = out[bs_t:]
                    mask = expr_lens > t
                    pre_out[mask, t, :] = out[mask]

                    #pre_out[true_bsz:true_bsz + bs_t, t, :] = out[bs_t:bs_t*2]
                    #pre_out[true_bsz*2:true_bsz*2+neg_bs_t, t, :] = out[bs_t*2:bs_t*2+neg_bs_t]
                else:
                    pre_out[:bs_t, t, :] = out
                # compute the switch loss at every timestep
                if self.hparams.use_force_attr:
                    if self.hparams.switch_loss == 'ce':
                        switch = torch.stack([1 - switch, switch], dim=1)
                        st = st.to(torch.long)
                    switch_loss += self.switch_loss(switch, st)
                    switch_loss_elems += bs_t

            # compute the entire loss at the end of the batch
            tmp_bsz = bsz
            if self.hparams.mmi_loss_weight > 0:
                true_bsz = int(bsz/bsz_divide)
                tmp_bsz = true_bsz
                ce_loss = self.ce_loss(pre_out[:true_bsz].reshape(-1, self.vs), target.reshape(-1))

                mmi_mask = (obj[true_bsz:true_bsz*2].sum(dim=-1) != 0)

                tmp_tar = target[mmi_mask].reshape(-1, 1)
                mm_loss = None
                if self.hparams.mmi_loss_version == "v1":
                    prob_true = self.logsoftmax(pre_out[:true_bsz][mmi_mask].reshape(-1, self.vs))
                    prob_neg = self.logsoftmax(pre_out[true_bsz:true_bsz * 2][mmi_mask].reshape(-1, self.vs))

                    mmi_loss = self.mmi_loss(input1=prob_true.gather(1, tmp_tar).squeeze(1),
                                             input2=prob_neg.gather(1, tmp_tar).squeeze(1),
                                             target=torch.ones((prob_true.shape[0],),
                                             dtype=torch.long, device=self.device))
                    mmi_loss = mmi_loss[tmp_tar.squeeze(1) != self.vocab.pad_idx].mean()

                elif self.hparams.mmi_loss_version == "v2":
                    sent_length = pre_out.size(1)

                    obj_pair = self.logsoftmax(pre_out[:true_bsz][mmi_mask]).reshape(-1, self.vs).gather(1, tmp_tar)
                    mmi_mask_bsz = mmi_mask.sum()
                    obj_pair = obj_pair.view(mmi_mask_bsz, sent_length)
                    obj_unpair = self.logsoftmax(pre_out[true_bsz:true_bsz * 2][mmi_mask]).reshape(-1, self.vs).gather(1, tmp_tar)
                    obj_unpair = obj_unpair.view(mmi_mask_bsz, sent_length)

                    obj_pairing_lengths = expr_lens[:true_bsz][mmi_mask]
                    obj_pairing_mask = torch.arange(sent_length)[None, :].to(obj_pair.device) < obj_pairing_lengths[:, None]

                    sentence_pair = self.logsoftmax(pre_out[:true_bsz]).reshape(-1, self.vs).gather(1, target.reshape(-1,1))
                    sentence_pair = sentence_pair.view(true_bsz, sent_length)

                    sentence_unpair = self.logsoftmax(pre_out[true_bsz * 2:]).reshape(-1, self.vs).gather(1, neg_sent_target.reshape(-1,1))
                    sentence_unpair = sentence_unpair.view(true_bsz, sent_length)

                    sentence_pairing_lengths = expr_lens[true_bsz*2:]
                    sentence_pairing_mask = torch.arange(sent_length)[None, :].to(obj_pair.device) < sentence_pairing_lengths[:, None]
                    obj_pairing_mask_no_mmi_mask = torch.arange(sent_length)[None, :].to(obj_pair.device) < expr_lens[:true_bsz][:, None]

                    object_pairing = {"unpair": (obj_unpair * obj_pairing_mask).sum(-1),
                                      "pair": (obj_pair * obj_pairing_mask).sum(-1)}

                    sentence_pairing = {"unpair": (sentence_unpair * sentence_pairing_mask).sum(-1),
                                        "pair": (sentence_pair * obj_pairing_mask_no_mmi_mask).sum(-1)}


                    mmi_loss = self.mmi_crit(pos_and_neg_object=object_pairing,
                                             pos_and_neg_sentenence=sentence_pairing,
                                              margin=self.hparams.mmi_loss_margin,
                                              vlamda=self.hparams.vlambda_weight,
                                              llamda=self.hparams.llambda_weight)
                else:
                    raise Exception("passed mmi_loss_version {} unrecognized".format( self.hparams.mmi_loss_version))

                # TODO: add loss for correct object with wrong sentence.

            else:
                ce_loss = self.ce_loss(pre_out.reshape(-1, self.vs), target.reshape(-1))
            if self.hparams.use_force_attr:
                switch_loss = switch_loss / switch_loss_elems
            if self.hparams.multi_task:
                assert action_target is not None and color_target is not None and location_target is not None, \
                    "the targets for multi task should be assigned"
                action_loss = self.action_loss(encoded['action'][:tmp_bsz], action_target)
                color_loss = self.color_loss(encoded['color'][:tmp_bsz], color_target)
                location_loss = self.location_loss(encoded['location'][:tmp_bsz], location_target)
                ce_loss = ce_loss + action_loss + color_loss + location_loss
            out = pre_out.argmax(dim=-1)
        else:
            out = self.beam_decode(expr_embs if self.training else expr_ids, const_in1, const_in2,
                                   attr_att=attr_embs, att_mask=attr_mask, attr_scr=attr_scr)

        reinforcer_loss = None
        reinforcer_acc = None
        if self.hparams.use_reinforcer and self.training:
            assert reinforcer_obj_input is not None, "reinforcer_obj_input is none!"
            #true_bsz =  self.hparams.batch_size_train #int(bsz / self.hparams.batch_size_train)
            true_bsz = int(bsz / bsz_divide)
            score = self.reinforcer.compute_score(obj_repr=reinforcer_obj_input,
                                          expr_ids=out, #expr_ids[:true_bsz],
                                          expr_lens=expr_lens).squeeze(-1)

            score = score[:true_bsz]
            #self.reward = lr_score * self.scale * mask
            #loss = -F.mean(F.sum(seq_prob, axis=0) / (xp.array(lang_length + 1)) * (
            #        self.reward - self.reward.mean()))
            #reinforcer_loss = self.reinforcer_ce_loss(pre_out[:true_bsz].reshape(-1, self.vs), target.reshape(-1)).view(true_bsz, -1).sum(-1) / expr_lens[:true_bsz]
            mask = torch.arange(pre_out.size(1))[None, :].to(pre_out.device) < expr_lens[:true_bsz][:, None]
            reinforcer_loss = (torch.max(self.logsoftmax(pre_out[:true_bsz]), dim=-1)[0] * mask).sum(-1) / expr_lens[:true_bsz]
            #self.reinforcer_ce_loss(pre_out[:true_bsz].reshape(-1, self.vs), target.reshape(-1)) #* (score - score.mean()))
            reinforcer_loss *= (score - score.mean())
            reinforcer_loss = -torch.mean(reinforcer_loss)
            reinforcer_acc = (score>0.5).float().mean()

        return out, ce_loss, switch_loss, mmi_loss, reinforcer_loss, reinforcer_acc

    def mmi_crit(self, pos_and_neg_object, pos_and_neg_sentenence, margin, vlamda=1, llamda=0):
        total_loss = 0

        # def triplet_loss(flow, num_label):
        #     pairGenP = flow[0]
        #     unpairGenP = flow[1]
        #     pairSentProbs = F.sum(pairGenP, axis=0) / (num_label + 1)
        #     unpairSentProbs = F.sum(unpairGenP, axis=0) / (num_label + 1)
        #     trip_loss = F.mean(F.relu(margin + unpairSentProbs - pairSentProbs))
        #     return trip_loss

        if vlamda != 0:
            vloss = torch.mean(torch.relu(margin + pos_and_neg_object["unpair"] - pos_and_neg_object["pair"]))
            #triplet_loss(lm_flows['visF'], num_labels['T'])
            total_loss += vlamda * vloss

        if llamda != 0:
            lloss = torch.mean(torch.relu(margin + pos_and_neg_sentenence["unpair"] - pos_and_neg_sentenence["pair"]))
            #triplet_loss(lm_flows['langF'], num_labels['F'])
            total_loss += llamda * lloss

        return total_loss

    def decode_step(self, const_in1, const_in2, embs, attr_att, att_mask, hidden, attr_scr, switch_target=None):
        h1, c1, h2, c2 = hidden
        # first layer
        in1 = [h2, embs, const_in1]
        in1 = torch.cat(in1, dim=1)
        h1, c1 = self.lstm1(in1, (h1, c1))

        # second layer
        in2 = [h1]
        # attention
        if self.hparams.use_attr_att:
            att_out = self.attention(attr_att, h1, mask=att_mask)
            in2.append(att_out)
        if const_in2 is not None:
            in2.append(const_in2)
        # second layer
        in2 = torch.cat(in2, dim=1)
        h2, c2 = self.lstm2(in2, (h2, c2))

        # process outputs for predicting next word
        out = self.out_layer(h2)

        # if we want to force attributes, we replace values in the output
        switch = None
        if self.hparams.use_force_attr:
            # decide for which sample in batch to replace: switch = 1, force attribute, otherwise do nothing
            switch = self.gate_layer_switch(h2)
            round_switch = torch.round(switch).type(torch.bool).squeeze(1)
            if round_switch.any() or (self.training and (switch_target > 0).any()):
                # select the indexes to fill from the output
                if self.training:
                    switch_batch_idx = switch_target.type(torch.bool).nonzero().squeeze(1)
                else:
                    switch_batch_idx = round_switch.nonzero().squeeze(1)
                # compute the outputs for the attributes with linear layer, where the switch says so
                attr_out = self.gate_attr_out_layer(h2[switch_batch_idx])  # attr vocab size
                tmp_out = out[switch_batch_idx].fill_(0.)  # full vocab size
                # fill with the new scores (and multiply those by the probability of those attributes
                tmp_out[:, self.attr2voc_map] = attr_out * attr_scr[switch_batch_idx]
                # fill the rest to -inf, so the softmax later sets everything to prob of 0
                tmp_out[tmp_out == 0] = 1**(-99)
                # place in original output
                out[switch_batch_idx] = tmp_out
            switch = switch.squeeze(1)
        return out, (h1, c1, h2, c2), switch

    def beam_decode(self, expr_ids, const_in1, const_in2, attr_att, att_mask, attr_scr):
        """
        :param expr_ids:
        :param const_in1:
        :param const_in2:
        :param attr_att:
        :param att_mask:
        :param attr_idx:
        :return:
        """
        decoded_batch = []
        # number_of_sentences = expr_embs.size(0)
        batch_size = expr_ids.size(0)
        # decoding goes sentence by sentence
        if hasattr(self.hparams, 'use_bert') and self.hparams.use_bert:
            expr_ids = expr_ids.squeeze(0)
        for idx in range(batch_size):
            # Number of sentence to generate
            endnodes = []
            h1, c1 = self.init_hidden(1)
            h2, c2 = self.init_hidden(1)
            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(hidden=(h1, c1, h2, c2), previous_node=None,
                                  word_id=expr_ids[idx:idx+1], log_prob=0, length=0, is_start_node=True)
            nodes = PriorityQueue()
            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1
            t = 0
            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000:
                    break

                # fetch the best node
                score, n = nodes.get()
                qsize -= 1

                if hasattr(self.hparams, 'use_bert') and self.hparams.use_bert:
                    # check if we need to terminate the node
                    if t > 0:
                        # end-of-sentence-token or max length reached
                        if (n.word_id.item() == self.tokenizer.sep_token and n.prev_node != None) \
                                or n.length >= 60:
                            endnodes.append((score, n))
                            # if we reached maximum # of sentences required
                            if len(endnodes) >= self.hparams.beam_topk:
                                break
                    ids = self.get_node_sequence(n)
                    bert_out = self.bert_model(ids.unsqueeze(0))
                    last_emb = bert_out.last_hidden_state[:,-1,:]
                else:
                    last_id = n.word_id
                    # check if we need to terminate the node
                    if t > 0:
                        # end-of-sentence-token or max length reached
                        if (n.word_id.item() == self.vocab.eos_idx and n.prev_node != None) \
                                or n.length >= self.hparams.max_length:
                            endnodes.append((score, n))
                            # if we reached maximum # of sentences required
                            if len(endnodes) >= self.hparams.beam_topk:
                                break

                    last_emb = self.embedding(last_id).squeeze(1)

                hidden = n.hidden
                # decode for one step using decoder
                at_mask_in = att_mask[idx:idx+1] if att_mask is not None else att_mask
                out, hidden, _ = self.decode_step(const_in1=const_in1[idx:idx+1],
                                                  const_in2=const_in2 if const_in2 is None else const_in2[idx:idx+1],
                                                  embs=last_emb,
                                                  attr_att=attr_att if attr_att is None else attr_att[idx:idx+1],
                                                  att_mask=at_mask_in,
                                                  hidden=hidden, attr_scr=attr_scr)
                t += 1
                log_prob, indexes = torch.topk(out, self.hparams.beam_topk, dim=-1)

                # create next nodes and put them in queue
                for k in range(self.hparams.beam_topk):
                    decoded_t = indexes[0, k].view(1).long()
                    log_p = log_prob[0, k].item()

                    node = BeamSearchNode(hidden=hidden, previous_node=n, word_id=decoded_t,
                                          log_prob=n.logp + log_p, length=n.length + 1)
                    score = -node.eval()
                    nodes.put((score, node))
                    qsize += 1

            # choose nbest paths, back trace them if we haven't found a complete optimal sentence yet
            if len(endnodes) == 0:
                ## for returning multiple optimal unfinished sentences
                # endnodes = [nodes.get() for _ in range(self.hparams.beam_topk)]
                ## just return the best unfinished sentence
                endnodes = [nodes.get()]

            # utterances = []
            # for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            score, n = sorted(endnodes, key=operator.itemgetter(0))[0]
            utterance = self.get_node_sequence(n)
            # utterances.append(utterance)
            decoded_batch.append(utterance[1:])
        return decoded_batch

    def get_node_sequence(self, n: BeamSearchNode):
        # utterance = []
        # utterance.append(n.word_id.item())
        # # back trace until we reach the start node, or our stack is empty.
        # while n.prev_node != None and n.prev_node.is_start_node is False:
        #     n = n.prev_node
        #     utterance.append(n.word_id.item())
        # # reverse the sequence to put in correct order
        # utterance = utterance[::-1]
        return n.all_past_ids

    def training_step(self, batch, batch_idx):
        # gt_expr, expr_len, obj, img, box, attr_prob, pred_class, class_hot, _, _, switch_target = sort_batch(batch)
        # gt_expr, expr_len, obj, img, box, attr_prob, pred_class = batch
        batch = sort_batch(batch)
        # prepare input
        for k, v in batch.items():
            if type(v) is list:
                batch[k] = torch.repeat_interleave(v[0], self.hparams.n_seq_per_ref)
            else:
                bs = v.shape[0]
                batch[k] = v.reshape(bs * self.hparams.n_seq_per_ref, -1)


        # prepare input
        if self.hparams.mmi_loss_weight > 0:
            longest = max(torch.cat([batch["neg_expr_length"], batch['expression_length']]))  # includes sos, excluding eos
        else:
            longest = max(batch['expression_length'])  # includes sos, excluding eos
        end = min(longest, self.hparams.max_length)
        input_expr = batch['expression'][:, :end]

        target_expr = batch['expression'][:, 1:end + 1]
        switch_target = None
        neg_sentence_switch_target = None
        if 'switch_target' in batch:
            switch_target = batch['switch_target'][:, :end]
            neg_sentence_switch_target = batch['neg_switch_target'][:, :end]

        if end == self.hparams.max_length:
            input_expr[input_expr[:, -1] != self.vocab.pad_idx, -1] = self.vocab.eos_idx

        attr_ids = None
        fm = None
        fo = None
        encb = None
        act_tar = None
        col_tar = None
        loc_tar = None
        raw_sent = None
        neg_sent_target = None
        if self.hparams.multi_task:
            fm = batch['full_image']
            fo = batch['full_object']
            encb = batch['encoder_box']
            act_tar = batch['action_target']
            col_tar = batch['color_target']
            loc_tar = batch['location_target']
        if self.hparams.mmi_loss_weight > 0:
            neg_expr = batch['neg_expr'][:, :end]
            neg_sent_target = batch['neg_expr'][:, 1:end + 1]
            if end == self.hparams.max_length:
                neg_sent_target[neg_sent_target[:, -1] != self.vocab.pad_idx, -1] = self.vocab.eos_idx

            neg_sent_len = batch['neg_expr_length']

            input_expr = torch.cat([input_expr, input_expr, neg_expr], dim=0)
            lens = torch.cat([batch['expression_length'], batch['expression_length'], neg_sent_len], dim=0).view(-1)
            img = torch.cat([batch['image_feature'], batch['image_feature'], batch['image_feature']], dim=0)
            count = torch.cat([batch['count'], batch['negative_count'], batch['count']], dim=0)
            obj = torch.cat([batch['object_feature'], batch['negative_object'], batch['object_feature']], dim=0)
            box = torch.cat([batch['bounding_box'], batch['negative_box'], batch['bounding_box']], dim=0)
            attr = torch.cat([batch['attribute_prob'], batch['negative_attr_prob'], batch['attribute_prob']], dim=0)
            cls = torch.cat([batch['class_emb'], batch['negative_class'], batch['class_emb']], dim=0)
            clshot = torch.cat([batch['class_hot'], batch['negative_class_hot'], batch['class_hot']], dim=0)
            mean_diff_feat = torch.cat([batch['diff_feats'], batch['neg_diff_feats'], batch['diff_feats']], dim=0)
            diff_boxes = torch.cat([batch['diff_boxes'], batch['neg_diff_boxes'], batch['diff_boxes']], dim=0)
            if 'attribute_idx' in batch:
                attr_ids = torch.cat([batch['attribute_idx'], batch['negative_attr_idx'], batch['attribute_idx']],
                                     dim=0)
            # input_expr = torch.cat([input_expr, input_expr], dim=0)
            # lens = torch.cat([batch['expression_length'], batch['expression_length']], dim=0)
            # img = torch.cat([batch['image_feature'], batch['image_feature']], dim=0)
            # count = torch.cat([batch['count'], batch['negative_count']], dim=0)
            # obj = torch.cat([batch['object_feature'], batch['negative_object']], dim=0)
            # box = torch.cat([batch['bounding_box'], batch['negative_box']], dim=0)
            # attr = torch.cat([batch['attribute_prob'], batch['negative_attr_prob']], dim=0)
            # cls = torch.cat([batch['class_emb'], batch['negative_class']], dim=0).unsqueeze(1)
            # clshot = torch.cat([batch['class_hot'], batch['negative_class_hot']], dim=0)
            # mean_diff_feat = torch.cat([batch['diff_feats'], batch['neg_diff_feats']], dim=0)
            # diff_boxes = torch.cat([batch['diff_boxes'], batch['neg_diff_boxes']], dim=0)
            # if 'attribute_idx' in batch:
            #     attr_ids = torch.cat([batch['attribute_idx'], batch['negative_attr_idx']], dim=0)
            if switch_target is not None:
                switch_target = torch.cat([switch_target, switch_target, neg_sentence_switch_target], dim=0)


            if self.hparams.multi_task:
                fm = torch.cat([batch['full_image'], batch['full_image']], dim=0)
                fo = torch.cat([batch['full_object'], batch['negative_full_object']], dim=0)
                encb = torch.cat([batch['encoder_box'], batch['negative_encoder_box']], dim=0)
            if hasattr(self.hparams, 'use_bert') and self.hparams.use_bert:
                exit("wrong combination. cannot use MMI with BERT. Not implemented yet")
        else:
            obj = batch['object_feature']
            box = batch['bounding_box']
            attr = batch['attribute_prob']
            cls = batch['class_emb'].unsqueeze(1)
            clshot = batch['class_hot']
            img = batch['image_feature']
            count = batch['count']
            lens = batch['expression_length']
            mean_diff_feat = batch['diff_feats']
            diff_boxes = batch['diff_boxes']
            if hasattr(self.hparams, 'use_bert') and self.hparams.use_bert:
                raw_sent = batch['raw_sentence']
            if 'attribute_idx' in batch:
                attr_ids = batch['attribute_idx']

        # forward pass
        _, loss, switch_loss, MMI_loss, reinforcer_loss, reinforcer_acc = self.forward(expr_ids=input_expr,
                                                                                        target=target_expr,
                                                                                        obj=obj,
                                                                                        box=box,
                                                                                        attr_scr=attr,
                                                                                        attr_ids=attr_ids,
                                                                                        cls=cls,
                                                                                        cls_hot=clshot,
                                                                                        img=img,
                                                                                        count=count,
                                                                                        expr_lens=lens,
                                                                                        switch_target=switch_target,
                                                                                        full_image=fm,
                                                                                        full_object=fo,
                                                                                        encoder_box=encb,
                                                                                        action_target=act_tar,
                                                                                        color_target=col_tar,
                                                                                        location_target=loc_tar,
                                                                                        raw_sentence=raw_sent,
                                                                                        diff_feats=mean_diff_feat,
                                                                                        diff_boxes=diff_boxes,
                                                                                        neg_sent_target=neg_sent_target)
        # loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if switch_loss is not None:
            self.log('switch_loss', switch_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            loss = loss + (switch_loss * self.hparams.switch_loss_weight)
        if MMI_loss is not None:
            self.log('mmi_loss', MMI_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            loss = loss + (MMI_loss * self.hparams.mmi_loss_weight)
        if reinforcer_loss is not None:
            self.log('reinforcer_loss', reinforcer_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('reinforcer_acc', reinforcer_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            loss = loss + reinforcer_loss

        return loss

    def validation_step(self, batch, batch_idx):
        #batch = sort_batch(batch)
        # gt_expr, expr_len, obj, img, box, attr_prob, pred_class, class_hot, image_id, region_id, _ = sort_batch(batch)
        # gt_expr, expr_len, obj, img, box, attr_prob, pred_class = batch
        # prepare input
        pred_class = batch['class_emb'].unsqueeze(1)
        # forward pass
        target_expr = None
        start = None
        if 'expression' in batch:
            target_expr = batch['expression'][:, 1:self.hparams.max_length+1]
            target_expr[target_expr[:, -1] != self.vocab.pad_idx, -1] = self.vocab.eos_idx
            start = batch['expression'][:, 0]
        gen_expr, _, _, _, _, _ = self.forward(expr_ids=None,
                                         target=target_expr,
                                         obj=batch['object_feature'],
                                         box=batch['bounding_box'],
                                         attr_scr=batch['attribute_prob'],
                                         attr_ids=batch['attribute_idx'] if 'attribute_idx' in batch else None,
                                         cls=pred_class,
                                         cls_hot=batch['class_hot'],
                                         img=batch['image_feature'],
                                         count=batch['count'],
                                         start=start,
                                         full_image=batch['full_image'] if self.hparams.multi_task else None,
                                         full_object=batch['full_object'] if self.hparams.multi_task else None,
                                         encoder_box=batch['encoder_box'] if self.hparams.multi_task else None,
                                         diff_feats=batch['diff_feats'] if hasattr(self.hparams, 'use_diff') and self.hparams.use_diff else None,
                                         diff_boxes=batch['diff_boxes'] if hasattr(self.hparams, 'use_diff') and self.hparams.use_diff else None)
        return gen_expr, \
               batch['image_id'].detach().cpu().numpy() if 'image_id' in batch else batch_idx

    def validation_epoch_end(self, outputs):
        if self.hparams.dataset == 'vg':
            ann_file, pred_file = self.vg_val_end(outputs)
        else:
            ann_file, pred_file = self.t2c_val_end(outputs)

        with HiddenOutput():
            if self.coco is None:
                self.coco = COCO(ann_file)
            coco_res = self.coco.loadRes(pred_file)
            coco_eval = COCOEvalCap(self.coco, coco_res, exclude_scorers=['spice'])
            coco_eval.evaluate()
        lower_eval = {}
        for method, score in coco_eval.eval.items():
            lower_eval[method.lower()] = score
        BEST = lower_eval[self.hparams.monitor] > self.best[self.hparams.monitor]
        if BEST:
            self.best['epoch'] = self.current_epoch
        for method, score in lower_eval.items():
            self.log(method, score, prog_bar=True)
            if BEST:
                self.best[method] = score
            self.log('best_ep_{}'.format(method), self.best[method])
            self.log('best_epoch', self.best['epoch'])

    def test_step(self, batch, batch_idx):
        start_t = time.time()
        out = self.validation_step(batch, batch_idx)
        end_t = time.time()
        inference_time = end_t - start_t
        return out, inference_time

    def test_epoch_end(self, outputs):
        # val_loss = np.mean([out[4].item() for out in outputs])
        inference_time = np.mean([out[1] for out in outputs])
        outputs = [out[0] for out in outputs]
        if self.hparams.dataset == 'vg':
            ann_file, pred_file = self.vg_val_end(outputs)
        else:
            _, pred_file = self.t2c_val_end(outputs, test_log_dir=self.hparams.resume_run_dir)
            # ann_file = 'processed/annotations/t2c_anns_val.json'
            ann_file = 'processed/annotations/t2c_anns_test.json'
        print('\n\n', pred_file, '\n\n')
        coco = COCO(ann_file)
        coco_res = coco.loadRes(pred_file)
        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.evaluate()

        self.print("\n\nTESTING RESULTS:\n======================")
        for method, score in coco_eval.eval.items():
            self.print('{:>15}:\t{}'.format(method, score))
        self.print('{:>15}:\t{}'.format('inference_time', inference_time))
        self.print("======================")
        print(','.join(list(coco_eval.eval.keys())))
        vals = [str(val) for val in coco_eval.eval.values()]
        print(','.join(vals))
        self.print("======================")

    def sample(self, batch):
        self.eval()
        out = self.validation_step(batch, 0)
        expr = self.vocab.to_string(out[0][0], join=True)
        return expr

    def t2c_val_end(self, outputs, test_log_dir=None):
        ann_file = 'processed/annotations/t2c_anns_val.json'
        if test_log_dir is None:
            if self.hparams.logger_type == 'tb':
                log_dir = self.logger.log_dir
            elif self.hparams.logger_type == 'wandb':
                log_dir = os.path.join(self.logger.save_dir, self.logger.experiment.name,
                                       self.logger.experiment.dir.split('/')[-2])
            else:
                log_dir = 'logs'
            pred_file = os.path.join(log_dir, 'predictions/epoch{:04d}_step{:07d}.json'.format(self.current_epoch,
                                                                                           self.global_step))
        else:
            pred_file = os.path.join(test_log_dir,
                                     'predictions/TEST_epoch{:04d}_step{:07d}.json'.format(self.current_epoch,
                                                                                           self.global_step))

        pred_dict = []
        for output in outputs:
            gen_out, im_id = output
            for gen, im in zip(gen_out, im_id):
                if hasattr(self.hparams, 'use_bert') and self.hparams.use_bert:
                    sentence = self.tokenizer.decode(gen)
                else:
                    sentence = self.vocab.to_string(gen, join=True)
                pd = {'image_id': int(im),
                      'caption': sentence}
                pred_dict.append(pd)
        with open(pred_file, 'w') as f:
            json.dump(pred_dict, f)
        return ann_file, pred_file

    def vg_val_end(self, outputs):
        if self.hparams.logger_type == 'tb':
            log_dir = self.logger.log_dir
        elif self.hparams.logger_type == 'wandb':
            log_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
        else:
            log_dir = 'logs'
        pred_file = os.path.join(log_dir, 'predictions/epoch{:03d}_step{:07d}.json'.format(self.global_step,
                                                                                           self.current_epoch))
        # os.makedirs(os.path.join(self.logger.log_dir, 'annotations'), exist_ok=True)
        ann_file = os.path.join(self.logger.log_dir, 'annotations/epoch{:03d}_step{:07d}.json'.format(self.global_step,
                                                                                              self.current_epoch))

        ann_dict = {'annotations': [], 'images': []}
        pred_dict = []
        coco_im_id = 0
        for output in outputs:
            gen_out, gt_out, reg_id, im_id = output
            for gen, gt, reg, im in zip(gen_out, gt_out, reg_id, im_id):
                ad = {'image_id': coco_im_id,
                      'id': coco_im_id,
                      'caption': self.vocab.to_string(gt, join=True),
                      'real_im_id': int(im),
                      'region_id': int(reg)}
                ann_dict['annotations'].append(ad)
                imd = {'id': coco_im_id, 'real_im_id': int(im)}
                ann_dict['images'].append(imd)
                pd = {'image_id': coco_im_id,
                      'caption': self.vocab.to_string(gen, join=True)}
                pred_dict.append(pd)
                coco_im_id += 1
        with open(ann_file, 'w') as f:
            json.dump(ann_dict, f)
        with open(pred_file, 'w') as f:
            json.dump(pred_dict, f)
        return ann_file, pred_file

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self): #train_dataloader
        shuffle = True
        if self.hparams.dataset == 'vg':
            dataset = VGDataset(datadir=self.hparams.datadir, feature_file=self.hparams.features_file,
                                data_file=self.hparams.data_file, split='train', rootdir=self.rootdir)
            shuffle = False
        elif self.hparams.dataset == 't2c':
            vocab_ids2attr_ids = None
            if self.hparams.use_force_attr:
                attr_ids2voc_ids = {idx: self.vocab.index(self.attr_vocab[idx]) for idx in range(len(self.attr_vocab))}
                vocab_ids2attr_ids = dict(map(reversed, attr_ids2voc_ids.items()))
            # if self.hparams.correct_sort:
            #     feat_fname = 'correct_sort_' + self.hparams.feature_filename
            # else:
            feat_fname = self.hparams.feature_filename
            if self.hparams.old_attr:
                feat_fname = 'OLD-ATTR_'+feat_fname
            if self.hparams.multi_task:
                feat_fname = 'IMAGE_'+feat_fname
            dataset = T2CDataset(datadir=self.hparams.datadir,
                                 feature_filename=feat_fname,
                                 attr_vocab=self.attr_vocab,
                                 class_vocab=self.class_vocab,
                                 split='train',
                                 pad_idx=self.vocab.pad_idx,
                                 start_idx=self.vocab.sos_idx,
                                 end_idx=self.vocab.eos_idx,
                                 vocab_ids2attr_ids=vocab_ids2attr_ids,
                                 max_attr=self.hparams.max_attrs,
                                 use_gt=self.hparams.attribute_gt,
                                 prob2hot=self.hparams.attribute_prob2hot,
                                 old_attr=self.hparams.old_attr,
                                 use_negatives=self.hparams.mmi_loss_weight > 0,
                                 multi_task=self.hparams.multi_task,
                                 use_bert=hasattr(self.hparams, 'use_bert') and self.hparams.use_bert)
        else:
            dataset = None
            exit('given dataset does not exist')

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size_train,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self): #val_dataloader
        if self.hparams.dataset == 'vg':
            dataset = VGDataset(datadir=self.hparams.datadir, feature_file=self.hparams.features_file,
                                data_file=self.hparams.data_file, split='val', rootdir=self.rootdir)
        elif self.hparams.dataset == 't2c':
            # if self.hparams.correct_sort:
            #     feat_fname = 'correct_sort_' + self.hparams.feature_filename
            # else:
            feat_fname = self.hparams.feature_filename
            if self.hparams.old_attr:
                feat_fname = 'OLD-ATTR_'+feat_fname
            if self.hparams.multi_task:
                feat_fname = 'IMAGE_'+feat_fname
            dataset = T2CDataset(datadir=self.hparams.datadir,
                                 feature_filename=feat_fname,
                                 attr_vocab=self.attr_vocab,
                                 class_vocab=self.class_vocab,
                                 split='val',
                                 pad_idx=self.vocab.pad_idx,
                                 start_idx=self.vocab.sos_idx,
                                 end_idx=self.vocab.eos_idx,
                                 max_attr=self.hparams.max_attrs,
                                 use_gt=self.hparams.attribute_gt,
                                 prob2hot=self.hparams.attribute_prob2hot,
                                 old_attr=self.hparams.old_attr,
                                 multi_task=self.hparams.multi_task,
                                 use_bert=hasattr(self.hparams, 'use_bert') and self.hparams.use_bert)
        else:
            dataset = None
            exit('given dataset does not exist')

        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self): #test_dataloader
        if self.hparams.dataset == 'vg':
            dataset = VGDataset(datadir=self.hparams.datadir, feature_file=self.hparams.features_file,
                                data_file=self.hparams.data_file, split='test', rootdir=self.rootdir)
        elif self.hparams.dataset == 't2c':
            # if self.hparams.correct_sort:
            #     feat_fname = 'correct_sort_' + self.hparams.feature_filename
            # else:
            feat_fname = self.hparams.feature_filename
            if self.hparams.old_attr:
                feat_fname = 'OLD-ATTR_'+feat_fname
            if self.hparams.multi_task:
                feat_fname = 'IMAGE_'+feat_fname
            dataset = T2CDataset(datadir=self.hparams.datadir,
                                 feature_filename=feat_fname,
                                 attr_vocab=self.attr_vocab,
                                 class_vocab=self.class_vocab,
                                 split='test',
                                 pad_idx=self.vocab.pad_idx,
                                 start_idx=self.vocab.sos_idx,
                                 end_idx=self.vocab.eos_idx,
                                 max_attr=self.hparams.max_attrs,
                                 use_gt=self.hparams.attribute_gt,
                                 prob2hot=self.hparams.attribute_prob2hot,
                                 old_attr=self.hparams.old_attr,
                                 multi_task=self.hparams.multi_task,
                                 use_bert=hasattr(self.hparams, 'use_bert') and self.hparams.use_bert)
        else:
            dataset = None
            exit('given dataset does not exist')

        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--feature_size', type=int, default=2048, help='size of CNN output and model input')
        parser.add_argument('--feature_hidden', type=int, default=512, help='size of hidden layers')
        parser.add_argument('--lstm_hidden', type=int, default=512, help='size of lstm hidden layers')
        parser.add_argument('--attention_hidden', type=int, default=512, help='size of attention hidden layers')
        parser.add_argument('--keywordmap_hidden', type=int, default=16, help='size of attention hidden layers')
        parser.add_argument('--embedding_size', type=int, default=300, choices=[300, 768],
                            help='size of embeddings. Glove=300, Bert=768.')
        parser.add_argument('--use_img',  type=str2bool, nargs='?', const=True, default=False,
                            help='whether to use img features')
        parser.add_argument('--use_attr_att',  type=str2bool, nargs='?', const=True, default=False,
                            help='whether to use attributes')
        parser.add_argument('--use_force_attr', type=str2bool, nargs='?', const=True, default=False,
                            help='train a gate that can force attributes directly in the caption.')
        parser.add_argument('--use_attr_hot', type=str, default='False', choices=['True', 'False', 'map'],
                            help='whether to use attributes as a boolean hot vector')
        parser.add_argument('--use_count', type=str2bool, nargs='?', const=True, default=False,
                            help='use distance count')
        parser.add_argument('--use_class_mod', type=str2bool, nargs='?', const=True, default=False,
                            help='whether to use class in model')
        parser.add_argument('--use_class_hot', type=str, default='False', choices=['True', 'False', 'map'],
                            help='whether to use class as a boolean hot vector')
        parser.add_argument('--use_class_att', type=str2bool, nargs='?', const=True, default=False,
                            help='whether to use class in attention. With use_attr at False: does nothing')
        parser.add_argument('--use_box', type=str2bool, nargs='?', const=True, default=False,
                            help='whether to use box location')
        parser.add_argument('--max_length', type=int, default=15, help='Max sequence length')
        parser.add_argument('--max_attrs', type=int, default=1, choices=[1], help='number of top scoring attributes to use. '
                                                                     'Assumes multiple arrays, not used with gt.'
                                                                     'Note: higher than 3 may cause errors')
        parser.add_argument('--lr', type=float, default=0.000005, help='learning rate')
        parser.add_argument('--beam_topk', type=int, default=10, help='beam search size')
        parser.add_argument('--switch_loss_weight', type=float, default=0.1, help='multiplied with the switch_loss')
        parser.add_argument('--switch_loss', type=str, default='mse', choices=['mse', 'l1', 'smooth_l1', 'ce'],
                            help='witch switch loss to use')
        parser.add_argument('--attribute_gt', type=str2bool, nargs='?', const=True, default=False,
                            help='whether to use ground-truth attributes or not')
        parser.add_argument('--attribute_prob2hot', type=str2bool, nargs='?', const=True, default=False,
                            help='convert the attribute probabiltities to one hot for the max_attributes')
        parser.add_argument('--old_attr', type=str2bool, nargs='?', const=True, default=False,
                            help='use the old T2C attributes, which are outdated and shouldnt be used.')
        parser.add_argument('--mmi_loss_weight', type=float, default=0.1, help='what weight to give to the mmi loss. '
                                                                             'If 0, only use the CE loss.')
        parser.add_argument('--mmi_loss_margin', type=float, default=0.1, help='What is the margin for the MMI loss.')
        parser.add_argument('--multi_task', type=str2bool, nargs='?', const=True, default=False,
                            help='Train multi-tasks with attr prediction.')
        parser.add_argument('--use_bert', type=str2bool, nargs='?', const=True, default=False,
                            help='Encode tokens with bert.')
        parser.add_argument('--use_diff', type=str2bool, nargs='?', const=True, default=False,
                            help='use the diff feats to help disambiguate between nearby boxes')
        parser.add_argument('--use_reinforcer', type=str, nargs="?", const=True, default="",
                            help="use_reinforcer")
        parser.add_argument('--mmi_loss_version', type=str, default="v1",
                            help="which mmi_loss to use. v1 is nn.MarginRankingLoss, v2 is what is used in SLR and SR")
        parser.add_argument('--vlambda_weight', type=float, default=1,
                            help="weight for vlambda in v2 mmi_loss")
        parser.add_argument('--llambda_weight', type=float, default=0,
                            help="weight for llambda in v2 mmi_loss")
        parser.add_argument('--n_seq_per_ref', type=int, default=3)

        return parser
