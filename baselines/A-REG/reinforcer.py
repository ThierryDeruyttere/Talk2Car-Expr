import operator  # allows us to make the queue of BeamSearchNode Nodes comparable
import argparse
import os
import sys
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
        expr_len, sorting = batch['expression_length'].sort(descending=True)
        for d_key, d_val in batch.items():
            if d_key == 'raw_sentence':
                return_dict[d_key] = [d_val[i] for i in sorting]
            else:
                return_dict[d_key] = d_val[sorting]
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


class LSTM_Attention(nn.Module):
    def __init__(self, embedding, input_size, output_size,
                 n_layers=1, use_attention=True):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = output_size
        self.embedding = embedding
        self.batch_first = True
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=int(output_size / 2),
                           # dropout=dropout,
                           num_layers=n_layers, bidirectional=True,
                           batch_first=self.batch_first)

        self.use_attention = use_attention
        self.embedding_to_score = nn.Linear(self.hidden_size, 1)

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_size // 2)
        c0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_size // 2)
        return h0.to(device), c0.to(device)

    def forward(self, inputs, input_lengths):
        #inputs = inputs.permute(1, 0)

        self.hidden = self.init_hidden(inputs.size(0), inputs.device)

        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False)
        outputs, self.hidden = self.rnn(packed, self.hidden)
        output, output_lengths = pad_packed_sequence(outputs, batch_first=self.batch_first)

        q_i = torch.cat([self.hidden[0][0], self.hidden[0][1]], -1)
        if self.use_attention:
            cws = output#.permute(1, 0, 2)
            mask = torch.arange(inputs.size(1))[None, :].to(inputs.device) < input_lengths[:, None]
            embedding_scores = self.embedding_to_score(cws * q_i.unsqueeze(1)).squeeze(-1)
            a_i = torch.softmax(embedding_scores.masked_fill((~mask), -float("inf")), dim=-1)
            ci = torch.sum(a_i.unsqueeze(-1) * cws, 1)
        else:
            ci = q_i

        return ci

class MetricNet(nn.Module):

    # def __init__(self):
    #     initializer = chainer.initializers.GlorotNormal(scale=math.sqrt(2))
    #     super(MetricNet, self).__init__(
    #         fc1=L.Linear(512 + 512, 512, initialW=initializer),
    #         norm1=L.BatchNormalization(512, eps=1e-5),
    #         fc2=L.Linear(512, 512, initialW=initializer),
    #         norm2=L.BatchNormalization(512, eps=1e-5),
    #         fc3=L.Linear(512, 1, initialW=initializer),
    #
    #         vis_norm=L.BatchNormalization(512, eps=1e-5),
    #     )
    #
    # def __call__(self, vis, lang):
    #     """
    #
    #     :param vis: visual features
    #     :param lang: encoded sentence features
    #     :return: joined which is of size [B,1]. It is thus a score vector
    #     """
    #     joined = F.concat([self.vis_norm(vis), lang], axis=1)
    #     joined = F.dropout(F.relu(self.norm1(self.fc1(joined))), ratio=0.2)
    #     joined = F.dropout(F.relu(self.norm2(self.fc2(joined))), ratio=0.2)
    #     joined = self.fc3(joined)
    #     return joined
    #
    def __init__(self, vis_input_size, lang_input_size):
        super(MetricNet, self).__init__()

        self.vis_proj = nn.Linear(vis_input_size, lang_input_size)

        self.compute_score = nn.Sequential(
            nn.Linear(lang_input_size*2, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1),
        )

    def forward(self, vis_feats, lang_feats):
        proj_vis = self.vis_proj(vis_feats)
        joined = torch.cat([proj_vis, lang_feats], dim=1)
        return self.compute_score(joined)

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
    return torch.index_select(a, dim, order_index)

class Reinforcer(pl.LightningModule):
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

        self.vs = len(vocab)

        #self.out_layer = nn.Linear(in_features=self.hparams.lstm_hidden, out_features=self.vs)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

        # for quicker validation
        self.coco = None
        self.init_weights()

        self.lang_encoder = LSTM_Attention(embedding=self.embedding,
                                           input_size=self.hparams.embedding_size,
                                           output_size=self.hparams.lstm_hidden
                                           )

        const_size1 = self.hparams.feature_hidden  # object feature vector
        const_size1 += self.hparams.feature_hidden if self.hparams.use_img else 0  # image feature
        const_size1 += len(self.attr_vocab) if self.hparams.use_attr_hot == 'True' else 0  # attr boolean hot vector
        const_size1 += self.hparams.keywordmap_hidden if self.hparams.use_attr_hot == 'map' else 0  # cls mapped vector
        const_size1 += len(self.class_vocab) if self.hparams.use_class_hot == 'True' else 0  # class one hot vector
        const_size1 += self.hparams.keywordmap_hidden if self.hparams.use_class_hot == 'map' else 0  # cls mapped vector
        const_size1 += (self.hparams.feature_hidden + 25) if hasattr(self.hparams,
                                                                     'use_diff') and self.hparams.use_diff else 0  # diff feats + diff boxes
        input_size1 = const_size1 + self.hparams.lstm_hidden  # output h2
        input_size1 += self.hparams.embedding_size  # previous predicted word
        self.lstm1 = nn.LSTMCell(input_size=input_size1, hidden_size=self.hparams.lstm_hidden)
        # Create the second lstm layer
        const_size2 = 0
        const_size2 += 4 if self.hparams.use_box else 0  # box location
        const_size2 += 6 if self.hparams.use_count else 0  # count vector, fixed size of 6
        const_size2 += self.hparams.embedding_size if self.hparams.use_class_mod else 0  # class_embedding in model
        input_size2 = const_size2 + self.hparams.lstm_hidden  # output h1
        input_size2 += self.hparams.embedding_size if self.hparams.use_attr_att else 0  # attribute_attention

        self.metric_net = MetricNet(vis_input_size=const_size1+const_size2,
                                    lang_input_size=self.hparams.lstm_hidden)


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
                raw_sentence=None, diff_feats=None, diff_boxes=None):
        # prepare the input by collecting some numbers
        bsz = obj.size(0)

        assert expr_ids is not None, "the epxr_ids should be assigned"
        max_len = expr_ids.size(1)

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


        # Visual features
        obj_repr = torch.cat([const_in1, const_in2], dim=1)

        # Textual features
        # pass expr_ids through lstm
        le_feats = self.lang_encoder(expr_ids, expr_lens)

        # Compute score
        score = self.metric_net(obj_repr, le_feats)
        reinforce_loss = None

        if self.training:

            targets = torch.zeros((bsz,1), device=self.device)
            targets[:bsz//3] = 1

            reinforce_loss = F.binary_cross_entropy_with_logits(score, targets)

        return reinforce_loss, score

    def compute_score(self, obj_repr, expr_ids, expr_lens):
        with torch.no_grad():
            # Textual features
            # pass expr_ids through lstm
            le_feats = self.lang_encoder(expr_ids, expr_lens)

            # Compute score
            score = self.metric_net(obj_repr, le_feats)
        return torch.sigmoid(score)

    def training_step(self, batch, batch_idx):
        # gt_expr, expr_len, obj, img, box, attr_prob, pred_class, class_hot, _, _, switch_target = sort_batch(batch)
        # gt_expr, expr_len, obj, img, box, attr_prob, pred_class = batch
        batch = sort_batch(batch)
        # prepare input
        longest = max(torch.cat([batch["neg_expr_length"], batch['expression_length']]))  # includes sos, excluding eos
        end = min(longest, self.hparams.max_length)
        input_expr = batch['expression'][:, :end]
        target_expr = batch['expression'][:, 1:end + 1]

        neg_expr = batch['neg_expr'][:, 1:end + 1]
        neg_sent_len = batch['neg_expr_length']

        switch_target = None
        if 'switch_target' in batch:
            switch_target = batch['switch_target'][:, :end]
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


        input_expr = torch.cat([target_expr, target_expr, neg_expr], dim=0)
        lens = torch.cat([batch['expression_length'], batch['expression_length'], neg_sent_len], dim=0)
        img = torch.cat([batch['image_feature'], batch['image_feature'], batch['image_feature']], dim=0)
        count = torch.cat([batch['count'], batch['negative_count'], batch['count']], dim=0)
        obj = torch.cat([batch['object_feature'], batch['negative_object'], batch['object_feature']], dim=0)
        box = torch.cat([batch['bounding_box'], batch['negative_box'], batch['bounding_box']], dim=0)
        attr = torch.cat([batch['attribute_prob'], batch['negative_attr_prob'], batch['attribute_prob']], dim=0)
        cls = torch.cat([batch['class_emb'], batch['negative_class'], batch['class_emb']], dim=0).unsqueeze(1)
        clshot = torch.cat([batch['class_hot'], batch['negative_class_hot'], batch['class_hot']], dim=0)
        mean_diff_feat = torch.cat([batch['diff_feats'], batch['neg_diff_feats'], batch['diff_feats']], dim=0)
        diff_boxes = torch.cat([batch['diff_boxes'], batch['neg_diff_boxes'], batch['diff_boxes']], dim=0)
        if 'attribute_idx' in batch:
            attr_ids = torch.cat([batch['attribute_idx'], batch['negative_attr_idx'], batch['attribute_idx']], dim=0)

        # obj = batch['object_feature']
        # box = batch['bounding_box']
        # attr = batch['attribute_prob']
        # cls = batch['class_emb'].unsqueeze(1)
        # clshot = batch['class_hot']
        # img = batch['image_feature']
        # count = batch['count']
        # lens = batch['expression_length']
        # mean_diff_feat = batch['diff_feats']
        # diff_boxes = batch['diff_boxes']

        if 'attribute_idx' in batch:
            attr_ids = batch['attribute_idx']

        # forward pass
        reinforce_loss, _ = self.forward(expr_ids=input_expr,
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
                                        diff_boxes=diff_boxes
                                        )
        # loss
        self.log('reinforce_loss', reinforce_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return reinforce_loss



    def validation_step(self, batch, batch_idx):
        #batch = sort_batch(batch)
        # gt_expr, expr_len, obj, img, box, attr_prob, pred_class, class_hot, image_id, region_id, _ = sort_batch(batch)
        # gt_expr, expr_len, obj, img, box, attr_prob, pred_class = batch
        # prepare input
        pred_class = batch['class_emb'].unsqueeze(1)
        longest = max(batch['expression_length'])  # includes sos, excluding eos
        end = min(longest, self.hparams.max_length)
        target_expr = batch['expression'][:, 1:end + 1]

        # forward pass
        # target_expr = None
        # start = None
        # if 'expression' in batch:
        #     target_expr = batch['expression'][:, 1:self.hparams.max_length+1]
        #     target_expr[target_expr[:, -1] != self.vocab.pad_idx, -1] = self.vocab.eos_idx
        #     start = batch['expression'][:, 0]

        attr_ids = None
        fm = None
        fo = None
        encb = None
        act_tar = None
        col_tar = None
        loc_tar = None
        raw_sent = None

        all_negs = batch['all_negatives']
        if len(all_negs) == 0:
            return {"acc": 1, "no_neg":1}

        num_negs = None
        for k, v in all_negs.items():
            if num_negs is None:
                num_negs = len(v[0])
            all_negs[k] = v[0]

        total_num_objs = num_negs + 1

        # We assign the target expr to all objects in the image
        # The goal is to let the network indicate which object belongs to the expr
        input_expr = tile(target_expr, dim=0, n_tile=total_num_objs)
        lens = tile(batch['expression_length'], dim=0, n_tile=total_num_objs)
        img = tile(batch['image_feature'], dim=0, n_tile=total_num_objs)
        count = torch.cat([batch['count'], all_negs['negative_count']], dim=0)
        obj = torch.cat([batch['object_feature'], all_negs['negative_object']], dim=0)
        box = torch.cat([batch['bounding_box'], all_negs['negative_box']], dim=0)
        attr = torch.cat([batch['attribute_prob'], all_negs['negative_attr_prob']], dim=0)
        cls = torch.cat([batch['class_emb'], all_negs['negative_class']], dim=0).unsqueeze(1)

        clshot = torch.cat([batch['class_hot'], all_negs['negative_class_hot']], dim=0)
        mean_diff_feat = torch.cat([batch['diff_feats'], all_negs['neg_diff_feats']], dim=0)
        diff_boxes = torch.cat([batch['diff_boxes'], all_negs['neg_diff_boxes']], dim=0)
        if 'attribute_idx' in batch:
            attr_ids = torch.cat([batch['attribute_idx'], all_negs['negative_attr_idx']], dim=0)


        switch_target = None
        if 'switch_target' in batch:
            switch_target = batch['switch_target'][:, :end]

        reinforce_loss, scores = self.forward(expr_ids=input_expr,
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
                                        diff_boxes=diff_boxes
                                        )

        scores = torch.sigmoid(scores)
        pos_sc = scores[0]
        max_neg_sc = scores[1:].max()
        acc = int(pos_sc > max_neg_sc)

        #self.log("val_acc", acc, on_epoch=True, on_step=False, prog_bar=True)
        return {"acc": acc, "no_neg": 0}

    def validation_epoch_end(self, outputs):
        acc = 0
        cnt = 0
        for out in outputs:
            if out["no_neg"] == 0:
                acc += out["acc"]
                cnt += 1

        self.log("val_acc", acc/cnt, prog_bar=True)

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
            if self.hparams.use_all_negatives:
                feat_fname = 'ALL-NEGS_' + feat_fname
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
            if self.hparams.use_all_negatives:
                feat_fname = 'ALL-NEGS_' + feat_fname
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
                                 use_negatives=True,
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
            if self.hparams.use_all_negatives:
                feat_fname = 'ALL-NEGS_' + feat_fname
            dataset = T2CDataset(datadir=self.hparams.use_all_negatives,
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
        parser.add_argument('--use_all_negatives', type=str2bool, nargs='?', const=True, default=False,
                            help="use all negatives. Also the ones from the different class than the referred object.")
        return parser
