import argparse
import h5py
import json
import pickle
import os
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random

class VGDataset(Dataset):
    def __init__(self, datadir, feature_file, data_file, split='train', rootdir=None, n_seq_per_ref=3):
        super(VGDataset, self).__init__()
        # create list with all the image files belonging to the split
        self.h5_file = os.path.join(datadir, feature_file)
        self.features = None
        self.n_seq_per_ref = n_seq_per_ref
        if rootdir is None:
            self.dataset_file = os.path.join('processed', 'dataset', 'vg_dataset-{}.pkl'.format(split))
        else:
            print('adding rootdir \"{}\"'.format(rootdir))
            self.dataset_file = os.path.join(rootdir, 'processed', 'dataset', 'vg_dataset-{}.pkl'.format(split))
        # load the features hdf5
        if os.path.exists(self.dataset_file):
            self.load()
        else:
            split_txt = os.path.join(datadir, '{}.txt'.format(split))
            split_images = {}
            with open(split_txt, 'r') as f:
                for line in f:
                    split_images[line.split()[0].split('/')[1]] = True
            # load the and reduce the data dict so it only contains the images
            # with attributes and belong to the current split
            with open(os.path.join(datadir, data_file), 'r') as f:
                full_data = json.load(f)

            self.attr2idx = {}
            self.class2idx = {}
            for i, a in enumerate(full_data['vg_attrs']):
                self.attr2idx[a] = i
            for i, c in enumerate(full_data['vg_classes']):
                self.class2idx[c] = i

            data_idx = 0
            self.data = {}
            self.img_ids2reg_ids = defaultdict(dict)
            for i, d_key in tqdm(enumerate(full_data['data'].keys()), desc='filter dataset'):
                d = full_data['data'][d_key]
                if d['image_name'] in split_images:
                    self.data[data_idx] = d
                    data_idx += 1
                    self.img_ids2reg_ids[d['image_id']][d['region_id']] = True
            print('\nremains after filtering: {}\n'.format(len(self.data)))
            self.save()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        if self.features is None:
            self.features = h5py.File(self.h5_file, 'r')
        attr_prob = torch.tensor(self.features['attr_prob'][d['hdf5_ix']])
        reg = torch.tensor(self.features['region'][d['hdf5_ix']])
        expr = torch.tensor(self.features['phrases'][d['hdf5_ix']], dtype=torch.long)
        expr_len = torch.tensor(self.features['phrase_lengths'][d['hdf5_ix']], dtype=torch.long)
        img = torch.tensor(self.features['img'][d['img_hdf5_ix']])
        box = torch.tensor([d['x'], d['y'], d['x']+d['width'], d['y']+d['height']], dtype=torch.float)
        # create pred class distribution
        pred_class = torch.tensor(self.class2idx[d['predicted_class']], dtype=torch.long)
        return expr, expr_len, reg, img, box, attr_prob, pred_class, d['image_id'], d['region_id']

    def load(self):
        with open(self.dataset_file, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.attr2idx = tmp_dict['attr2idx']
        self.class2idx = tmp_dict['class2idx']
        self.data = tmp_dict['data']

    def save(self):
        os.makedirs(os.path.dirname(self.dataset_file), exist_ok=True)
        with open(self.dataset_file, 'wb') as f:
            pickle.dump({
                'attr2idx': self.attr2idx,
                'class2idx': self.class2idx,
                'data': self.data,
            }, f)

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--datadir', type=str,
                            default='/cw/liir/NoCsBack/testliir/datasets/refer_expr/vg/',
                            help='path to directory with data files')
        parser.add_argument('--features_file', type=str, default='sorted_features.h5')
        parser.add_argument('--data_file', type=str, default='sorted_data.json')
        return parser


class T2CDataset(Dataset):
    def __init__(self, datadir, feature_filename, attr_vocab, class_vocab, split='train',
                 pad_idx=0, start_idx=1, end_idx=2, class_one_hot=False, vocab_ids2attr_ids=None,
                 max_attr=1, use_gt=False, prob2hot=False, old_attr=False, use_negatives=False,
                 multi_task=False, use_bert=False, n_seq_per_ref=3):
        super(T2CDataset, self).__init__()

        # create path to the h5 file
        feature_file = '{}_{}.h5'.format(feature_filename, split)
        self.h5_file = os.path.join(datadir, feature_file)
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.class_one_hot = class_one_hot
        self.max_attr = max_attr
        self.use_gt = use_gt
        self.prob2hot = prob2hot
        self.old_attr = old_attr
        self.use_negatives = use_negatives
        self.vocab_ids2attr_ids = vocab_ids2attr_ids
        self.multi_task = multi_task
        self.use_bert = use_bert
        self.split = split
        self.n_seq_per_ref = 3

        if self.use_bert:
            with open(os.path.splitext(self.h5_file)[0]+'_sentences.json', 'r') as f:
                self.sentences = json.load(f)

        if self.multi_task:
            self.vocab_ids2loc_ids = dict(map(reversed, attr_vocab.loc_id2vocab_id.items()))
            self.vocab_ids2act_ids = dict(map(reversed, attr_vocab.act_id2vocab_id.items()))
            self.vocab_ids2col_ids = dict(map(reversed, attr_vocab.col_id2vocab_id.items()))


        # get the dataset length
        f = h5py.File(self.h5_file, 'r')
        self.len_ = len(f['descriptions'])
        f.close()

        # create features place holder
        self.features = None

        # load the classes and attributes names
        self.attr_vocab = attr_vocab
        self.class_vocab = class_vocab

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        # store the open hdf5 file in memory
        if self.features is None:
            self.features = h5py.File(self.h5_file, 'r')
        len_ = self.features['lengths'][index]
        # collect attributes
        if self.old_attr:
            attr = torch.tensor(self.features['attributes'][index], dtype=torch.float)
        else:
            if self.use_gt:
                attr_prob = torch.tensor(self.features['gt_attr_location'][index] + \
                                         self.features['gt_attr_color'][index] + \
                                         self.features['gt_attr_action'][index], dtype=torch.float)
                _, attr_idx = torch.topk(attr_prob, k=attr_prob.sum().item())
                raise NotImplementedError
            else:
                loc_idx = self.features['attr_location'][index].argsort()[-self.max_attr:][::-1]
                col_idx = self.features['attr_color'][index].argsort()[-self.max_attr:][::-1]
                act_idx = self.features['attr_action'][index].argsort()[-self.max_attr:][::-1]
                voc_loc_idx = [self.attr_vocab.loc_id2vocab_id[idx] for idx in loc_idx]
                voc_col_idx = [self.attr_vocab.col_id2vocab_id[idx] for idx in col_idx]
                voc_act_idx = [self.attr_vocab.act_id2vocab_id[idx] for idx in act_idx]
                attr_idx = torch.cat([torch.tensor(voc_loc_idx, dtype=torch.long),
                                      torch.tensor(voc_col_idx, dtype=torch.long),
                                      torch.tensor(voc_act_idx, dtype=torch.long)])
                if self.prob2hot:
                    loc_prob = torch.zeros((len(self.features['attr_location'][index]),), dtype=torch.float)
                    loc_prob[list(loc_idx)] = 1
                    col_prob = torch.zeros((len(self.features['attr_color'][index]),), dtype=torch.float)
                    col_prob[list(col_idx)] = 1
                    act_prob = torch.zeros((len(self.features['attr_action'][index]),), dtype=torch.float)
                    act_prob[list(act_idx)] = 1
                    attr_prob = torch.cat([loc_prob, col_prob, act_prob])
                else:
                    attr_prob = torch.cat([torch.tensor(self.features['attr_location'][index], dtype=torch.float),
                                           torch.tensor(self.features['attr_color'][index], dtype=torch.float),
                                           torch.tensor(self.features['attr_action'][index], dtype=torch.float)])
        # collect region
        reg = torch.tensor(self.features['object_features'][index], dtype=torch.float)
        # collect gt expresion
        expr_len = torch.tensor(len_, dtype=torch.long)
        expr = torch.full(size=(self.features['descriptions'].shape[1]+1,), fill_value=self.pad_idx, dtype=torch.long)
        expr[0] = self.start_idx
        expr[1:len(self.features['descriptions'][index])+1] = torch.tensor(self.features['descriptions'][index], dtype=torch.long)
        # collect image features
        img = torch.tensor(self.features['image_features'][index])
        # collect the box coordinates
        box = torch.tensor(self.features['bboxes'][index], dtype=torch.float)
        # collect the class features
        cls_hot = torch.zeros((len(self.class_vocab),), dtype=torch.float)
        cls_hot[self.features['classes'][index]] = 1
        cls = torch.tensor(self.features['classes'][index], dtype=torch.long)
        # collect count vec
        count_vec = torch.zeros((6,), dtype=torch.float)
        if self.features['count'][index] > 6:
            count_vec[5] = 1
        else:
            count_vec[self.features['count'][index]-1] = 1

        diff_feat = torch.Tensor(self.features['mean_difference_feat'][index])
        diff_boxes = torch.Tensor(self.features['mean_difference_box'][index])
        # return everything
        return_dict = {
            'expression': expr,
            'expression_length': expr_len,
            'object_feature': reg,
            'image_feature': img,
            'bounding_box': box,
            'class_emb': cls,
            'class_hot': cls_hot,
            'count': count_vec,
            # 'attribute_prob': attr_prob,
            # 'attribute_idx': attr_idx,
            'diff_feats': diff_feat,
            'diff_boxes': diff_boxes,
            'image_id': index
        }
        if self.use_bert:
            return_dict['raw_sentence'] = self.sentences[index]
        if self.old_attr:
            return_dict['attribute_prob'] = attr
        else:
            return_dict['attribute_prob'] = attr_prob
            return_dict['attribute_idx'] = attr_idx
        # get image
        if self.multi_task:
            return_dict['full_image'] = torch.tensor(self.features['full_image'][index], dtype=torch.float)
            return_dict['full_object'] = torch.tensor(self.features['full_object'][index], dtype=torch.float)
            x, y, w, _ = self.features['bboxes'][index]
            return_dict['encoder_box'] = torch.tensor([x/1600, (x + w/2) / 1600, (x+w)/1600], dtype=torch.float)
            return_dict['location_target'] = torch.tensor(
                self.vocab_ids2loc_ids[np.argmax(self.features['gt_attr_location'][index])], dtype=torch.long)
            return_dict['action_target'] = torch.tensor(
                self.vocab_ids2act_ids[np.argmax(self.features['gt_attr_action'][index])], dtype=torch.long)
            return_dict['color_target'] = torch.tensor(
                self.vocab_ids2col_ids[np.argmax(self.features['gt_attr_color'][index])], dtype=torch.long)

        n_seq = self.n_seq_per_ref if self.split == "train" else 1
        if n_seq > 1:
            for k,v in return_dict.items():
                if type(v) is torch.Tensor:
                    return_dict[k] = torch.cat([v.unsqueeze(0)] * n_seq, dim=0)
                else:
                    return_dict[k] = [v] * n_seq

        if self.use_negatives:

            out_dict = {}

            if n_seq > 1:
                selected_negs = []
                for _ in range(n_seq):
                    out_dict = {}
                    neg_int = 0 if self.features['negative_num'][index] == 0 else random.randint(0, self.features[
                        'negative_num'][index] - 1)
                    self.get_negative_data(index, neg_int, out_dict)
                    selected_negs.append(out_dict)

                all_negs_dict = {}
                for negative in selected_negs:
                    for neg_key, neg_val in negative.items():
                        if neg_key not in all_negs_dict:
                            all_negs_dict[neg_key] = []
                        all_negs_dict[neg_key].append(neg_val)

                for k, v in all_negs_dict.items():
                    return_dict[k] = torch.stack(v)

            else:
                neg_int = 0 if self.features['negative_num'][index] == 0 else random.randint(0, self.features[
                    'negative_num'][index] - 1)
                self.get_negative_data(index, neg_int, out_dict)

                for k,v in out_dict.items():
                    return_dict[k] = v

            if self.split == "val":
                all_negs = []
                for ix_neg in range(self.features['negative_num'][index]):
                    neg_out_dict = {}
                    self.get_negative_data(index, ix_neg, neg_out_dict)
                    all_negs.append(neg_out_dict)

                all_negs_dict = {}
                for negative in all_negs:
                    for neg_key, neg_val in negative.items():
                        if neg_key not in all_negs_dict:
                            all_negs_dict[neg_key] = []
                        all_negs_dict[neg_key].append(neg_val)

                for k,v in all_negs_dict.items():
                    all_negs_dict[k] = torch.stack(v)

                return_dict["all_negatives"] = all_negs_dict

        # when the switch targets
        if self.vocab_ids2attr_ids is not None:
            return_dict['switch_target'] = torch.tensor([int(id_.item() in self.vocab_ids2attr_ids.keys())
                                                         for id_ in expr[1:]],
                                                        dtype=torch.float)

        #if self.use_negatives:
        #    return_dict["expression"] = return_dict["expression"][:max(neg_sent_len, len_)]
        #else:
        #    return_dict["expression"] = return_dict["expression"][:len_]

        return return_dict

    def get_negative_data(self, index, neg_int, return_dict):
        return_dict['negative_object'] = torch.tensor(self.features['negative_object'][index, neg_int],
                                                      dtype=torch.float)
        return_dict['negative_box'] = torch.tensor(self.features['negative_box'][index, neg_int],
                                                   dtype=torch.float)
        return_dict['negative_class'] = torch.tensor(self.features['negative_class'][index, neg_int],
                                                     dtype=torch.long)
        neg_hot = torch.zeros((len(self.class_vocab),), dtype=torch.float)
        neg_hot[self.features['negative_class'][index, neg_int]] = 1
        return_dict['negative_class_hot'] = neg_hot
        return_dict['neg_diff_feats'] = torch.tensor(self.features['mean_neg_difference_feat'][index, neg_int],
                                                     dtype=torch.float)
        return_dict['neg_diff_boxes'] = torch.tensor(self.features['mean_neg_difference_box'][index, neg_int],
                                                     dtype=torch.float)
        # ADDED BY THIERRY
        rand_ix = random.randint(0, len(self) - 1)
        while rand_ix == index:
            rand_ix = random.randint(0, len(self) - 1)
        neg_sent_len = self.features['lengths'][rand_ix]
        neg_sent_len = torch.tensor(neg_sent_len, dtype=torch.long)
        neg_expr = torch.full(size=(self.features['descriptions'].shape[1] + 1,), fill_value=self.pad_idx,
                              dtype=torch.long)
        neg_expr[0] = self.start_idx
        neg_expr[1:len(self.features['descriptions'][rand_ix]) + 1] = torch.tensor(
            self.features['descriptions'][rand_ix], dtype=torch.long)
        return_dict['neg_expr'] = neg_expr  # [:max(neg_sent_len, len_)]
        return_dict['neg_expr_length'] = neg_sent_len
        ######## END ADDED BY THIERRY ########
        neg_count_vec = torch.zeros((6,), dtype=torch.float)
        if self.features['negative_count'][index, neg_int] > 6:
            neg_count_vec[5] = 1
        else:
            neg_count_vec[self.features['negative_count'][index, neg_int] - 1] = 1
        return_dict['negative_count'] = neg_count_vec
        neg_loc_idx = self.features['negative_location'][index, neg_int].argsort()[-self.max_attr:][::-1]
        neg_col_idx = self.features['negative_color'][index, neg_int].argsort()[-self.max_attr:][::-1]
        neg_act_idx = self.features['negative_action'][index, neg_int].argsort()[-self.max_attr:][::-1]
        neg_voc_loc_idx = [self.attr_vocab.loc_id2vocab_id[idx] for idx in neg_loc_idx]
        neg_voc_col_idx = [self.attr_vocab.col_id2vocab_id[idx] for idx in neg_col_idx]
        neg_voc_act_idx = [self.attr_vocab.act_id2vocab_id[idx] for idx in neg_act_idx]
        neg_attr_idx = torch.cat([torch.tensor(neg_voc_loc_idx, dtype=torch.long),
                                  torch.tensor(neg_voc_col_idx, dtype=torch.long),
                                  torch.tensor(neg_voc_act_idx, dtype=torch.long)])
        if self.multi_task:
            x, y, w, _ = self.features['negative_box'][index, neg_int]
            return_dict['negative_encoder_box'] = torch.tensor([x / 1600, (x + w / 2) / 1600, (x + w) / 1600],
                                                               dtype=torch.float)
            return_dict['negative_full_object'] = torch.tensor(
                self.features['negative_full_object'][index, neg_int], dtype=torch.float)
        if self.prob2hot:
            neg_loc_prob = torch.zeros((len(self.features['negative_location'][index, neg_int]),),
                                       dtype=torch.float)
            neg_loc_prob[list(neg_loc_idx)] = 1
            neg_col_prob = torch.zeros((len(self.features['negative_color'][index, neg_int]),),
                                       dtype=torch.float)
            neg_col_prob[list(neg_col_idx)] = 1
            neg_act_prob = torch.zeros((len(self.features['negative_action'][index, neg_int]),),
                                       dtype=torch.float)
            neg_act_prob[list(neg_act_idx)] = 1
            neg_attr_prob = torch.cat([neg_loc_prob, neg_col_prob, neg_act_prob])
        else:
            neg_attr_prob = torch.cat(
                [torch.tensor(self.features['negative_location'][index, neg_int], dtype=torch.float),
                 torch.tensor(self.features['negative_color'][index, neg_int], dtype=torch.float),
                 torch.tensor(self.features['negative_action'][index, neg_int], dtype=torch.float)])
        return_dict['negative_attr_prob'] = neg_attr_prob
        return_dict['negative_attr_idx'] = neg_attr_idx

        if self.vocab_ids2attr_ids is not None:
            return_dict['neg_switch_target'] = torch.tensor([int(id_.item() in self.vocab_ids2attr_ids.keys())
                                                         for id_ in neg_expr[1:]],
                                                        dtype=torch.float)


    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--datadir', type=str,
                            default='/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/h5_feats/',
                            help='path to directory with data files')
        parser.add_argument('--feature_filename', type=str, default='resnet_features')
        parser.add_argument('--attributes_file', type=str, default='attributes.json')
        parser.add_argument('--classes_file', type=str, default='talk2car_classes.json')
        return parser
