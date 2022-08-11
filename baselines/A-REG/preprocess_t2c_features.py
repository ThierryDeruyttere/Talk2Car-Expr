import argparse
import os
import json
import h5py
import nltk
import numpy as np
from tqdm import tqdm, trange
from vocabulary import Vocabulary, KeywordVocabulary
from PIL import Image
import torch
from torchvision.models import resnet152
from torchvision import transforms as T
import time
# from transformers import BertGenerationTokenizer


if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')

parser = argparse.ArgumentParser(description='preprocessing')
parser.add_argument('--old_attr', action='store_true', help='whether to use the old attributes or not. default not.')
parser.add_argument('--max_len', type=int, default=15, help='maximum expression length')
parser.add_argument('--add_image', action='store_true', help='add images to the hdf5')
# parser.add_argument('--bert', action='store_true', help='use bert embeddings')
args = parser.parse_args()

print('creating transforms...')
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])


IMG_ROOT = SET_THIS_PATH #i.e., '/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/imgs'

emb_file = 'glove.840B.300d.txt'
train_json = '../dataset/talk2car_descriptions_train.json'
val_json = '../dataset/talk2car_descriptions_val.json'
test_json = '../dataset/talk2car_descriptions_test.json'
location_train_json = '../dataset/paper_attribute_predictions/location/location_train.json'
location_val_json = '../dataset/paper_attribute_predictions/location/location_val.json'
location_test_json = '../dataset/paper_attribute_predictions/location/location_test.json'
color_train_json = '../dataset/paper_attribute_predictions/color/color_train.json'
color_val_json = '../dataset/paper_attribute_predictions/color/color_val.json'
color_test_json = '../dataset/paper_attribute_predictions/color/color_test.json'
action_train_json = '../dataset/paper_attribute_predictions/action/action_train.json'
action_val_json = '../dataset/paper_attribute_predictions/action/action_val.json'
action_test_json = '../dataset/paper_attribute_predictions/action/action_test.json'
positive_train_json = '../dataset/positives_train.json'
positive_val_json = '../dataset/positives_val.json'
positive_test_json = '../dataset/positives_test.json'
negative_train_json = '../dataset/negatives/negs_same_class_train.json'
negative_val_json = '../dataset/negatives/negs_same_class_val.json'
negative_test_json = '../dataset/negatives/negs_same_class_test.json'
h5_path = 'h5_feats/'


# import the resnet model
print('creating resnet...')
# TODO: REWRITE TO USE THIERRYs ATTRIBUTE RESNET
resnet = resnet152(pretrained=True).to(dev).eval()
resnet.fc = torch.nn.Identity()

# Collect the data from the json files
print('loading jsons...')
with open(train_json, 'r') as f:
    train_data = json.load(f)
with open(val_json, 'r') as f:
    val_data = json.load(f)
with open(test_json, 'r') as f:
    test_data = json.load(f)
with open(location_train_json, 'r') as f:
    loc_train_data = json.load(f)
with open(location_val_json, 'r') as f:
    loc_val_data = json.load(f)
with open(location_test_json, 'r') as f:
    loc_test_data = json.load(f)
with open(color_train_json, 'r') as f:
    col_train_data = json.load(f)
with open(color_val_json, 'r') as f:
    col_val_data = json.load(f)
with open(color_test_json, 'r') as f:
    col_test_data = json.load(f)
with open(action_train_json, 'r') as f:
    act_train_data = json.load(f)
with open(action_val_json, 'r') as f:
    act_val_data = json.load(f)
with open(action_test_json, 'r') as f:
    act_test_data = json.load(f)
with open(positive_train_json, 'r') as f:
    pos_train_data = json.load(f)
with open(positive_val_json, 'r') as f:
    pos_val_data = json.load(f)
with open(positive_test_json, 'r') as f:
    pos_test_data = json.load(f)
with open(negative_train_json, 'r') as f:
    neg_train_data = json.load(f)
with open(negative_val_json, 'r') as f:
    neg_val_data = json.load(f)
with open(negative_test_json, 'r') as f:
    neg_test_data = json.load(f)

# create vocabs
print('creating vocabs...')
# attr_vocab = KeywordVocabulary(emb_file=emb_file, data='attributes', dataset='t2c', multi_file=not old_attr,
#                                vocab_name='correct_sort_keyword_vocab')
attr_vocab = KeywordVocabulary(emb_file=emb_file, data='attributes', dataset='t2c', multi_file=not args.old_attr,
                               vocab_name='keyword_vocab')
class_vocab = KeywordVocabulary(emb_file=emb_file, data='classes', dataset='t2c')

# if args.bert:
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# else:
vocab = Vocabulary(emb_file=emb_file, dataset='t2c', keyword_vocab=[attr_vocab, class_vocab])

# create h5 files
print('creating h5 files...')
# creating the variables containing the hdf5 files
# if old_attr:
#     train_feat = h5py.File(
#         '/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/OLD-ATTR_correct_sort_resnet_features_train.h5',
#         mode='w', libver='latest')
#     val_feat = h5py.File('/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/OLD-ATTR_correct_sort_resnet_features_val.h5',
#                          mode='w', libver='latest')
#     test_feat = h5py.File(
#         '/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/OLD-ATTR_correct_sort_resnet_features_test.h5',
#         mode='w', libver='latest')
#
# else:
#     train_feat = h5py.File('/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/correct_sort_resnet_features_train.h5',
#                            mode='w', libver='latest')
#     val_feat = h5py.File('/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/correct_sort_resnet_features_val.h5',
#                            mode='w', libver='latest')
#     test_feat = h5py.File('/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/correct_sort_resnet_features_test.h5',
#                            mode='w', libver='latest')
os.makedirs(h5_path, exist_ok=True)
h5_name = 'resnet_features'
if args.old_attr:
    h5_name = 'OLD-ATTR_' + h5_name
if args.add_image:
    h5_name = 'IMAGE_' + h5_name
train_h5file = os.path.join(h5_path, h5_name+'_train.h5')
val_h5file = os.path.join(h5_path, h5_name+'_val.h5')
test_h5file = os.path.join(h5_path, h5_name+'_test.h5')

# loop over each split
for data, h5name, split, location, color, action, positive, negative in zip([train_data, val_data, test_data],
                                                                            [train_h5file, val_h5file, test_h5file],
                                                                            ['train', 'val', 'test'],
                                                                            [loc_train_data, loc_val_data, loc_test_data],
                                                                            [col_train_data, col_val_data, col_test_data],
                                                                            [act_train_data, act_val_data, act_test_data],
                                                                            [pos_train_data, pos_val_data, pos_test_data],
                                                                            [neg_train_data, neg_val_data, neg_test_data]):


    h5file = h5py.File(h5name, mode='w', libver='latest')
    out_sentences = os.path.splitext(h5file.filename)[0]+'_sentences.json'
    sent_list = []
    print('processing h5file {}'.format(os.path.basename(h5file.filename)))
    ann_dict = {'annotations': [], 'images': []}
    keys = []
    lengths = []

    print('sorting...')
    # sort the keys based on description length
    for key in tqdm(data.keys(), desc='collecting lengths'):
        d = data[key]
        desc = nltk.word_tokenize(d['description'])
        desc = [word for word in desc if word.isalnum()]
        length = len(desc)+1  # +1 for te EOS token
        keys.append(key)
        lengths.append(length)
    sorting_idx = np.argsort(np.array(lengths))[::-1]
    keys = [keys[i] for i in sorting_idx]
    lengths = [lengths[i] for i in sorting_idx]

    # NEGATIVE JSON DICT
    # command_token : [{box: [], color: [], location: [], action: [], class: [], count: []}]
    if negative is not None:
        max_negs = max([len(neg) for neg in negative.values()])

    # create all of the hdf5 datasets where variables are stored
    img = h5file.create_dataset('image_features', shape=(len(keys), 2048), dtype='f')
    obj = h5file.create_dataset('object_features', shape=(len(keys), 2048), dtype='f')
    box = h5file.create_dataset('bboxes', shape=(len(keys), 4), dtype='f')
    att = h5file.create_dataset('attributes', shape=(len(keys), len(attr_vocab)), dtype='i')
    at_loc = h5file.create_dataset('attr_location', shape=(len(keys), len(attr_vocab.loc_id2vocab_id)), dtype='f')
    at_col = h5file.create_dataset('attr_color', shape=(len(keys), len(attr_vocab.col_id2vocab_id)), dtype='f')
    at_act = h5file.create_dataset('attr_action', shape=(len(keys), len(attr_vocab.act_id2vocab_id)), dtype='f')
    gt_at_loc = h5file.create_dataset('gt_attr_location', shape=(len(keys), len(attr_vocab)), dtype='f')
    gt_at_col = h5file.create_dataset('gt_attr_color', shape=(len(keys), len(attr_vocab)), dtype='f')
    gt_at_act = h5file.create_dataset('gt_attr_action', shape=(len(keys), len(attr_vocab)), dtype='f')
    cls = h5file.create_dataset('classes', shape=(len(keys),), dtype='i')
    cnt = h5file.create_dataset('count', shape=(len(keys),), dtype='i')
    lns = h5file.create_dataset('lengths', shape=(len(keys),), dtype='i')
    des = h5file.create_dataset('descriptions', shape=(len(keys), args.max_len), dtype='i')
    # create negatives datasets
    num_neg = h5file.create_dataset('negative_num', shape=(len(keys),), dtype='i')
    neg_box = h5file.create_dataset('negative_box', shape=(len(keys), max_negs, 4), dtype='f')
    neg_obj = h5file.create_dataset('negative_object', shape=(len(keys), max_negs, 2048), dtype='f')
    neg_loc = h5file.create_dataset('negative_location', shape=(len(keys), max_negs,
                                                                len(attr_vocab.loc_id2vocab_id)), dtype='f')
    neg_col = h5file.create_dataset('negative_color', shape=(len(keys), max_negs,
                                                             len(attr_vocab.col_id2vocab_id)), dtype='f')
    neg_act = h5file.create_dataset('negative_action', shape=(len(keys), max_negs,
                                                              len(attr_vocab.act_id2vocab_id)), dtype='f')
    neg_cnt = h5file.create_dataset('negative_count', shape=(len(keys), max_negs), dtype='i')
    neg_cls = h5file.create_dataset('negative_class', shape=(len(keys), max_negs), dtype='i')
    mean_diff_feat = h5file.create_dataset('mean_difference_feat', shape=(len(keys), 2048), dtype='f')
    mean_diff_box = h5file.create_dataset('mean_difference_box', shape=(len(keys), 25), dtype='f')
    mean_neg_diff_feat = h5file.create_dataset('mean_neg_difference_feat', shape=(len(keys), max_negs, 2048), dtype='f')
    mean_neg_diff_box = h5file.create_dataset('mean_neg_difference_box', shape=(len(keys), max_negs, 25), dtype='f')
    if args.add_image:
        full_img_h5 = h5file.create_dataset('full_image', shape=(len(keys), 3, 224, 224), dtype='f')
        full_obj_h5 = h5file.create_dataset('full_object', shape=(len(keys), 3, 224, 224), dtype='f')
        full_neg_obj_h5 = h5file.create_dataset('negative_full_object', shape=(len(keys), max_negs, 3, 224, 224),
                                                dtype='f')

    times_list = []
    for i in trange(len(keys), desc='filling hdf5'):
        start = time.time()
        key = keys[i]
        len_i = lengths[i] if lengths[i] < args.max_len else args.max_len
        lns[i] = len_i
        # cls[i] = class_vocab.word2idx[data[key]['class']]
        cls[i] = class_vocab.word2idx[positive[key]['class']]
        cnt[i] = positive[key]['count']
        # process attributes
        # ## Old single attribute list version
        if args.old_attr:
            att_idxs = sorted([attr_vocab.word2idx[a] for a in data[key]['attrs']])
            att[i, att_idxs] = 1
        # ## new multi attribute list version
        else:
            at_loc[i] = positive[key]['location']
            at_col[i] = positive[key]['color']
            at_act[i] = positive[key]['action']
            att_idxs = [attr_vocab.word2idx[location[key]['gt']]]
            gt_at_loc[i, att_idxs] = 1
            att_idxs = [attr_vocab.word2idx[color[key]['gt']]]
            gt_at_col[i, att_idxs] = 1
            att_idxs = [attr_vocab.word2idx[action[key]['gt']]]
            gt_at_act[i, att_idxs] = 1

        # store negative data
        num_neg[i] = len(negative[key])
        for neg_i in range(len(negative[key])):
            neg_loc[i, neg_i] = negative[key][neg_i]['location']
            neg_col[i, neg_i] = negative[key][neg_i]['color']
            neg_act[i, neg_i] = negative[key][neg_i]['action']
            neg_cnt[i, neg_i] = negative[key][neg_i]['count']
            neg_cls[i, neg_i] = class_vocab.word2idx[negative[key][neg_i]['class']]

        # prepare images before running the resnet
        raw_image = Image.open(os.path.join(
            IMG_ROOT, data[key]['img']))

        img_w, img_h = raw_image.size
        process_imgs = []
        box_img = raw_image.crop((positive[key]['box'][0], positive[key]['box'][1],
                                 positive[key]['box'][0] + positive[key]['box'][2],
                                 positive[key]['box'][1] + positive[key]['box'][3]))
        process_imgs.append(normalize(transform(raw_image)))
        process_imgs.append(normalize(transform(box_img)))
        # also process negative box images
        for neg_i in range(len(negative[key])):
            if negative[key][neg_i]['box'][2] == 0 or negative[key][neg_i]['box'][3] == 0:
                # the width or height is equal to zero, so skip
                continue
            box_img = raw_image.crop((negative[key][neg_i]['box'][0], negative[key][neg_i]['box'][1],
                                      negative[key][neg_i]['box'][0] + negative[key][neg_i]['box'][2],
                                      negative[key][neg_i]['box'][1] + negative[key][neg_i]['box'][3]))
            process_imgs.append(normalize(transform(box_img)))
            neg_box[i][neg_i] = [negative[key][neg_i]['box'][0]/img_w, negative[key][neg_i]['box'][1]/img_h,
                                 (negative[key][neg_i]['box'][0]+negative[key][neg_i]['box'][2])/img_w,
                                 (negative[key][neg_i]['box'][1]+negative[key][neg_i]['box'][3])/img_h]

        # run te resnet
        if len(process_imgs) > 16:
            out = []
            for step_i in range(1, int(np.ceil(len(process_imgs)/16))+1):
                p_img = process_imgs[16*(step_i-1):16*(step_i)]
                out.append(resnet(torch.stack(p_img).to(dev)).detach().cpu())
            out = torch.cat(out, dim=0)
        else:
            out = resnet(torch.stack(process_imgs).to(dev)).detach().cpu()
        if args.add_image:
            full_img_h5[i] = process_imgs[0]
            full_obj_h5[i] = process_imgs[1]
        img[i] = out[0]
        obj[i] = out[1]
        box[i] = [positive[key]['box'][0]/img_w, positive[key]['box'][1]/img_h,
                  positive[key]['box'][2]/img_w, positive[key]['box'][3]/img_h]

        # Process negative data further
        list_feats = [out[1]]
        list_boxes = [positive[key]['box']]
        for neg_i in range(out.shape[0]-2):
            neg_obj[i, neg_i] = out[neg_i+2]
            if args.add_image:
                full_neg_obj_h5[i, neg_i] = process_imgs[neg_i+2]
            # add negatives to the lists for differences
            if negative[key][neg_i]['box'][2] == 0 or negative[key][neg_i]['box'][3] == 0:
                continue
            list_feats.append(out[neg_i+2])
            list_boxes.append([negative[key][neg_i]['box'][0], negative[key][neg_i]['box'][1],
                               negative[key][neg_i]['box'][2], negative[key][neg_i]['box'][3]])

        for feat_index in range(len(list_feats)):
            px, py, pw, ph = list_boxes[feat_index]
            pcx, pcy = px + pw / 2, py + ph / 2
            rest_feats = list_feats.copy()
            rest_boxes = list_boxes.copy()
            del rest_feats[feat_index]
            del rest_boxes[feat_index]
            list_distances = []
            for rest_b in rest_boxes:
                ax, ay = rest_b[0] + rest_b[2] / 2, \
                         rest_b[1] + rest_b[3] / 2
                list_distances.append((pcx - ax) ** 2 + (pcy - ay) ** 2)
            # create the diff features
            order_ids = np.argsort(list_distances)
            mean_l = []
            box_l = []
            for id_ in order_ids[:5]:
                mean_l.append(rest_feats[id_])
                nx, ny, nw, nh = rest_boxes[id_]
                box_l += [(nx-pcx)/pw, (ny-pcy)/ph, (nx+nw-pcx)/pw, (ny+nh-pcy)/ph, nw*nh/(pw*ph)]
            if len(mean_l) > 0:
                if feat_index == 0:
                    mean_diff_feat[i] = torch.mean(torch.cat(mean_l, dim=0), dim=0)
                    mean_diff_box[i, :len(box_l)] = box_l
                else:
                    neg_i = feat_index - 1
                    mean_neg_diff_feat[i, neg_i] = torch.mean(torch.cat(mean_l, dim=0), dim=0)
                    mean_neg_diff_box[i, neg_i, :len(box_l)] = box_l

        # process desc
        sent_list.append(data[key]['description'])
        desc = nltk.word_tokenize(data[key]['description'])
        desc = [word.lower() for word in desc if word.isalnum()]
        des[i, :] = vocab.to_indices(desc, append_sos=False, append_eos=True, max_length=15)
        # CHECK FROM OLD ERROR. FIRST TOKEN ALWAYS REPLACES BY UNK.
        if des[i, 0] == 4:
            print(nltk.word_tokenize(data[key]['description']), desc)
            exit(1)
        ad = {'image_id': i,
              'id': i,
              'caption': ' '.join(desc),
              'image_name': data[key]['img']}
        ann_dict['annotations'].append(ad)
        imd = {'id': i, 'image_name': data[key]['img']}
        ann_dict['images'].append(imd)
        times_list.append(time.time() - start)
    with open(out_sentences, 'w') as f:
        json.dump(sent_list, f)
    ann_file = os.path.join('processed', 'annotations', 't2c_anns_{}.json'.format(split))
    os.makedirs(os.path.dirname(ann_file), exist_ok=True)
    with open(ann_file, 'w') as ann_f:
        json.dump(ann_dict, ann_f)
    print("\n\n===================\n\n"
          "preprocess inference time: {}"
          "\n\n===================\n\n".format(np.mean(times_list)*1000))
