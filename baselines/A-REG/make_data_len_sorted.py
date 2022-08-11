import h5py
import json
from tqdm import tqdm
import numpy as np


sort_data = {}

with open('/cw/liir/NoCsBack/testliir/datasets/refer_expr/vg/new_vg_data.json', 'r') as f:
    data = json.load(f)
sort_data['vg_attrs'] = data['vg_attrs']
sort_data['vg_classes'] = data['vg_classes']
sort_data['data'] = {}

with h5py.File('/cw/liir/NoCsBack/testliir/datasets/refer_expr/vg/features.h5', 'r') as f:
    with h5py.File('/cw/liir/NoCsBack/testliir/datasets/refer_expr/vg/sorted_features.h5', 'w') as sort_f:
        new_img = sort_f.create_dataset_like('img', f['img'])
        new_reg = sort_f.create_dataset_like('region', f['region'])
        new_attr = sort_f.create_dataset_like('attr_prob', f['attr_prob'])
        new_phr = sort_f.create_dataset_like('phrases', f['phrases'])
        new_len = sort_f.create_dataset_like('phrase_lengths', f['phrase_lengths'])

        sorting_idx = np.argsort(np.array([f['phrase_lengths'][d['hdf5_ix']] for d in tqdm(data['data'],
                                                                                           desc='get sort')]))

        h5_ix = 0
        img_h5_ix = 0
        old_new_h5_ix = {}
        old_new_img_h5_ix = {}
        for i, idx in tqdm(enumerate(sorting_idx), desc='sorting'):
            d = data['data'][idx]
            # fix the hdf5_ix
            old_h5 = d['hdf5_ix']
            old_img_h5 = d['img_hdf5_ix']
            if old_h5 in old_new_h5_ix:
                d['hdf5_ix'] = old_new_h5_ix[old_h5]
            else:
                old_new_h5_ix[old_h5] = h5_ix
                d['hdf5_ix'] = h5_ix
                h5_ix += 1

            # fix the img_hdf5_ix
            if old_img_h5 in old_new_img_h5_ix:
                d['img_hdf5_ix'] = old_new_img_h5_ix[old_img_h5]
            else:
                old_new_img_h5_ix[old_img_h5] = img_h5_ix
                d['img_hdf5_ix'] = img_h5_ix
                img_h5_ix += 1
            sort_data['data'][i] = d
            # copy over the hdf5 data to the correct indices
            new_img[d['img_hdf5_ix']] = f['img'][old_img_h5]
            new_reg[d['hdf5_ix']] = f['region'][old_h5]
            new_attr[d['hdf5_ix']] = f['attr_prob'][old_h5]
            new_phr[d['hdf5_ix']] = f['phrases'][old_h5]
            new_len[d['hdf5_ix']] = f['phrase_lengths'][old_h5]
with open('/cw/liir/NoCsBack/testliir/datasets/refer_expr/vg/sorted_data.json', 'w') as f:
    json.dump(sort_data, f)
