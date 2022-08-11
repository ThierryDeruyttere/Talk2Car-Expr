import h5py
import json
from nltk import word_tokenize
from tqdm import tqdm
from vocabulary import Vocabulary
import numpy as np

with open('/cw/liir/NoCsBack/testliir/datasets/refer_expr/vg/sorted_data.json', 'r') as f:
    data = json.load(f)

max_len = max(len(word_tokenize(data['data'][d_key]['phrase'])) for d_key in tqdm(data['data'].keys(),
                                                                                  desc='find lengths'))
vocab = Vocabulary('/cw/liir/NoCsBack/testliir/datasets/embeddings/glove.840B.300d.txt')
with h5py.File('/cw/liir/NoCsBack/testliir/datasets/refer_expr/vg/sorted_features.h5', 'r+') as f:
    if 'phrases' in f.keys():
        del f['phrases']
    phrases = f.create_dataset('phrases', shape=(len(data['data']), max_len+2), dtype=int)
    for d_key in tqdm(data['data'].keys(), desc='fill dataset'):
        d = data['data'][d_key]
        phrase = word_tokenize(d['phrase'])
        phrase = vocab.to_indices(phrase, append_eos=True, append_sos=True)
        phrases[d['hdf5_ix'], :len(phrase)] = phrase
    if 'phrase_lengths' in f.keys():
        del f['phrase_lengths']
    pl = f.create_dataset('phrase_lengths', dtype=int, data=np.sum(f['phrases'][:] > 0, axis=1))
