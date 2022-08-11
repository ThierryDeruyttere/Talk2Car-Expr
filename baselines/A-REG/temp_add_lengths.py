import h5py
import json
from nltk import word_tokenize
from tqdm import tqdm
from vocabulary import Vocabulary
import numpy as np

with h5py.File('/cw/liir/NoCsBack/testliir/datasets/refer_expr/vg/features.h5', 'r+') as f:
    if 'phrase_lengths' in f.keys():
        del f['phrase_lengths']
    data = np.sum(f['phrases'][:] > 0, axis=1)
    pl = f.create_dataset('phrase_lengths', dtype=int, data=data)
