import os
import torch
import pickle
import json
from collections import Counter
from typing import Union
from nltk import word_tokenize
import numpy as np
from tqdm import tqdm


class Glove:
    def __init__(self, emb_file):
        self.embs = {}
        self.process(emb_file)

    def process(self, emb_file):
        with open(emb_file, 'r') as f:
            i = -1
            for line in tqdm(f, desc='loading Glove embeddings'):
                i += 1
                elems = line.split(' ')
                self.embs[elems[0]] = np.array(elems[1:]).astype(np.float)


class Vocabulary:
    def __init__(self, emb_file=None, threshold=5, max_tokens=0, vocab_name='vocab', dataset='vg',
                 only_pad=False, no_pad=False, rootdir=None, keyword_vocab: list = None) -> None:
        """
        Generic class for storing the vocabulary of a dataset.
        """
        super().__init__()
        # check if a file exists for the current settings. Note: when it does not exist, some commands
        # must be run manually
        # assert conf.vocabulary.pad_token != "" and conf.vocabulary.eos_token != "" \
        #        and conf.vocabulary.sos_token != "" and conf.vocabulary.unk_token != "" \
        #        and conf.vocabulary.sep_token != "", "the vocab tokens should be filled in the config"
        self.no_pad = no_pad
        self.only_pad = only_pad
        self.dataset = dataset
        self.emb_file = emb_file
        self.weights = []
        self.pad_token = '<pad>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        self.sep_token = '<sep>'
        file_name = '{}.dataset-{}.thresh-{}.max_tokens-{}.pkl'.format(vocab_name, dataset, threshold, max_tokens)
        if rootdir is None:
            self.vocab_file = os.path.join('processed', 'vocabulary', file_name)
        else:
            # print('adding rootdir \"{}\"'.format(rootdir))
            self.vocab_file = os.path.join(rootdir, 'processed', 'vocabulary', file_name)
        if os.path.exists(self.vocab_file):
            self.load()
        else:
            print('No precomputed Vocabulary {} file exists. Constructing it'.format(vocab_name))
            embeddings = Glove(emb_file)
            os.makedirs(os.path.dirname(self.vocab_file), exist_ok=True)
            # create variables for the special tokens
            # create empty dicts and lists for storing words and index mappings
            self.word2idx, self.words, self.counts = {}, {}, {}
            self.create_words = []
            # count special and embedding tokens counters
            self.num_embs = 0
            self.num_special = 0
            # store the special tokens in the dictionary and get their index
            # create pad
            if not no_pad:
                self.add_word(self.pad_token, special=True, n=999999999)
                self.pad_idx = self.word2idx[self.pad_token]
            # create sos
            if not only_pad:
                self.add_word(self.sos_token, special=True, n=999999998)
                self.sos_idx = self.word2idx[self.sos_token]
            # create eos
            if not only_pad:
                self.add_word(self.eos_token, special=True, n=999999997)
                self.eos_idx = self.word2idx[self.eos_token]
            # create sep
            if not only_pad:
                self.add_word(self.sep_token, special=True, n=999999996)
                self.sep_idx = self.word2idx[self.sep_token]
            # create unk
            if not only_pad:
                self.add_word(self.unk_token, special=True, n=999999995)
                self.unk_idx = self.word2idx[self.unk_token]
            # define the Vocabulary Threshold and minimum number of word occurrences
            self.threshold = threshold
            self.max_tokens = max_tokens
            self.fill_vocab(embeddings)
            self.finalize(keyword_vocab)
            self.save()
            print('Vocabulary finished and saved.')

    def __len__(self):
        """
        :return: The number of words in the vocabulary.
        """
        return len(self.words)

    def __getitem__(self, idx: int) -> str:
        """
        Get the word attached to a specific index.
        :param idx: The index of a word in the vocabulary
        :return: The word that is attached to this index, or the unknown token
        """
        return self.words.get(idx, self.unk_token)

    def index(self, word: str) -> int:
        """
        Get the index attached to a specific word.
        :param word: A word to search in the vocabulary
        :return: The index that is attached to this index, or the unknown token index
        """
        return self.word2idx.get(word, self.unk_idx)

    def to_indices(self, tokens: list, append_sos: bool = False, append_eos: bool = False,
                   max_length=None) -> torch.Tensor:
        """
        Turn a given string into a tensor of the indices of the tokens in the string.
        :param tokens     : Sequence to change to indices.
        :param append_sos : Whether to add the sos token at the beginning of the sequence.
        :param append_eos : Whether to add the eos token at the beginning of the sequence.
        :param max_length : the max length including SOS and EOS if they are appended.
        :return: A tensor with the indices of the tokens in the string
        """
        if max_length is None:
            length = len(tokens) + append_sos + append_eos
        else:
            length = max_length
        ids = torch.full((length,), fill_value=self.word2idx[self.pad_token], dtype=torch.long)
        i = 0
        if append_sos:
            ids[i] = self.sos_idx
        for i, token in enumerate(tokens, start=append_sos):
            ids[i] = self.index(token)
            if i == (length-1)-append_eos:  # list start at 0/if eos needed, leave space
                break
        if append_eos:
            ids[i+1] = self.eos_idx
        return ids

    def to_string(self, tensor, join=True):
        """
        Turn a given tensor sequence of indices into a string.
        :param tensor: The tensor to transform into a string.
        :return:
        """
        # if the tensor consists of multiple sequences, transform each one separately.
        if (torch.is_tensor(tensor) and tensor.dim() > 1) or (isinstance(tensor, np.ndarray) and tensor.ndim > 1) \
                or (isinstance(tensor, list) and isinstance(tensor[0], list)):
            # if join:
            #     return '\n'.join(self.to_string(t, join) for t in tensor)
            # else:
            return [self.to_string(t, join) for t in tensor]
        if join:
            sentence = ' '.join(self[int(i)] for i in tensor if i != self.eos_idx)
            # remove the padding from the string
            sentence = (sentence + ' ').replace(self.pad_token, '').rstrip()
        else:
            sentence = [self[int(i)] for i in tensor if i != self.eos_idx]
        return sentence

    def add_word(self, word: str, n: int = 1, special: bool = False, emb: bool = False, weight=None):
        """
        Add a word to the vocabulary. Also adds it to the index mapping and counts it.
        :param word:  The word to store in the vocabulary.
        :param n: For how many counts it has to be added.
        :param special: Whether it is a special token kept at the beginning of the dict
        :param emb: Whether the word has an pretrained embedding associated to it
        """
        # test if the word variable is None, also return None
        if word is None or word == "":
            return None
        # test if it already exists
        word = word.lower()
        if emb or special:
            if word not in self.word2idx.keys():
                # add the word to the lists and dictionaries and add at the count.
                if emb:
                    self.num_embs += 1
                    self.weights.append(weight)
                else:
                    self.num_special += 1
                    if self.emb_file is not None:
                        self.weights.append([0]*300)
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.words[idx] = word
                self.counts[idx] = n
            else:
                # add the count to the word
                idx = self.word2idx[word]
                self.counts[idx] += n
        else:
            self.create_words.append(word)

    def fill_with_vg(self, embeddings=None):
        with open('/cw/liir/NoCsBack/testliir/datasets/visual_genome/region_descriptions.json', 'r') as f:
            images = json.load(f)
        for image in tqdm(images, desc='process image region desc'):
            for region in image['regions']:
                description = region['phrase'].lower()
                tokens = word_tokenize(description)
                for token in tokens:
                    if token in embeddings.embs:
                        self.add_word(token, n=1, emb=True, weight=embeddings.embs[token])
                    else:
                        self.add_word(token, n=1, emb=False)

    def fill_with_t2c(self, embeddings=None):
        for file_ in ['/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/talk2car_descriptions_train.json',
                      '/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/talk2car_descriptions_val.json',
                      '/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/talk2car_descriptions_test.json']:
            with open(file_, 'r') as f:
                data = json.load(f)
            for region in tqdm(data.values(), desc='process image region desc'):
                description = region['description'].lower()
                tokens = word_tokenize(description)
                for token in tokens:
                    if token.isalnum():
                        if token in embeddings.embs:
                            self.add_word(token, n=1, emb=True, weight=embeddings.embs[token])
                        elif 'train' in file_:
                            self.add_word(token, n=1, emb=False)

    def fill_vocab(self, embeddings=None):
        if self.dataset == 'vg':
            self.fill_with_vg(embeddings=embeddings)
        elif self.dataset == 't2c':
            self.fill_with_t2c(embeddings=embeddings)
        else:
            exit('Unknown dataset')

    def finalize(self, keyword_vocab: list=None):
        """
        Finish building the vocabulary by discarding words that do not fit the requirements
        """

        # if there is no maximum size, use all the words.
        if self.max_tokens <= 0:
            num_words = len(self.create_words)+self.num_special+self.num_embs
        else:
            num_words = self.max_tokens

        # create a counter for all the words in the vocabulary that are not the special tokens or have embeddings
        counter = Counter(self.create_words)

        # sort the words by times occuring and only take the top <num_words>
        print('finalizing vocabulary.')
        for word, count in counter.most_common(num_words-self.num_special-self.num_embs):
            # if they occur more often then the threshold and it is not added already (because of emb or special),
            # add them to the finalized vocabulary
            if count >= self.threshold and word not in self.words.values():
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.words[idx] = word
                self.counts[idx] = count
                if self.weights:
                    self.weights.append([0]*300)
        sort_idx = np.argsort(np.array(list(self.counts.values())))[::-1]
        tmp_weights = []
        tmp_w2i = {}
        tmp_w = {}
        tmp_c = {}
        j = 0
        for i, idx in tqdm(enumerate(sort_idx), desc='finish sorting'):
            tmp_weights.append(self.weights[idx])
            w = self.words[idx]
            tmp_w[i] = w
            tmp_w2i[w] = i
            tmp_c[i] = self.counts[idx]
            j = i
        if keyword_vocab is not None:
            for keywords in keyword_vocab:
                for key_w, key_ind in keywords.word2idx.items():
                    if key_w not in tmp_w2i:
                        j += 1
                        tmp_w2i[key_w] = j
                        tmp_w[j] = key_w
                        tmp_c[j] = 1
                        tmp_weights.append(list(keywords.weights[key_ind]))
        self.weights = np.array(tmp_weights)
        self.word2idx = tmp_w2i
        self.words = tmp_w
        self.counts = tmp_c
        if not self.no_pad:
            self.pad_idx = self.word2idx[self.pad_token]
            assert self.pad_token == self.words[self.pad_idx]
        if not self.only_pad:
            self.sos_idx = self.word2idx[self.sos_token]
            self.eos_idx = self.word2idx[self.eos_token]
            self.unk_idx = self.word2idx[self.unk_token]
            self.sep_idx = self.word2idx[self.sep_token]
            # do a final quick fail check
            assert self.sos_token == self.words[self.sos_idx]
            assert self.eos_token == self.words[self.eos_idx]

    def load(self):
        with open(self.vocab_file, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.clear()
        self.__dict__.update(tmp_dict)

    def save(self):
        with open(self.vocab_file, 'wb') as f:
            pickle.dump(self.__dict__, f)


class KeywordVocabulary(Vocabulary):
    def __init__(self, emb_file=None, threshold=0, max_tokens=0, dataset='vg', vocab_name='keyword_vocab', data='attributes',
                 rootdir=None, multi_file=False) -> None:
        """
        Generic class for storing the vocabulary of a dataset.
        """
        assert data in ['attributes', 'classes'], \
            'given data subset is unknown. Should be one of: \'attributes\',\'classes\''
        self.data = data
        self.multi_file = multi_file
        self.loc_id2vocab_id = {}
        self.col_id2vocab_id = {}
        self.act_id2vocab_id = {}
        if not self.multi_file and data == 'attributes':
            vocab_name = 'OLD-ATTRIBUTES-{}-{}'.format(vocab_name, data)
        else:
            vocab_name = '{}-{}'.format(vocab_name, data)
        super().__init__(emb_file, threshold, max_tokens, dataset=dataset, vocab_name=vocab_name,
                         only_pad=True, no_pad=True, rootdir=rootdir)

    def add_word(self, word: str, n: int = 1, special: bool = False, emb: bool = False, weight=None):
        """
        Add a word to the vocabulary. Also adds it to the index mapping and counts it.
        :param word:  The word to store in the vocabulary.
        :param n: For how many counts it has to be added.
        :param special: Whether it is a special token kept at the beginning of the dict
        :param emb: Whether the word has an pretrained embedding associated to it
        """
        # test if the word variable is None, also return None
        if word is None or word == "":
            return None
        # test if it already exists
        word = word.lower()
        if emb or special:
            if word not in self.word2idx.keys():
                # add the word to the lists and dictionaries and add at the count.
                if emb:
                    self.num_embs += 1
                    self.weights.append(weight)
                else:
                    self.num_special += 1
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.words[idx] = word
                self.counts[idx] = n
            else:
                # add the count to the word
                idx = self.word2idx[word]
                self.counts[idx] += n
        else:
            self.create_words.append(word)

    def fill_vocab(self, embeddings=None):
        if self.dataset == 'vg':
            with open('/cw/liir/NoCsBack/testliir/datasets/refer_expr/vg/new_vg_data.json', 'r') as f:
                data = json.load(f)
            if self.data == 'attributes':
                l = data['vg_attrs']
            else:
                l = data['vg_classes']
        elif self.dataset == 't2c':
            if self.data == 'attributes':
                if self.multi_file:
                    enum_i = 0
                    for file_, d in zip(['/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/t2c_location_voc.json',
                                         '/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/t2c_color_voc.json',
                                         '/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/t2c_action_voc.json'],
                                        [self.loc_id2vocab_id, self.col_id2vocab_id, self.act_id2vocab_id]):
                        with open(file_, 'r') as f:
                            l = json.load(f)
                        group_id = 0
                        for word in l:
                            word = word.lower()
                            if word in embeddings.embs:
                                print("embedding found")
                                weight = embeddings.embs[word]
                            # failsafe to capture keywords with spaces, such as "in front"
                            elif word != 'not appicable' and ' ' in word and word.split(' ')[-1] in embeddings.embs:
                                weight = embeddings.embs[word.split(' ')[-1]]
                            else:
                                print("NOT FOUND!!: "+word)
                                weight = list(np.random.normal(0, 0.01, 300))
                            self.add_word(word, n=99999-enum_i, emb=True, weight=weight)
                            enum_i += 1
                            d[group_id] = self.word2idx[word]
                            group_id += 1
                    return
                else:
                    with open('/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/attributes.json', 'r') as f:
                        l = json.load(f)
            else:
                with open('/cw/liir/NoCsBack/testliir/datasets/refer_expr/Talk2Car/talk2car_classes.json', 'r') as f:
                    l = json.load(f)
        else:
            exit('unknown dataset')
        for enum_i, word in enumerate(l):
            word = word.lower()
            synset_subs = word.split('.')
            if word in embeddings.embs:
                weight = embeddings.embs[word]
            elif synset_subs[-1] in embeddings.embs:
                weight = embeddings.embs[synset_subs[-1]]
            else:
                weight = list(np.random.normal(0, 0.01, 300))
            self.add_word(word, n=999999-enum_i, emb=True, weight=weight)
