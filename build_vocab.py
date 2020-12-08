import re
import pickle
from collections import Counter
import nltk
from pycocotools.coco import COCO
import tqdm
from senticap_data_loader import SenticapReader
import argparse

class Vocab:
    '''vocabulary'''
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.ix = 0

    def add_word(self, word):
        if word not in self.w2i:
            self.w2i[word] = self.ix
            self.i2w[self.ix] = word
            self.ix += 1

    def extend(self, Vocab):
        current_length = self.__len__()
        assert self.ix == current_length
        for word in Vocab.w2i.keys():
            self.add_word(word)


    def __call__(self, word):
        if word not in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[word]

    def __len__(self):
        return len(self.w2i)


def flickrstyle_build_vocab(mode_list=['factual', 'humorous'], data_dir='./data/'):
    '''build vocabulary'''
    # define vocabulary
    vocab = Vocab()
    # add special tokens
    vocab.add_word('<pad>')
    vocab.add_word('<s>')
    vocab.add_word('</s>')
    vocab.add_word('<unk>')

    # add words
    for mode in mode_list:
        if mode == 'factual':
            captions = flickrstyle_extract_captions(mode=mode, data_dir=data_dir)
            words = nltk.tokenize.word_tokenize(captions)
            counter = Counter(words)
            words = [word for word, cnt in counter.items() if cnt >= 2]
        else:
            captions = flickrstyle_extract_captions(mode=mode, data_dir=data_dir)
            words = nltk.tokenize.word_tokenize(captions)

        for word in words:
            vocab.add_word(word)

    return vocab


def flickrstyle_extract_captions(mode='factual', data_dir='./data/'):
    '''extract captions from data files for building vocabulary'''
    text = ''
    if mode == 'factual':
        with open(data_dir+"factual/factual_train.txt", 'r') as f:
            res = f.readlines()

        r = re.compile(r'\d*.jpg,\d*')
        for line in res:
            line = r.sub(' ', line)
            line = line.replace('.', '')
            # remove first image file name
            line = ' '.join(line.split(' ')[1:])
            line = line.strip()
            text += line + ' '

    else:
        if mode == 'humorous':
            with open(data_dir+"humor/funny_train.txt", 'r') as f:
                res = f.readlines()
        else:
            with open(data_dir+"romantic/romantic_train.txt", 'r', encoding="latin1") as f:
                res = f.readlines()

        for line in res:
            line = line.replace('.', '')
            line = line.strip()
            text += line + ' '

    return text.strip().lower()



def COCO_build_vocab(dataDir='./data/COCO', dataType='train2014'):
    '''build vocabulary'''
    # define vocabulary
    vocab = Vocab()
    # add special tokens
    vocab.add_word('<pad>')
    vocab.add_word('<s>')
    vocab.add_word('</s>')
    vocab.add_word('<unk>')

    # initialize COCO api for caption annotations
    annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
    coco_caps = COCO(annFile)
    for anns in tqdm.tqdm(
            coco_caps.imgToAnns.values(),
            total=len(coco_caps.imgToAnns.values())):
        # anns is 5-element list
        for ann in anns:
            words = nltk.tokenize.word_tokenize(ann['caption'].strip().lower())
            for word in words:
                vocab.add_word(word)

    return vocab



def senticap_build_vocab(filename="./data/senticap_dataset/data/senticap_dataset.json"):
    '''build vocabulary'''
    # define vocabulary
    vocab = Vocab()
    # add special tokens
    vocab.add_word('<pad>')
    vocab.add_word('<s>')
    vocab.add_word('</s>')
    vocab.add_word('<unk>')

    sr = SenticapReader(filename)
    sr.readJson(filename)
    for im in sr.images:
        sentences = im.getSentences()
        for sent in sentences:
            for word in sent.tokens:
                vocab.add_word(word)

    return vocab

if __name__ == '__main__':
    vocab = Vocab()
    # add special tokens
    vocab.add_word('<pad>')
    vocab.add_word('<s>')
    vocab.add_word('</s>')
    vocab.add_word('<unk>')
    print(vocab.__len__())

    flickrstyle_vocab = flickrstyle_build_vocab(mode_list=['factual', 'humorous', 'romantic'], data_dir="data/FlickrStyle/FlickrStyle_v0.9/")
    print(flickrstyle_vocab.__len__())

    vocab.extend(flickrstyle_vocab)
    print(vocab.__len__())

    COCO_vocab = COCO_build_vocab(dataDir='./data/COCO', dataType='train2014')
    print(COCO_vocab.__len__())

    vocab.extend(COCO_vocab)
    print(vocab.__len__())

    senticap_vocab = senticap_build_vocab(filename="./data/senticap_dataset/data/senticap_dataset.json")
    print(senticap_vocab.__len__())

    vocab.extend(senticap_vocab)
    print(vocab.__len__())

    with open('data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
