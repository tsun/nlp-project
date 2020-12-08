import random
import numpy as np
import torch
import os
import os.path as osp
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter
logger_init = False

def init_logger(_log_file, use_file_logger=True, dir='log/', suffix=''):
    log_file = osp.join(dir, _log_file + suffix + '.log')
    #logging.basicConfig(filename=log_file, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',level=logging.DEBUG)
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    logger.addHandler(chlr)
    if use_file_logger:
        fhlr = logging.FileHandler(log_file)
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)

    global logger_init
    logger_init = True

def init_tensorboard_writer(dir='tensorboard/', _writer_file=None):
    writer = SummaryWriter(osp.join(dir, _writer_file))
    return writer

# set random number generators' seeds
def resetRNGseed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_sentence(id_list, vocab):
    '''
    convert the given ids to the sentence
    :param id_list: a list of word ids
    :param vocab:
    :return: a string
    '''
    sentence = ''
    for id in id_list:
        word = vocab.i2w[id]
        sentence += word
        sentence += ' '
    return sentence

def get_sentence_list(id_list, vocab):
    '''
    convert the given ids to the sentence
    :param id_list: a list of word ids
    :param vocab:
    :return: a list
    '''
    sentence = []
    for id in id_list:
        word = vocab.i2w[id]
        sentence.append(word)
    return sentence

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def compute_soft_embedding(logits, temperature, word_embedding, epsilon=1e-7):
    '''
    compute the embedding of soft word mentioned in 3.8
    :param logits: direct outputs from forward function of caption generator, which is Wo[c_t;, h_t] in eq3
        shape: (batch_size, seq_len, vocab_size)
    :param temperature: temperature \tau in eq 3
    :param word_embedding: nn.Embedding
    :param epsilon: a very small value in case temperature is very small ~ 0
    :return: the corresponding embedding for logits
    '''
    # first, compute p_t in eq3
    p = torch.softmax(logits / (temperature + epsilon), dim=-1)
    # then treat p as a soft word and compute the word embedding
    emb = torch.matmul(p, word_embedding.weight)
    return emb

def save_model(word_embedding, style_embedding, captionGenerator, imageEncoder, args, iter=-1, is_pretrain=False):
    # iter==-1 means last iteration
    if not os.path.exists(os.path.join(args.model_path, args.pretrained_timestamp if is_pretrain else args.timestamp)):
        os.makedirs(os.path.join(args.model_path, args.pretrained_timestamp if is_pretrain else args.timestamp))
    word_embedding_file_name = os.path.join(args.model_path, args.pretrained_timestamp if is_pretrain else args.timestamp,
                                            '{}word_embedding{}.pth'.format('' if not is_pretrain else 'pretrained_', '' if iter==-1 else '_{}'.format(iter)))
    torch.save(word_embedding.state_dict(), word_embedding_file_name)
    logging.info('Word embedding saved to {}\n'.format(word_embedding_file_name))

    style_embedding_file_name = os.path.join(args.model_path, args.pretrained_timestamp if is_pretrain else args.timestamp,
                                             '{}style_embedding{}.pth'.format('' if not is_pretrain else 'pretrained_', '' if iter==-1 else '_{}'.format(iter)))
    torch.save(style_embedding.state_dict(), style_embedding_file_name)
    logging.info('Style embedding saved to {}\n'.format(style_embedding_file_name))

    checkpoint_file_name = os.path.join(args.model_path, args.pretrained_timestamp if is_pretrain else args.timestamp,
                                        '{}caption_generator{}.pth'.format('' if not is_pretrain else 'pretrained_', '' if iter==-1 else '_{}'.format(iter)))
    torch.save(captionGenerator.state_dict(), checkpoint_file_name)
    logging.info('Caption generator saved to {}\n'.format(checkpoint_file_name))

    checkpoint_file_name = os.path.join(args.model_path, args.pretrained_timestamp if is_pretrain else args.timestamp,
                                        '{}image_encoder{}.pth'.format('' if not is_pretrain else 'pretrained_', '' if iter==-1 else '_{}'.format(iter)))
    torch.save(imageEncoder.state_dict(), checkpoint_file_name)
    logging.info('Image encoder saved to {}\n'.format(checkpoint_file_name))