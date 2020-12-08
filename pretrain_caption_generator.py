'''
for pretrain the caption generator
currently only use 3 style, 0-factual, 1-humor, 2-roman
'''

import os
import pickle
import argparse
import torch
import tqdm
from torch.autograd import Variable
import torch.optim as optim
from build_vocab import Vocab
from flickrstyle_data_loader import get_data_loader as flickrstyle_get_data_loader
from flickrstyle_data_loader import get_styled_data_loader as flickrstyle_get_styled_data_loader
from coco_data_loader import get_data_loader as COCO_get_data_loader
from senticap_data_loader import get_data_loader as senticap_get_data_loader
from utils import *
from model import *
from loss import *
import logging
import time
timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())

def pretrainer(args):
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # load vocablary
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)
    # print('Vocabulary loaded from {}'.format(args.vocab_path))
    # print('Vocabulary size: {}'.format(vocab_size))
    logging.info('Vocabulary loaded from {}'.format(args.vocab_path))
    logging.info('Vocabulary size: {}'.format(vocab_size))

    # paired data: COCO dataloader
    COCO_data_loader = COCO_get_data_loader(args.COCO_dir, args.COCO_dataType, vocab, require_img=True,
                                            batch_size=args.paired_batch_size, shuffle=True)
    coco_data_iter = iter(COCO_data_loader)

    # unpaired data
    # flickrstyle dataloader
    img_path = os.path.join(args.flickrstyle_dir, 'Images')
    factual_cap_path = os.path.join(args.flickrstyle_dir, 'factual/factual_train.txt')
    humorous_cap_path = os.path.join(args.flickrstyle_dir, 'humor/funny_train.txt')
    romantic_cap_path = os.path.join(args.flickrstyle_dir, 'romantic/romantic_train.utf.txt')
    flickrstyle_factual_data_loader_train = flickrstyle_get_data_loader(img_path, factual_cap_path, vocab, require_img=False,
                                                                  batch_size=args.unpaired_batch_size,
                                                                  shuffle=True, split='Train')
    flickrstyle_humorous_data_loader_train = flickrstyle_get_styled_data_loader(humorous_cap_path, vocab,
                                                                  batch_size=args.unpaired_batch_size,
                                                                  shuffle=True, split='Train')
    flickrstyle_romantic_data_loader_train = flickrstyle_get_styled_data_loader(romantic_cap_path, vocab,
                                                                  batch_size=args.unpaired_batch_size,
                                                                  shuffle=True, split='Train')

    style_data_loader_train = [flickrstyle_factual_data_loader_train, flickrstyle_humorous_data_loader_train,
                               flickrstyle_romantic_data_loader_train]
    style_data_iter_train = [iter(flickrstyle_factual_data_loader_train), iter(flickrstyle_humorous_data_loader_train),
                             iter(flickrstyle_romantic_data_loader_train)]

    # # senticap dataloader
    # senticap_data_loader = senticap_get_data_loader(args.senticap_dir, args.senticap_img_dir, vocab, require_img=False,
    #                                              batch_size=args.language_batch_size, shuffle=True)

    # define global embeddings
    word_embedding = nn.Embedding(len(vocab), args.emb_dim)
    style_embedding = nn.Embedding(args.style_num, args.emb_dim)
    if torch.cuda.is_available():
        word_embedding = word_embedding.cuda()
        style_embedding = style_embedding.cuda()

    # import models
    imageEncoder = ImageEncoder(args.emb_dim)
    captionGenerator = CaptionGenerator(word_embedding, style_embedding, args.CaptionGenerator_lstm_hidden_dim)

    if args.continue_training:
        logging.info('continue training from pretrained models and embeddings')
        word_embedding_file_name = os.path.join(args.model_path, 'pretrained_word_embedding.pth')
        style_embedding_file_name = os.path.join(args.model_path, 'pretrained_style_embedding.pth')
        checkpoint_file_name = os.path.join(args.model_path, 'pretrained_caption_generator.pth')
        word_embedding.load_state_dict(torch.load(word_embedding_file_name))
        style_embedding.load_state_dict(torch.load(style_embedding_file_name))
        captionGenerator.load_state_dict(torch.load(checkpoint_file_name))

    if torch.cuda.is_available():
        imageEncoder = imageEncoder.cuda()
        captionGenerator = captionGenerator.cuda()

    # optimizer for pretraining
    optimizer_cap = optim.Adam(list(imageEncoder.parameters())
                               + list(captionGenerator.parameters()), lr=args.lr_pretrain_caption)

    logging.info('start pretraining ...')
    for i in tqdm.tqdm(range(args.maximum_pretrain_iter), total=args.maximum_pretrain_iter):
        imageEncoder.train()
        imageEncoder.zero_grad()
        captionGenerator.train()
        captionGenerator.zero_grad()

        # use paired data (factual) from coco
        try:
            coco_data = next(coco_data_iter)
        except:
            coco_data_iter = iter(COCO_data_loader)
            coco_data = next(coco_data_iter)
        img = coco_data[0].float()
        cap = coco_data[1].long()
        cap_len = coco_data[2].long()
        style_code = torch.zeros(cap.shape[0]).long()  # 0 for fact
        if torch.cuda.is_available():
            img = img.cuda()
            cap = cap.cuda()
            # cap_len = cap_len.cuda()  # this is done in the masked_cross_entropy
            style_code = style_code.cuda()
        # get the caption result with guided caption
        img_emb = imageEncoder(img)
        pred_cap = captionGenerator(img_emb, style_code, guided_caption=cap)
        # need to explicitly skip the first token to avoid nan problem
        pred_cap_pair_loss = masked_cross_entropy(pred_cap[:, 1:, :].contiguous(), cap[:, 1:].contiguous(), cap_len - 1)
        pred_cap_pair_loss.backward()
        optimizer_cap.step()

        # use unpaired style data to train
        # sample factual data

        pred_cap_style_loss_all = []
        for style_id in range(args.style_num):
            imageEncoder.zero_grad()
            captionGenerator.zero_grad()
            try:
                style_data = next(style_data_iter_train[style_id])
            except:
                style_data_iter_train[style_id] = iter(style_data_loader_train[style_id])
                style_data = next(style_data_iter_train[style_id])
            cap = style_data[0].long()
            cap_len = style_data[1].long()
            style_code = torch.LongTensor([style_id]).repeat(cap.shape[0]).long()
            if torch.cuda.is_available():
                cap = cap.cuda()
                style_code = style_code.cuda()
            pred_cap_style = captionGenerator(None, style_code, guided_caption=cap)
            pred_cap_style_loss = masked_cross_entropy(pred_cap_style[:, 1:, :].contiguous(), cap[:, 1:].contiguous(),
                                                       cap_len - 1)* args.unpaired_loss_weight
            pred_cap_style_loss.backward()
            pred_cap_style_loss_all.append(pred_cap_style_loss.item())
            optimizer_cap.step()

        if i % args.log_step == 0 or i==args.maximum_pretrain_iter-1:
            # print('\n[pretrain] Iter {}: paired loss: {:.6f} style loss: {:.6f} {:.6f} {:.6f}\n'.format(
            #     i + 1, pred_cap_pair_loss.item(), *pred_cap_style_loss_all))
            logging.info('\n[pretrain] Iter {}: paired loss: {:.6f} style loss: {:.6f} {:.6f} {:.6f}\n'.format(
                 i+1, pred_cap_pair_loss.item(), *pred_cap_style_loss_all))

        if i % args.save_step == 0:
            # save the word embedding, style embedding and image caption generator
            save_model(word_embedding, style_embedding, captionGenerator, imageEncoder, args, iter=i, is_pretrain=True)

            # for quick debug, save current models are last models
            save_model(word_embedding, style_embedding, captionGenerator, imageEncoder, args, is_pretrain=True)

        # sample one image to see the result
        with torch.no_grad():
            imageEncoder.eval()
            captionGenerator.eval()
            sample_img_emb = imageEncoder(img[0].unsqueeze(0))
            sample_style_code = torch.zeros(1).long()  # 0 for fact
            if torch.cuda.is_available():
                sample_style_code = sample_style_code.cuda()
            sample_cap = captionGenerator.sample(sample_img_emb, sample_style_code)
            if i % args.log_step == 0 or i == args.maximum_pretrain_iter - 1:
                logging.info('sample results: \n{}\n'.format(get_sentence(sample_cap, vocab)))
            else:
                print('sample results: \n{}\n'.format(get_sentence(sample_cap, vocab)))

    # save the word embedding, style embedding and image caption generator
    save_model(word_embedding, style_embedding, captionGenerator, imageEncoder, args, is_pretrain=True)

def pretrain():
    parser = argparse.ArgumentParser(
        description='MSCap: Multi-Style Image Captioning with Unpaired Stylized Text')

    # environment
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='which gpu to use')
    parser.add_argument('--random_seed', default='0', type=int,
                        help='random seed')
    parser.add_argument('--timestamp', type=str,
                        help='timestamp')

    # tensorboard and logger
    parser.add_argument('--use_file_logger', default='True', type=str2bool,
                        help='whether use file logger')
    parser.add_argument('--use_tensorboard', default='True', type=str2bool,
                        help='whether use tensorboard')
    parser.add_argument('--tensorboard_dir', default='tensorboard_debug', type=str,
                        help='tensorboard directory')
    parser.add_argument('--writer', default=None, type=SummaryWriter,
                        help='tensorboard writer')

    parser.add_argument('--model_path', type=str, default='checkpoint',
                        help='path for saving trained models')
    parser.add_argument('--pretrained_timestamp', type=str, default='',
                        help='timestamp for trained models')
    parser.add_argument('--load_pretrained', action='store_true',
                        help='load pretrained image caption generator and embedding')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabrary')
    parser.add_argument('--flickrstyle_dir', type=str,
                        default='./data/FlickrStyle/FlickrStyle_v0.9',
                        help='path for flickrstyle root dir')
    parser.add_argument('--COCO_dir', type=str,
                        default='./data/COCO',
                        help='path for COCO root dir')
    parser.add_argument('--senticap_dir', type=str,
                        default='./data/senticap_dataset',
                        help='path for senticap root dir')

    parser.add_argument('--senticap_img_dir', type=str,
                        default='./data/COCO/train2014',
                        help='path for senticap img dir')

    parser.add_argument('--COCO_dataType', type=str,
                        default='train2014',
                        help='datatype for COCO, train2014/val2014')

    parser.add_argument('--style_num', type=int, default=3,
                        help='number of styles')
    parser.add_argument('--style_name_list', type=list, default=['Fact', 'Humor', 'Roman'],
                        help='list of styles name')

    parser.add_argument('--continue_training', action='store_true', help='load the pretrained result and update')
    parser.add_argument('--maximum_pretrain_iter', type=int, default=40000)
    parser.add_argument('--maximum_iter', type=int, default=10000)

    parser.add_argument('--paired_batch_size', type=int, default=64,
                        help='mini batch size for paired model training')
    parser.add_argument('--unpaired_batch_size', type=int, default=64,
                        help='mini batch size for unpaired model training')
    parser.add_argument('--unpaired_loss_weight', type=float, default=0.1,
                        help='weight of unpaired data loss')

    parser.add_argument('--emb_dim', type=int, default=512,
                        help='embedding size of word, image')

    parser.add_argument('--BackTranslator_gru_hidden_dim', type=int, default=512,
                        help='dimension for BackTranslator GRU hidden state')
    parser.add_argument('--BackTranslator_max_length', type=int, default=100,
                        help='maximum BackTranslator output sentence length')

    parser.add_argument('--CaptionGenerator_lstm_hidden_dim', type=int, default=512,
                        help='dimension for CaptionGenerator LSTM hidden state')

    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='hidden state size of factored LSTM')
    parser.add_argument('--factored_dim', type=int, default=512,
                        help='size of factored matrix')
    parser.add_argument('--lr_pretrain_caption', type=float, default=0.0002,
                        help='learning rate for caption model pretraining')
    parser.add_argument('--lr_caption', type=float, default=0.0002,
                        help='learning rate for caption model training')
    parser.add_argument('--lr_classifier', type=float, default=0.0002,
                        help='learning rate for style classifier training')
    # parser.add_argument('--lr_language', type=int, default=0.0005,
    #                     help='learning rate for language model training')
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--log_step', type=int, default=50,
                        help='steps for print log ')
    parser.add_argument('--save_step', type=int, default=5000,
                        help='steps for save model ')

    args = parser.parse_args()
    args.timestamp = timestamp
    args.pretrained_timestamp = timestamp

    resetRNGseed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    dir = '{}'.format(args.timestamp)
    if not logger_init:
        init_logger(dir, args.use_file_logger)
    if args.use_tensorboard:
        args.writer = init_tensorboard_writer(args.tensorboard_dir, dir + '_' + str(args.random_seed))

    logging.info(args)

    pretrainer(args)

if __name__ == '__main__':
    pretrain()
