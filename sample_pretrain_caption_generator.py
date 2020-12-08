import os
import pickle
import argparse
import torch
import tqdm
from torch.autograd import Variable
import torch.optim as optim
from build_vocab import Vocab
from flickrstyle_data_loader import get_data_loader as flickrstyle_get_data_loader
from flickrstyle_data_loader import get_data_loader_eval as flickrstyle_get_data_loader_eval
from flickrstyle_data_loader import get_styled_data_loader as flickrstyle_get_styled_data_loader
from flickrstyle_data_loader import get_styled_with_image_data_loader as flickrstyle_get_styled_with_image_data_loader
from flickrstyle_data_loader import get_styled_with_image_data_loader_eval as flickrstyle_get_styled_with_image_data_loader_eval
from coco_data_loader import get_data_loader as COCO_get_data_loader
from senticap_data_loader import get_data_loader as senticap_get_data_loader
import math

from utils import *
from model import *
from loss import *
from pretrain_caption_generator import pretrainer
import time
timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())

import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from torch.utils.data import Dataset, DataLoader

def sampler(args):
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # load vocablary
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)
    print('Vocabulary loaded from {}'.format(args.vocab_path))
    print('Vocabulary size: {}'.format(vocab_size))

    # paired data: COCO dataloader
    COCO_data_loader = COCO_get_data_loader(args.COCO_dir, args.COCO_dataType, vocab, require_img=True,
                                            batch_size=args.language_batch_size, shuffle=True)
    coco_data_iter = iter(COCO_data_loader)

    # unpaired data
    # flickrstyle dataloader
    # flickrstyle dataloader
    img_path = os.path.join(args.flickrstyle_dir, 'Images')
    factual_cap_path = os.path.join(args.flickrstyle_dir, 'factual/factual_train.txt')
    humorous_cap_path = os.path.join(args.flickrstyle_dir, 'humor/funny_train.txt')
    romantic_cap_path = os.path.join(args.flickrstyle_dir, 'romantic/romantic_train.utf.txt')
    humorous_img_path = os.path.join(args.flickrstyle_dir, 'humor/train.p')
    romantic_img_path = os.path.join(args.flickrstyle_dir, 'romantic/train.p')
    flickrstyle_factual_data_loader_train = flickrstyle_get_data_loader_eval(img_path, factual_cap_path, vocab,
                                                                        batch_size=args.unpaired_batch_size,
                                                                        shuffle=True, split='Train')
    flickrstyle_humorous_data_loader_train = flickrstyle_get_styled_with_image_data_loader_eval(img_path, humorous_cap_path, humorous_img_path, vocab,
                                                                                batch_size=args.unpaired_batch_size,
                                                                                shuffle=True, split='Train')
    flickrstyle_romantic_data_loader_train = flickrstyle_get_styled_with_image_data_loader_eval(img_path, romantic_cap_path, romantic_img_path, vocab,
                                                                                batch_size=args.unpaired_batch_size,
                                                                                shuffle=True, split='Train')

    flickrstyle_factual_data_loader_test = flickrstyle_get_data_loader_eval(img_path, factual_cap_path, vocab,
                                                                       batch_size=args.unpaired_batch_size,
                                                                       shuffle=True, split='Test')
    flickrstyle_humorous_data_loader_test = flickrstyle_get_styled_with_image_data_loader_eval(img_path, humorous_cap_path,
                                                                                          humorous_img_path, vocab,
                                                                                          batch_size=args.unpaired_batch_size,
                                                                                          shuffle=True, split='Test')
    flickrstyle_romantic_data_loader_test = flickrstyle_get_styled_with_image_data_loader_eval(img_path, romantic_cap_path,
                                                                                          romantic_img_path, vocab,
                                                                                          batch_size=args.unpaired_batch_size,
                                                                                          shuffle=True, split='Test')

    style_data_loader_train = [flickrstyle_factual_data_loader_train, flickrstyle_humorous_data_loader_train,
                               flickrstyle_romantic_data_loader_train]
    style_data_iter_train = [iter(flickrstyle_factual_data_loader_train), iter(flickrstyle_humorous_data_loader_train),
                             iter(flickrstyle_romantic_data_loader_train)]

    style_data_loader_test = [flickrstyle_factual_data_loader_test, flickrstyle_humorous_data_loader_test,
                              flickrstyle_romantic_data_loader_test]
    style_data_iter_test = [iter(flickrstyle_factual_data_loader_test), iter(flickrstyle_humorous_data_loader_test),
                            iter(flickrstyle_romantic_data_loader_test)]

    # # senticap dataloader
    # senticap_data_loader = senticap_get_data_loader(args.senticap_dir, args.senticap_img_dir, vocab, require_img=False,
    #                                              batch_size=args.language_batch_size, shuffle=True)

    # define global embeddings and models
    word_embedding = nn.Embedding(len(vocab), args.emb_dim)
    style_embedding = nn.Embedding(args.style_num, args.emb_dim)
    imageEncoder = ImageEncoder(args.emb_dim)
    captionGenerator = CaptionGenerator(word_embedding, style_embedding, args.CaptionGenerator_lstm_hidden_dim)

    if torch.cuda.is_available():
        word_embedding = word_embedding.cuda()
        style_embedding = style_embedding.cuda()
        imageEncoder = imageEncoder.cuda()
        captionGenerator = captionGenerator.cuda()

    # load pretrained models if possible
    if args.is_pretrained_model:
        print('loading pretrained image encoder, caption generator model and embeddings')
        word_embedding_file_name = os.path.join(args.model_path, args.pretrained_timestamp,'pretrained_word_embedding{}.pth'
                                                .format('' if args.iter_to_load is None else '_{}'.format(args.iter_to_load)))
        word_embedding.load_state_dict(torch.load(word_embedding_file_name))

        style_embedding_file_name = os.path.join(args.model_path, args.pretrained_timestamp, 'pretrained_style_embedding{}.pth'
                                                 .format('' if args.iter_to_load is None else '_{}'.format(args.iter_to_load)))
        style_embedding.load_state_dict(torch.load(style_embedding_file_name))

        checkpoint_file_name = os.path.join(args.model_path, args.pretrained_timestamp, 'pretrained_image_encoder{}.pth'
                                            .format('' if args.iter_to_load is None else '_{}'.format(args.iter_to_load)))
        imageEncoder.load_state_dict(torch.load(checkpoint_file_name))

        checkpoint_file_name = os.path.join(args.model_path, args.pretrained_timestamp, 'pretrained_caption_generator{}.pth'
                                            .format('' if args.iter_to_load is None else '_{}'.format(args.iter_to_load)))
        captionGenerator.load_state_dict(torch.load(checkpoint_file_name))
    else:
        print('loading trained image encoder, caption generator model and embeddings')
        word_embedding_file_name = os.path.join(args.model_path, args.trained_timestamp, 'word_embedding{}.pth'
                                                .format('' if args.iter_to_load is None else '_{}'.format(args.iter_to_load)))
        word_embedding.load_state_dict(torch.load(word_embedding_file_name))

        style_embedding_file_name = os.path.join(args.model_path, args.trained_timestamp, 'style_embedding{}.pth'
                                                 .format('' if args.iter_to_load is None else '_{}'.format(args.iter_to_load)))
        style_embedding.load_state_dict(torch.load(style_embedding_file_name))

        checkpoint_file_name = os.path.join(args.model_path, args.trained_timestamp, 'image_encoder{}.pth'
                                            .format('' if args.iter_to_load is None else '_{}'.format(args.iter_to_load)))
        imageEncoder.load_state_dict(torch.load(checkpoint_file_name))

        checkpoint_file_name = os.path.join(args.model_path, args.trained_timestamp, 'caption_generator{}.pth'
                                            .format('' if args.iter_to_load is None else '_{}'.format(args.iter_to_load)))
        captionGenerator.load_state_dict(torch.load(checkpoint_file_name))

    # sample on training data
    # sample_data_loader = COCO_data_loader
    # sample_data_loader = flickrstyle_humorous_data_loader_test

    # sample on test data
    # for ds, sample_data_loader in enumerate(style_data_loader_test):
    for ds, sample_data_loader in enumerate([flickrstyle_humorous_data_loader_test]): # just fake iteration
        logging.info('+++++++++++++++++++++++++++++++++++++')
        logging.info('+++++++++++++++++++++++++++++++++++++')
        for i in tqdm.tqdm(range(args.maximum_iter), total=args.maximum_iter):
             # ==== iterate each style and train the networks

            data = next(iter(sample_data_loader))

            img = data[0].float()
            cap = data[1].long()
            cap_len = data[2].long()
            img_file = data[3]
            if torch.cuda.is_available():
                img = img.cuda()
                cap = cap.cuda()
                cap_len = cap_len.cuda()

            for j in range(img.shape[0]):
                logging.info('image file: {}\n'.format(img_file[j]))
                if args.is_plot_image:
                    plt.axis('off')
                    I = img[j].cpu().permute(1, 2, 0)
                    plt.imshow(I)
                    plt.show()
                for style_id in range(args.style_num):
                    # sample one image to see the result
                    with torch.no_grad():
                        imageEncoder.eval()
                        captionGenerator.eval()
                        sample_img_emb = imageEncoder(img[j].unsqueeze(0))
                        # sample_style_code = torch.zeros(1).long()  # 0 for fact
                        sample_style_code = torch.full((1,), style_id, dtype=torch.int).long()  # test each style
                        if torch.cuda.is_available():
                            sample_style_code = sample_style_code.cuda()
                        if torch.cuda.is_available():
                            sample_style_code = sample_style_code.cuda()
                        sample_cap = captionGenerator.sample(sample_img_emb, sample_style_code)
                    logging.info('sample results for style {}: \n{}\n'.format(style_id, get_sentence(sample_cap, vocab)))

                # get ground truth captions from all three datasets
                # logging.info('ground truth caption {} {}: \n{}\n'.format('gt', ds, get_sentence(cap[j].tolist(), vocab)))

                cap_factual = flickrstyle_factual_data_loader_test.dataset.dataset.get_caption(img_file[j])
                if cap_factual is not None:
                    for capi in cap_factual:
                          logging.info(
                    'ground truth caption {}  for style {}: \n{}'.format('gt', 0, capi))

                cap_humor = flickrstyle_humorous_data_loader_test.dataset.dataset.get_caption(img_file[j])
                if cap_humor is not None:
                    logging.info(
                        'ground truth caption {}  for style {}: \n{}'.format('gt', 1, cap_humor))

                cap_romantic = flickrstyle_romantic_data_loader_test.dataset.dataset.get_caption(img_file[j])
                if cap_romantic is not None:
                    logging.info(
                        'ground truth caption {}  for style {}: \n{}'.format('gt', 2, cap_romantic))
                logging.info('--------------------------------------')

def sample():
    parser = argparse.ArgumentParser(
            description='MSCap: Multi-Style Image Captioning with Unpaired Stylized Text')

    # environment
    parser.add_argument('--gpu_id', default='1', type=str,
                        help='which gpu to use')
    parser.add_argument('--random_seed', default='3', type=int,
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

    parser.add_argument('--load_pretrained', action='store_true',
                        help='use pretrained embedding and caption generator')
    parser.add_argument('--model_path', type=str, default='checkpoint',
                        help='path for saving trained models')
    parser.add_argument('--pretrained_timestamp', type=str, default='2020-11-28_13.44.06',
                        help='timestamp for trained models')
    parser.add_argument('--trained_timestamp', type=str, default='2020-12-05_15.27.00',
                        help='timestamp for trained models')
    parser.add_argument('--is_pretrained_model', default='False', type=str2bool,
                        help='whether load pretrained model')
    parser.add_argument('--iter_to_load', default=5000, type=int,
                        help='iteration of pretrained/trained model to load')

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

    parser.add_argument('--n_critic', type=int, default=1, help='the rate to train the generator')

    parser.add_argument('--continue_training', action='store_true', help='load the pretrained result and update')
    parser.add_argument('--maximum_pretrain_iter', type=int, default=40000)
    parser.add_argument('--maximum_iter', type=int, default=10)

    parser.add_argument('--paired_batch_size', type=int, default=64,
                        help='mini batch size for paired model training')
    parser.add_argument('--unpaired_batch_size', type=int, default=64,
                        help='mini batch size for unpaired model training')
    parser.add_argument('--unpaired_loss_weight', type=float, default=0.1,
                        help='weight of unpaired data loss')
    parser.add_argument('--language_batch_size', type=int, default=64)
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
    parser.add_argument('--lr_discriminator', type=float, default=0.0002,
                        help='learning rate for discriminator training')
    parser.add_argument('--lr_classifier', type=float, default=0.0002,
                        help='learning rate for style classifier training')
    parser.add_argument('--lr_translator', type=float, default=0.0002,
                        help='learning rate for back translator training')
    # parser.add_argument('--lr_language', type=float, default=0.0005,
    #                     help='learning rate for language model training')
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--log_step_caption', type=int, default=50,
                        help='steps for print log while train caption model')
    parser.add_argument('--log_step', type=int, default=50,
                        help='steps for print log ')
    parser.add_argument('--save_step', type=int, default=5000,
                        help='steps for save model ')

    parser.add_argument('--is_plot_image', default='False', type=str2bool,
                        help='whether plot image')

    args = parser.parse_args()
    args.timestamp = timestamp

    resetRNGseed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    dir = '{}'.format(args.timestamp)
    if not logger_init:
        init_logger(dir, args.use_file_logger, suffix='_eval_'+args.trained_timestamp)

    logging.info(args)
    sampler(args)



if __name__ == '__main__':
    sample()