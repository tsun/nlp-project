'''
idea 1: instead of using the back translator to grounded the generated caption,
only compare the sentence level representation between generated caption and the ground truth factual caption
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
from flickrstyle_data_loader import get_styled_with_image_data_loader as flickrstyle_get_styled_with_image_data_loader

from coco_data_loader import get_data_loader as COCO_get_data_loader
from senticap_data_loader import get_data_loader as senticap_get_data_loader
import math
from eval import *
from utils import *
from model import *
from loss import *
from pretrain_caption_generator import pretrainer
import time
timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())

def trainer(args):
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
                                            batch_size=args.paired_batch_size, shuffle=True)
    coco_data_iter = iter(COCO_data_loader)

    # unpaired data
    # flickrstyle dataloader
    img_path = os.path.join(args.flickrstyle_dir, 'Images')
    factual_cap_path = os.path.join(args.flickrstyle_dir, 'factual/factual_train.txt')
    humorous_cap_path = os.path.join(args.flickrstyle_dir, 'humor/funny_train.txt')
    romantic_cap_path = os.path.join(args.flickrstyle_dir, 'romantic/romantic_train.utf.txt')
    humorous_img_path = os.path.join(args.flickrstyle_dir, 'humor/train.p')
    romantic_img_path = os.path.join(args.flickrstyle_dir, 'romantic/train.p')
    flickrstyle_factual_data_loader_train = flickrstyle_get_data_loader(img_path, factual_cap_path, vocab, require_img=False,
                                                                        batch_size=args.unpaired_batch_size,
                                                                        shuffle=True, split='Train')
    flickrstyle_humorous_data_loader_train = flickrstyle_get_styled_data_loader(humorous_cap_path, vocab,
                                                                                batch_size=args.unpaired_batch_size,
                                                                                shuffle=True, split='Train')
    flickrstyle_romantic_data_loader_train = flickrstyle_get_styled_data_loader(romantic_cap_path, vocab,
                                                                                batch_size=args.unpaired_batch_size,
                                                                                shuffle=True, split='Train')

    flickrstyle_factual_data_loader_test = flickrstyle_get_data_loader(img_path, factual_cap_path, vocab, require_img=True,
                                                                       batch_size=args.unpaired_batch_size,
                                                                       shuffle=True, split='Test')
    flickrstyle_humorous_data_loader_test = flickrstyle_get_styled_with_image_data_loader(img_path, humorous_cap_path, humorous_img_path, vocab,
                                                                               batch_size=args.unpaired_batch_size,
                                                                               shuffle=True, split='Test')
    flickrstyle_romantic_data_loader_test = flickrstyle_get_styled_with_image_data_loader(img_path, romantic_cap_path, romantic_img_path, vocab,
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
    #                                              batch_size=args.unpaired_batch_size, shuffle=True)

    # define global embeddings and models
    word_embedding = nn.Embedding(len(vocab), args.emb_dim)
    style_embedding = nn.Embedding(args.style_num, args.emb_dim)
    imageEncoder = ImageEncoder(args.emb_dim)
    captionGenerator = CaptionGenerator(word_embedding, style_embedding, args.CaptionGenerator_lstm_hidden_dim)
    sentenceEncoder = SentenceEncoder(in_dim=args.emb_dim, lstm_layer=args.sentence_encoder_layer_num)
    discriminator = Discriminator(in_dim=args.emb_dim)
    styleClassifier = StyleClassifier(in_dim=args.emb_dim, out_dim=args.style_num)

    if torch.cuda.is_available():
        word_embedding = word_embedding.cuda()
        style_embedding = style_embedding.cuda()
        imageEncoder = imageEncoder.cuda()
        captionGenerator = captionGenerator.cuda()
        discriminator = discriminator.cuda()
        styleClassifier = styleClassifier.cuda()
        sentenceEncoder = sentenceEncoder.cuda()

    # load pretrained models if possible
    if args.load_pretrained:
        print('loading pretrained image encoder, caption generator model and embeddings')
        word_embedding_file_name = os.path.join(args.model_path, args.pretrained_timestamp,'pretrained_word_embedding.pth')
        word_embedding.load_state_dict(torch.load(word_embedding_file_name))

        style_embedding_file_name = os.path.join(args.model_path, args.pretrained_timestamp, 'pretrained_style_embedding.pth')
        style_embedding.load_state_dict(torch.load(style_embedding_file_name))

        checkpoint_file_name = os.path.join(args.model_path, args.pretrained_timestamp, 'pretrained_image_encoder.pth')
        imageEncoder.load_state_dict(torch.load(checkpoint_file_name))

        checkpoint_file_name = os.path.join(args.model_path, args.pretrained_timestamp, 'pretrained_caption_generator.pth')
        captionGenerator.load_state_dict(torch.load(checkpoint_file_name))
    else:
        print('start pretraining for caption generator and embeddings')
        pretrainer(args)

    # optimizer for training
    optimizer_cap = optim.Adam(list(imageEncoder.parameters())
                               + list(captionGenerator.parameters()), lr=args.lr_caption)
    optimizer_dis = optim.Adam(discriminator.parameters(), lr=args.lr_discriminator)
    optimizer_cls = optim.Adam(styleClassifier.parameters(), lr=args.lr_classifier)
    optimizer_tsl = optim.Adam(sentenceEncoder.parameters(), lr=args.lr_translator)
    criterion_cls = nn.CrossEntropyLoss()  # for style classification
    criterion_adv = nn.BCELoss()  # for adversarial loss
    criterion_mse = nn.MSELoss()
    real_label = 1.
    fake_label = 0.

    # train all networks: loss and optimizer
    # temperature = 1
    for i in tqdm.tqdm(range(args.maximum_iter), total=args.maximum_iter):
        # misc
        # temperature = 1 # temperature in section 3.8, should decrease gradually towards 0
        temperature = 1 - (2 / (1 + math.exp(-10 * i / args.maximum_iter)) - 1)
        use_guided_translation = True  # use ground truth translation to speed up the training process

        # ==== data preprocessing ====
        # sample from paired data
        try:
            coco_data = next(coco_data_iter)
        except:
            coco_data_iter = iter(COCO_data_loader)
            coco_data = next(coco_data_iter)
        coco_img = coco_data[0].float()
        coco_cap = coco_data[1].long()
        coco_cap_len = coco_data[2].long()
        if torch.cuda.is_available():
            coco_img = coco_img.cuda()
            coco_cap = coco_cap.cuda()
        # sample from unpaired data
        style_data_all = []
        for style_id in range(args.style_num):
            try:
                style_data = next(style_data_iter_train[style_id])
            except:
                style_data_iter_train[style_id] = iter(style_data_loader_train[style_id])
                style_data = next(style_data_iter_train[style_id])
            style_code = torch.full((style_data[0].shape[0],), style_id, dtype=torch.int).long()
            if torch.cuda.is_available():
                style_data_all.append((style_data[0].long().cuda(),
                                       style_data[1].long().cuda(), style_code.cuda()))  # caption, length, style code
            else:
                style_data_all.append((style_data[0].long(), style_data[1].long(), style_code))  # caption, length, style code

        # ==== train the discriminator and classifier ====
        # ==== iterate each style and train the networks

        discriminator_loss_total = 0.
        classifier_loss_total = 0.

        for style_id in range(args.style_num):
            imageEncoder.zero_grad()
            captionGenerator.zero_grad()
            discriminator.zero_grad()
            styleClassifier.zero_grad()

            emb_style = word_embedding(style_data_all[style_id][0])
            length = style_data_all[style_id][1]

            style_batch_size = emb_style.size(0)  # batch size for style data, but not equal to image when read the last iter in data loader
            style_code = style_data_all[style_id][2]

            # compute the adversarial loss (for discriminator) and classification loss for unpaired caption
            label_batch = torch.full((style_batch_size, ), real_label, dtype=torch.float)

            if torch.cuda.is_available():
                label_batch = label_batch.cuda()

            # handling variable lengths
            if args.pack_lstm_input:
                emb_style_pack = torch.nn.utils.rnn.pack_padded_sequence(emb_style, length, batch_first=True)
                pred_dis = discriminator(emb_style_pack, length, is_pack=True)
                pred_cls = styleClassifier(emb_style_pack, length, is_pack=True)
            else:
                pred_dis = discriminator(emb_style)
                pred_cls = styleClassifier(emb_style)

            dis_loss_real = criterion_adv(pred_dis, label_batch)
            dis_loss_real.backward(retain_graph=True)
            cls_loss_real = criterion_cls(pred_cls, style_code)
            cls_loss_real.backward(retain_graph=True)

            discriminator_loss_total += dis_loss_real.item()
            classifier_loss_total += cls_loss_real.item()

            # compute the adversarial loss (for discriminator) with fake caption
            img_emb = imageEncoder(coco_img)
            style_code = torch.full((img_emb.shape[0],), style_id, dtype=torch.int).long()
            if torch.cuda.is_available():
                style_code = style_code.cuda()
            logits_fake = captionGenerator(img_emb, style_code, max_len=30)
            cap_emb_fake = compute_soft_embedding(logits_fake, temperature, word_embedding)
            label_batch = torch.full((img_emb.shape[0],), fake_label, dtype=torch.float)
            if torch.cuda.is_available():
                label_batch = label_batch.cuda()

            # handling variable lengths
            if args.pack_lstm_input:
                length_fake = torch.full((cap_emb_fake.shape[0],), cap_emb_fake.shape[1], dtype=torch.int).long()
                if torch.cuda.is_available():
                    length_fake = length_fake.cuda()

                cap_emb_fake_pack = torch.nn.utils.rnn.pack_padded_sequence(cap_emb_fake.detach(), length_fake,
                                                                            batch_first=True)

                pred_dis = discriminator(cap_emb_fake_pack, length_fake, is_pack=True)
                pred_cls = styleClassifier(cap_emb_fake_pack, length_fake, is_pack=True)
            else:
                pred_dis = discriminator(cap_emb_fake.detach())
                pred_cls = styleClassifier(cap_emb_fake.detach())

            dis_loss_fake = criterion_adv(pred_dis, label_batch)
            dis_loss_fake.backward(retain_graph=True)

            cls_loss_fake = criterion_cls(pred_cls, style_code)
            cls_loss_fake.backward(retain_graph=True)

            discriminator_loss_total += dis_loss_fake
            classifier_loss_total += cls_loss_fake

            # update the discriminator and style classifier
            optimizer_dis.step()
            optimizer_cls.step()

        # ==== train the generator ====
        # language model loss
        if (i + 1) % args.n_critic == 0:
            imageEncoder.zero_grad()
            captionGenerator.zero_grad()
            style_code_factual = torch.zeros(coco_cap.shape[0]).long()
            if torch.cuda.is_available():
                style_code_factual = style_code_factual.cuda()
            pred_cap = captionGenerator(img_emb, style_code_factual, guided_caption=coco_cap)
            pred_cap_pair_loss = masked_cross_entropy(pred_cap[:, 1:, :].contiguous(),
                                                       coco_cap[:, 1:].contiguous(),
                                                       coco_cap_len - 1)
            pred_cap_pair_loss.backward()
            optimizer_cap.step()


            pred_cap_style_loss_all = []
            for style_id in range(args.style_num):
                imageEncoder.zero_grad()
                captionGenerator.zero_grad()

                cap = style_data_all[style_id][0]
                cap_len = style_data_all[style_id][1]
                style_code = style_data_all[style_id][2]
                pred_cap_style = captionGenerator(None, style_code, guided_caption=cap)
                pred_cap_style_loss = masked_cross_entropy(pred_cap_style[:, 1:, :].contiguous(), cap[:, 1:].contiguous(),
                                                           cap_len - 1) * args.unpaired_loss_weight
                pred_cap_style_loss.backward()
                pred_cap_style_loss_all.append(pred_cap_style_loss.item())
                optimizer_cap.step()


        # ==== iterate each style and train the networks
        # ==== train the generator ====
        if (i + 1) % args.n_critic == 0:
            adv_loss_total = 0.
            cls_loss_total = 0.
            tsl_loss_total = 0.

            # generator loss
            for style_id in range(args.style_num):
                imageEncoder.zero_grad()
                captionGenerator.zero_grad()
                discriminator.zero_grad()
                styleClassifier.zero_grad()
                sentenceEncoder.zero_grad()

                # compute the adversarial loss (for discriminator) with fake caption
                img_emb = imageEncoder(coco_img)
                style_code = torch.full((img_emb.shape[0],), style_id, dtype=torch.int).long()
                if torch.cuda.is_available():
                    style_code = style_code.cuda()

                logits_fake = captionGenerator(img_emb, style_code, max_len=30)
                cap_emb_fake = compute_soft_embedding(logits_fake, temperature, word_embedding)
                label_batch = torch.full((img_emb.shape[0],), real_label, dtype=torch.float) # fake labels are real for generator
                if torch.cuda.is_available():
                    label_batch = label_batch.cuda()

                # compute the adversarial loss, classification loss for caption generator
                # handling variable lengths
                if args.pack_lstm_input:
                    length_fake = torch.full((cap_emb_fake.shape[0],), cap_emb_fake.shape[1],
                                             dtype=torch.int).long()
                    if torch.cuda.is_available():
                        length_fake = length_fake.cuda()

                    cap_emb_fake_pack = torch.nn.utils.rnn.pack_padded_sequence(cap_emb_fake.detach(), length_fake,
                                                                                batch_first=True)

                    pred_dis = discriminator(cap_emb_fake_pack, length_fake, is_pack=True)
                    pred_cls = styleClassifier(cap_emb_fake_pack, length_fake, is_pack=True)

                    # also compare the sentence level representation instead of using back translator
                    sen_code_fake = sentenceEncoder(cap_emb_fake_pack, length_fake, is_pack=True)
                    cap_emb_coco = word_embedding(coco_cap)
                    cap_emb_coco_pack = torch.nn.utils.rnn.pack_padded_sequence(cap_emb_coco, coco_cap_len, batch_first=True)
                    sen_code_coco = sentenceEncoder(cap_emb_coco_pack, coco_cap_len, is_pack=True)
                else:
                    pred_dis = discriminator(cap_emb_fake)
                    pred_cls = styleClassifier(cap_emb_fake)
                    sen_code_fake = sentenceEncoder(cap_emb_fake)
                    cap_emb_coco = word_embedding(coco_cap)
                    sen_code_coco = sentenceEncoder(cap_emb_coco)

                adv_loss = criterion_adv(pred_dis, label_batch)
                # label_batch.fill_(fake_label)
                # adv_loss = -criterion_adv(pred_dis, label_batch)

                adv_loss *= args.adv_loss_weight
                adv_loss.backward(retain_graph=True)
                adv_loss_total += adv_loss.item()

                # compute the style classification loss
                cls_loss = criterion_cls(pred_cls, style_code)
                cls_loss.backward(retain_graph=True)
                cls_loss_total += cls_loss.item()

                # compute the sentence representation difference
                tsl_loss = criterion_mse(sen_code_fake, sen_code_coco)
                tsl_loss *= args.tsl_loss_weight
                tsl_loss.backward(retain_graph=True)
                tsl_loss_total += tsl_loss.item()

            # update the caption generator and back translator
            optimizer_cap.step()
            optimizer_tsl.step()

        # log and save
        info = '\n[train] Iter {}: [discriminator] adv loss: {:.6f} cls loss: {:.6f} '.format(
            i + 1,  discriminator_loss_total, classifier_loss_total)
        if (i + 1) % args.n_critic == 0:
            info += '[generator] adv loss: {:.6f} cls loss: {:.6f} tsl loss: {:.6f} \n'.format(
                adv_loss_total, cls_loss_total, tsl_loss_total
            )
            info += '[train] Iter {}: paired loss: {:.6f} style loss: {:.6f} {:.6f} {:.6f}\n'.format(
                i + 1, pred_cap_pair_loss.item(), *pred_cap_style_loss_all)
        else:
            info += '\n'

        # eval model
        if i % args.eval_step == 0 or i == args.maximum_iter - 1:
            for ss in range(len(style_data_loader_test)):
                # eval_res = eval(style_data_loader_test[ss], ss, imageEncoder, captionGenerator, discriminator,
                #                 styleClassifier, backTranslator, word_embedding, style_embedding, vocab, args)
                eval_res = evaluator(style_data_loader_test[ss], ss, imageEncoder, captionGenerator, styleClassifier, word_embedding, vocab, args)
                info += '[eval] Iter {}: style: {} style_cls_acc: {:.6f} bleu1: {:.6f} bleu3: {:.6f} \n'.format(i, ss, *eval_res)

        if i % args.log_step == 0 or i == args.maximum_iter - 1:
            # this will print on both terminal and file
            logging.info(info)
        else:
            print(info)

        for style_id in range(args.style_num):
            # sample one image to see the result
            with torch.no_grad():
                imageEncoder.eval()
                captionGenerator.eval()
                sample_img_emb = imageEncoder(coco_img[0].unsqueeze(0))
                # sample_style_code = torch.zeros(1).long()  # 0 for fact
                sample_style_code = torch.full((1,), style_id, dtype=torch.int).long()  # test each style
                if torch.cuda.is_available():
                    sample_style_code = sample_style_code.cuda()
                if torch.cuda.is_available():
                    sample_style_code = sample_style_code.cuda()
                sample_cap = captionGenerator.sample(sample_img_emb, sample_style_code)
                if i % args.log_step == 0 or i == args.maximum_iter - 1:
                    logging.info('sample results for style {}: \n{}\n'.format(style_id, get_sentence(sample_cap, vocab)))
                else:
                    print('sample results for style {}: \n{}\n'.format(style_id, get_sentence(sample_cap, vocab)))

        imageEncoder.train()
        captionGenerator.train()

        if i % args.save_step == 0 and args.save_model:
            # save the word embedding, style embedding and image caption generator
            save_model(word_embedding, style_embedding, captionGenerator, imageEncoder, args, iter=i)

            # for quick debug, save current models are last models
            save_model(word_embedding, style_embedding, captionGenerator, imageEncoder, args)


    # save the word embedding, style embedding and image caption generator
    # for quick debug, save current models are last models
    if args.save_model:
        save_model(word_embedding, style_embedding, captionGenerator, imageEncoder, args)


def train():
    parser = argparse.ArgumentParser(
            description='MSCap: Multi-Style Image Captioning with Unpaired Stylized Text (with idea1)')

    # environment
    parser.add_argument('--gpu_id', default='2', type=str,
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

    parser.add_argument('--load_pretrained', action='store_true', default='True',
                        help='use pretrained embedding and caption generator')
    parser.add_argument('--model_path', type=str, default='checkpoint',
                        help='path for saving trained models')
    parser.add_argument('--pretrained_timestamp', type=str, default='2020-11-28_13.44.06',
                        help='timestamp for trained models')
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

    parser.add_argument('--n_critic', type=int, default=5, help='the rate to train the generator')

    parser.add_argument('--continue_training', action='store_true', help='load the pretrained result and update')
    parser.add_argument('--maximum_pretrain_iter', type=int, default=40000)
    parser.add_argument('--maximum_iter', type=int, default=10000)

    parser.add_argument('--paired_batch_size', type=int, default=48,
                        help='mini batch size for paired model training')
    parser.add_argument('--unpaired_batch_size', type=int, default=48,
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
    parser.add_argument('--sentence_encoder_layer_num', type=int, default=2,
                        help='the number of lstm layers for sentence feature encoder')

    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='hidden state size of factored LSTM')
    parser.add_argument('--factored_dim', type=int, default=512,
                        help='size of factored matrix')
    parser.add_argument('--lr_pretrain_caption', type=float, default=0.0002,
                        help='learning rate for caption model pretraining')
    parser.add_argument('--lr_caption', type=float, default=0.00005,
                        help='learning rate for caption model training')
    parser.add_argument('--lr_discriminator', type=float, default=0.00005,
                        help='learning rate for discriminator training')
    parser.add_argument('--lr_classifier', type=float, default=0.00005,
                        help='learning rate for style classifier training')
    parser.add_argument('--lr_translator', type=float, default=0.0005,
                        help='learning rate for back translator training')
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--log_step_caption', type=int, default=50,
                        help='steps for print log while train caption model')
    parser.add_argument('--log_step', type=int, default=50,
                        help='steps for print log ')
    parser.add_argument('--eval_step', type=int, default=1000,
                        help='steps for print log ')
    parser.add_argument('--save_step', type=int, default=5000,
                        help='steps for save model ')
    parser.add_argument('--save_model', default='True', type=str2bool,
                        help='whether save model')

    parser.add_argument('--pack_lstm_input', default='True', type=str2bool,
                        help='whether to pack lstm input')

    parser.add_argument('--adv_loss_weight', type=float, default=0.1,
                        help='adversarial loss weight')
    parser.add_argument('--tsl_loss_weight', type=float, default=5,
                        help='back translator loss weight')

    parser.add_argument('--max_eval_number', type=int, default=1000,
                        help='maximum data for evaluation ')

    args = parser.parse_args()
    args.timestamp = timestamp
    if not args.load_pretrained:
        args.pretrained_timestamp = timestamp

    resetRNGseed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    dir = '{}'.format(args.timestamp)
    if not logger_init:
        init_logger(dir, args.use_file_logger)
    if args.use_tensorboard:
        args.writer = init_tensorboard_writer(args.tensorboard_dir, dir + '_' + str(args.random_seed))

    logging.info(args)
    trainer(args)



if __name__ == '__main__':
    train()