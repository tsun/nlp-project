import torch
from nltk.translate.bleu_score import sentence_bleu
from utils import get_sentence_list

def evaluator(dataloader, style, imageEncoder, captionGenerator, styleClassifier, word_embedding, vocab, args):
    cum_acc = 0.
    cum_num = 0.
    cum_bleu1 = 0.
    cum_bleu3 = 0.

    for data in iter(dataloader):
        if cum_num > args.max_eval_number:
            break

        img = data[0].float()
        cap = data[1].long()
        cap_len = data[2].long()
        if torch.cuda.is_available():
            img = img.cuda()
            cap = cap.cuda()
            cap_len = cap_len.cuda()
        emb = word_embedding(cap)

        if args.pack_lstm_input:
            emb_pack = torch.nn.utils.rnn.pack_padded_sequence(emb, cap_len, batch_first=True)
            pred_cls = styleClassifier(emb_pack, cap_len, is_pack=True)
        else:
            pred_cls = styleClassifier(emb)

        cum_acc += (pred_cls.max(-1)[1]==style).sum()


        img_emb = imageEncoder(img)
        style_code = torch.full((1,), style, dtype=torch.int).long()
        if torch.cuda.is_available():
            style_code = style_code.cuda()
        for i in range(img_emb.shape[0]):
            reference_cap = [get_sentence_list(cap[i].tolist(), vocab)]  # outer list is required
            sample_cap = get_sentence_list(captionGenerator.sample(img_emb[i].unsqueeze(0), style_code), vocab)
            bleu1 = sentence_bleu(reference_cap, sample_cap, weights=(1, 0, 0, 0))
            bleu3 = sentence_bleu(reference_cap, sample_cap, weights=(0.33, 0.33, 0.33, 0))
            cum_bleu1 += bleu1
            cum_bleu3 += bleu3

        cum_num += cap.shape[0]

    acc = cum_acc/cum_num
    bleu1 = cum_bleu1/cum_num
    bleu3 = cum_bleu3 / cum_num
    return [acc, bleu1, bleu3]

def eval(dataloader, style, imageEncoder, captionGenerator, discriminator,
         styleClassifier, backTranslator, word_embedding, style_embedding, vocab, args):
    res = evaluator(dataloader, style, imageEncoder, captionGenerator, styleClassifier, word_embedding, vocab, args)

    return res
