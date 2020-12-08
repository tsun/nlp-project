import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from constant import get_symbol_id

class ImageEncoder(nn.Module):
    def __init__(self, emb_dim):
        '''
        Load the pretrained ResNet152 and replace fc
        '''
        super(ImageEncoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.final_layer = nn.Linear(2048, emb_dim)

    def forward(self, images):
        '''Extract the image feature vectors'''
        with torch.no_grad():
            features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        # todo: add a fully connect layer to convert the original 2048 dim feature to desired hidden dimension 512
        features = self.final_layer(features)
        return features

class CaptionGenerator(nn.Module):
    def __init__(self, word_embedding, style_embedding, lstm_hidden_dim=512):
        '''
        initialize the caption generator, with **batch_first** LSTM layer
        :param word_embedding: nn.Embedding with shape (vocab_size, word_embed_dim) or None
        :param style_embedding: nn.Embedding with shape (num_style, style_embed_dim) or None
        :param lstm_hidden_dim: dimension for LSTM hidden state
            note: according to eq1 & 2, I think the hidden dimension should be the same with the image feature
        '''
        super(CaptionGenerator, self).__init__()

        self.word_embedding = word_embedding
        self.style_embedding = style_embedding
        self.vocab_size = word_embedding.weight.size(0)
        self.word_embedding_dim = word_embedding.weight.size(1)
        self.style_embedding_dim = style_embedding.weight.size(1)

        # LSTM
        lstm_input_dim = self.word_embedding_dim + self.style_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_Wii = nn.Linear(lstm_input_dim, lstm_hidden_dim)
        self.lstm_Wif = nn.Linear(lstm_input_dim, lstm_hidden_dim)
        self.lstm_Wio = nn.Linear(lstm_input_dim, lstm_hidden_dim)
        self.lstm_Wic = nn.Linear(lstm_input_dim, lstm_hidden_dim)
        self.lstm_Whi = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.lstm_Whf = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.lstm_Who = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.lstm_Whc = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)

        # multimodal part
        # eq 1
        self.mm_omega_g = Variable(torch.Tensor(lstm_hidden_dim))
        self.mm_omega_g.fill_(0)
        if torch.cuda.is_available():
            self.mm_omega_g = self.mm_omega_g.cuda()
        self.mm_Wg = nn.Linear(lstm_hidden_dim * 3, lstm_hidden_dim)
        # eq 1/2, for computing c_t^l (see the following paragraph)
        self.mm_Wl = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        # eq 3
        self.output_Wo = nn.Linear(lstm_hidden_dim * 2, self.vocab_size, bias=False)

    def forward_step(self, image_code, word_embedded, h_0, c_0, style_embedded):
        '''
        note: current version is a pure language model
        :param image_code: Tensor with size (batch_size, feature_dimension) or None
        :param word_embedded: embedding of last generated word
        :param h_0: last hidden state
        :param c_0: last cell
        :param style_embedded: style embedding of given style code
        :return outputs: unnormalized logits Wo[c_t; h_t]) in eq3 (no temperature parameter, no softmax!)
        :return h_t: hidden state of LSTM
        :return c_t: cell state of LSTM (not the c_t in the paper!)
        '''

        # pure linguistic part: LSTM
        lang_embedded = torch.cat((word_embedded, style_embedded), -1)  # shape: (batch_size, word_embed_dim + style_embed_dim)
        i_t = torch.sigmoid(self.lstm_Wii(lang_embedded) + self.lstm_Whi(h_0))  # shape: (batch_size, hidden_dim)
        f_t = torch.sigmoid(self.lstm_Wif(lang_embedded) + self.lstm_Whf(h_0))  # shape: (batch_size, hidden_dim)
        o_t = torch.sigmoid(self.lstm_Wio(lang_embedded) + self.lstm_Who(h_0))  # shape: (batch_size, hidden_dim)
        c_tilda = torch.tanh(self.lstm_Wic(lang_embedded) + self.lstm_Whc(h_0))  # shape: (batch_size, hidden_dim)
        c_t = f_t * c_0 + i_t * c_tilda  # shape: (batch_size, hidden_dim)
        h_t = o_t * F.tanh(c_t)  # shape: (batch_size, hidden_dim)

        # multimodal fusion part
        l_t = torch.sigmoid(self.mm_Wl(h_t))  # only use linguistic feature for computing l_t for eq1, 2
        cl_t = l_t * torch.tanh(c_t)
        if image_code is None:
            # directly set g_t = 1
            c_t_h_t = torch.cat((cl_t, h_t), -1)  # shape: (batch_size, hidden_dim * 2)
        else:
            # add image feature when computing g_t in eq 1
            g_t = torch.tanh(self.mm_Wg(torch.cat((cl_t, h_t, image_code), dim=-1)))  # shape: (batch_size, hidden_dim)
            g_t = torch.sigmoid(torch.matmul(g_t, self.mm_omega_g))  # convert to scalar: (batch_size)
            g_t = g_t.unsqueeze(1)
            cl_t = g_t * cl_t + (1 - g_t) * image_code  # shape: (batch_size, hidden_dim) (use cl_t here to represent c_t in eq2)
            c_t_h_t = torch.cat((cl_t, h_t), -1)  # shape: (batch_size, hidden_dim * 2)

        outputs = self.output_Wo(c_t_h_t)
        return outputs, h_t, c_t


    def forward(self, image_code, style_code, max_len=30, guided_caption=None):
        '''
        generate the caption based on input image code and style code
        :param image_code: Tensor with size (batch_size, feature_dimension) or None
        :param style_code: style code with batch size (batch_size, 1)
        :param max_len: max length of the sentence (used when guided caption is not provided)
        :param guided_caption: the word IDs for known caption
            if not None, use the known caption to guide the generation of caption during training
            i.e. the next word will generated based on current word in guided caption
        :return: unnormalized logits Wo[c_t; h_t]) in eq3 (no temperature parameter, no softmax!)
        '''
        # convert the style code to style embedding
        style_embedded = self.style_embedding(style_code)  # shape: (batch_size, embed_dim)

        # initialize the hidden state
        batch_size = style_embedded.size(0)
        h_t = Variable(torch.Tensor(batch_size, self.lstm_hidden_dim))
        c_t = Variable(torch.Tensor(batch_size, self.lstm_hidden_dim))
        nn.init.uniform_(h_t)
        nn.init.uniform_(c_t)
        if torch.cuda.is_available():
            h_t = h_t.cuda()
            c_t = c_t.cuda()

        # generate captions word by word based on current word either in generated caption or guided caption
        all_outputs = []

        # todo: the first output should be unnormalized probability
        last_output = torch.zeros(style_code.shape[0], self.vocab_size)
        if torch.cuda.is_available():
            last_output = last_output.cuda()
        last_output[:, 1] = 1.0
        all_outputs.append(last_output)

        if guided_caption is None:
            # generate the start word <s>
            last_id = torch.LongTensor([1]).repeat(batch_size, 1)
            if torch.cuda.is_available():
                last_id = last_id.cuda()
            last_id = Variable(last_id, volatile=True)
            last_emb = self.word_embedding(last_id).squeeze(1)

            # generate from the second words
            for ix in range(max_len - 1):
                outputs, h_t, c_t = self.forward_step(image_code, last_emb, h_t, c_t, style_embedded)
                all_outputs.append(outputs)
        else:
            guided_caption_embedded = self.word_embedding(guided_caption)
            # directly copy the start word
            # all_outputs.append(guided_caption_embedded[:, 0, :].detach()) mismatch content <--------------------------
            # generate from the second words
            for ix in range(guided_caption_embedded.size(1) - 1):
                emb = guided_caption_embedded[:, ix, :]
                outputs, h_t, c_t = self.forward_step(image_code, emb, h_t, c_t, style_embedded)
                all_outputs.append(outputs)
        all_outputs = torch.stack(all_outputs, 1)
        return all_outputs

    def sample(self, image_code, style_code, beam_size=5, max_len=30):
        '''
        generate the captions by beam search (can only process one image and one style due to beam search?)
        :param image_code: image feature
        :param style_code: style code with batch size (batch_size, 1)
        :param beam_size:
        :param max_len:
        :return:
        '''
        # convert style code to style embedding
        batch_size = image_code.size(0)
        if style_code.shape[0] != batch_size:
            assert style_code.shape[0] == 1
            style_code = style_code.repeat(batch_size, 1)
        style_embedded = self.style_embedding(style_code)

        # initialize hidden state
        h_t = Variable(torch.Tensor(1, self.lstm_hidden_dim))
        c_t = Variable(torch.Tensor(1, self.lstm_hidden_dim))
        nn.init.uniform_(h_t)
        nn.init.uniform_(c_t)
        if torch.cuda.is_available():
            h_t = h_t.cuda()
            c_t = c_t.cuda()

        # candidates: [score, decoded_sequence, h_t, c_t]
        symbol_id = torch.LongTensor([1])
        symbol_id = Variable(symbol_id, volatile=True)
        if torch.cuda.is_available():
            symbol_id = symbol_id.cuda()
        candidates = [[0, symbol_id, h_t, c_t, [get_symbol_id('<s>')]]]

        # beam search
        t = 0
        while t < max_len - 1:
            t += 1
            tmp_candidates = []
            end_flag = True
            for score, last_id, h_t, c_t, id_seq in candidates:
                if id_seq[-1] == get_symbol_id('</s>'):
                    tmp_candidates.append([score, last_id, h_t, c_t, id_seq])
                else:
                    end_flag = False
                    last_id = last_id.view(-1)
                    word_emb = self.word_embedding(last_id)
                    output, h_t, c_t = self.forward_step(image_code, word_emb, h_t, c_t, style_embedded)
                    output = output.squeeze(0).squeeze(0)
                    # log softmax
                    output = F.log_softmax(output, dim=-1)
                    output, indices = torch.sort(output, descending=True)
                    output = output[:beam_size]
                    indices = indices[:beam_size]
                    score_list = score + output
                    for score, wid in zip(score_list, indices):
                        tmp_candidates.append(
                            [score, wid, h_t, c_t, id_seq + [int(wid.cpu().data.numpy())]]
                        )
            if end_flag:
                break
            # sort by normarized log probs and pick beam_size highest candidate
            candidates = sorted(tmp_candidates,
                                key=lambda x: -x[0].cpu().data.numpy() / len(x[-1]))[:beam_size]

        return candidates[0][-1]


class BackTranslator(nn.Module):
    def __init__(self, word_embedding, style_embedding, gru_hidden_dim=512, max_length=100, dropout_p=0.1):
        '''
        use batch_first GRU, so the input and output tensors are provided as (batch, seq, feature)
        :param word_embedding: nn.Embedding with shape (vocab_size, word_embed_dim) or None
        :param style_embedding: nn.Embedding with shape (num_style, style_embed_dim) or None
        :param gru_hidden_dim: dimension for GRU hidden state
        :param max_length: maximum output sentence length
        :param dropout_p: dropout ratio (for decoder)
        '''
        super(BackTranslator, self).__init__()

        self.word_embedding = word_embedding
        self.style_embedding = style_embedding
        self.vocab_size = word_embedding.weight.size(0)
        self.word_embedding_dim = word_embedding.weight.size(1)
        input_dim = word_embedding.weight.size(1) + style_embedding.weight.size(1)

        # encoder: single GRU
        self.gru_hidden_dim = gru_hidden_dim
        self.encoder_gru = nn.GRU(input_dim, gru_hidden_dim, batch_first=True)

        # decoder: attention based GRU
        self.max_length = max_length
        self.decoder_attn = nn.Linear(gru_hidden_dim + self.word_embedding_dim, max_length)
        self.decoder_attn_combine = nn.Linear(gru_hidden_dim + self.word_embedding_dim, gru_hidden_dim)
        self.dropout_p = dropout_p
        self.decoder_dropout = nn.Dropout(dropout_p)
        self.decoder_gru = nn.GRU(gru_hidden_dim, gru_hidden_dim, batch_first=True)

        self.output_layer = nn.Linear(gru_hidden_dim, self.vocab_size)

    def encoder_forward_step(self, input_embed, encoder_h0):
        '''
        :param input_embed: embedding for current word and style code (concatenated), shape (batch_size, 1, dim)
        :param encoder_h0: last hidden state for encoder GRU
        :return:
        '''
        # encoder part
        output, encoder_ht = self.encoder_gru(input_embed, encoder_h0)  # shape: (batch_size, 1, hidden_dim), (1, batch_size, hidden_dim)
        return output, encoder_ht

    def decoder_forward_step(self, word_id, decoder_h0, encoder_outputs):
        '''
        :param word_id: id of last generated word, shape (batch_size)
        :param decoder_h0: last hidden state for decoder GRU, shape (1, batch_size, hidden_dim)
        :param encoder_outputs: outputs of the whole sequence of encoder outputs, shape: (batch_size, max len, hidden dim)
        :return: output: shape: (batch_size, vocab_size)
        :return: hidden: shape: (1, batch_size, hidden_dim)
        '''
        word_embedded = self.word_embedding(word_id)  # shape: (batch_size, word_embed_dim)
        word_embedded = self.decoder_dropout(word_embedded)  # shape: (batch_size, 1, word_embed_dim)
        attn_inputs = torch.cat((word_embedded[:, 0, :], decoder_h0[0]), dim=-1)  # shape: (batch_size, word_embed_dim + hidden_dim)
        attn_weights = F.softmax(self.decoder_attn(attn_inputs), dim=-1)  # shape: (batch_size, max_len)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # shape: (batch_size, 1, hidden_dim)

        output = torch.cat((word_embedded, attn_applied), -1)  # shape: (batch_size, 1, word_embed_dim + hidden dim)
        output = output.squeeze(dim=1)  # shape: (batch_size, word_embed_dim + hidden dim)
        output = self.decoder_attn_combine(output).unsqueeze(dim=1)  # shape: (batch_size, 1, word_embed_dim + hidden dim)
        output = F.relu(output)
        output, hidden = self.decoder_gru(output, decoder_h0)  # shape: (batch_size, 1, hidden_dim), (1, batch_size, hidden_dim)
        output = F.log_softmax(self.output_layer(output[:, 0, :]), dim=1)  # shape: (batch_size, vocab_size)
        return output, hidden

    def encoder_init_hidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.gru_hidden_dim)
        if torch.cuda.is_available():
            h0 = h0.cuda()
        return h0

    def forward(self, text_embedded, style_code, guided_caption=None, encoder_only=False):
        '''
        :param text_embedded: word embedding of input text, shape (batch_size, seq_len, embed_dim)
        :param style_code: style id from 0 to num_style - 1
        :param guided_caption: word ids for guided caption, shape (batch_size, seq_len)
        :return: all_outputs: shape: (batch_size, seq_len, vocab_size), can be inputted into criterion = nn.NLLLoss() to compute loss
        '''
        # convert style code to style embedding
        batch_size = text_embedded.size(0)
        if style_code.shape[0] != batch_size:
            assert style_code.shape[0] == 1
            style_code = style_code.repeat(batch_size, 1)
        style_embedded = self.style_embedding(style_code)  # shape: (batch_size, dim)

        # encoder
        encoder_hidden = self.encoder_init_hidden(batch_size)
        input_length = text_embedded.size(1)
        encoder_outputs = torch.zeros(batch_size, self.max_length, self.gru_hidden_dim)
        if torch.cuda.is_available():
            encoder_outputs = encoder_outputs.cuda()
        for ei in range(input_length):
            encoder_input = torch.cat((text_embedded[:, ei, :], style_embedded), dim=-1).unsqueeze(1)  # shape: (batch_size, 1, dim)
            encoder_output, encoder_hidden = self.encoder_forward_step(encoder_input, encoder_hidden)  # shape: (batch_size, 1, hidden_dim), (1, batch_size, hidden_dim)
            encoder_outputs[:, ei, :] = encoder_output[:, 0, :]

        if not encoder_only:
            # decoder
            decoder_input = torch.LongTensor([1]).repeat(batch_size, 1)  # shape: (batch_size, 1)
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()
            decoder_hidden = encoder_hidden  # shape: (1, batch_size, hidden_dim)
            all_outputs = []
            # all_outputs.append(decoder_input)  mismatch content <--------------------------------------------------------
            last_output = torch.zeros(text_embedded.shape[0], self.vocab_size)
            if torch.cuda.is_available():
                last_output = last_output.cuda()
            last_output[:, 1] = 1.0
            all_outputs.append(last_output)
            # generate the translation one by one
            if guided_caption is None:
                # use own predictions as the next inputs
                for di in range(self.max_length - 1):
                    decoder_output, decoder_hidden = self.decoder_forward_step(
                        decoder_input, decoder_hidden, encoder_outputs)  # shape: (batch_size, vocab_size), (1, batch_size, hidden_dim)
                    topv, topi = decoder_output.topk(1, dim=1)  # shape: (batch_size, 1)
                    decoder_input = topi.detach()  # detach from history as input  # shape: (batch_size, 1)
                    all_outputs.append(decoder_output)
                    # todo: how to deal with endding with </s>? in the toturial the batch size is just 1 during training so it can break directly
            else:
                # feed the guided caption as the next input
                target_length = guided_caption.size(1)
                for di in range(target_length - 1):
                    decoder_output, decoder_hidden = self.decoder_forward_step(
                        decoder_input, decoder_hidden, encoder_outputs)  # shape: (batch_size, vocab_size), (1, batch_size, hidden_dim)
                    decoder_input = guided_caption[:, di].unsqueeze(1)  # shape: (batch_size, 1)
                    all_outputs.append(decoder_output)
            all_outputs = torch.stack(all_outputs, dim=1)  # shape: (batch_size, seq_len, vocab_size)
            return all_outputs, encoder_hidden
        else:
            return None, encoder_hidden


class Discriminator(nn.Module):
    # real/fake
    def __init__(self, in_dim, lstm_h_dim=1024, lstm_layer=2,  classifier_h_dim=1024):
        super(Discriminator, self).__init__()
        self.feature = nn.Sequential(
            nn.LSTM(in_dim, lstm_h_dim, lstm_layer, batch_first=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_h_dim, classifier_h_dim),
            nn.LeakyReLU(),
            nn.Linear(classifier_h_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, l=None, is_pack=False):
        if is_pack:
            x, _ = self.feature(x)
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            h = []
            for i in range(x.shape[0]):
                h.append(x[i,l[i]-1,:])
            h = torch.stack(h, 0)
        else:
            x = self.feature(x)
            h = x[0][:, -1, :]  # only use the last hidden state
        logit = self.classifier(h)
        return logit

class StyleClassifier(nn.Module):
    # style label
    def __init__(self, in_dim, out_dim, lstm_h_dim=1024, lstm_layer=2,  classifier_h_dim=1024):
        super(StyleClassifier, self).__init__()
        self.feature = nn.Sequential(
            nn.LSTM(in_dim, lstm_h_dim, lstm_layer, batch_first=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_h_dim, classifier_h_dim),
            nn.LeakyReLU(),
            nn.Linear(classifier_h_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x, l=None, is_pack=False):
        if is_pack:
            x, _ = self.feature(x)
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            h = []
            for i in range(x.shape[0]):
                h.append(x[i, l[i] - 1, :])
            h = torch.stack(h, 0)
        else:
            x = self.feature(x)
            h = x[0][:, -1, :]  # only use the last hidden state
        logit = self.classifier(h)
        return logit


class SentenceEncoder(nn.Module):
    def __init__(self, in_dim, lstm_h_dim=1024, lstm_layer=2):
        '''
        sentence level feature encoder, to replace the back translation module
        :param in_dim:
        :param out_dim:
        :param lstm_h_dim:
        :param lstm_layer:
        '''
        super(SentenceEncoder, self).__init__()
        self.feature = nn.Sequential(
            nn.LSTM(in_dim, lstm_h_dim, lstm_layer, batch_first=True)
        )

    def forward(self, x, l=None, is_pack=False):
        if is_pack:
            x, _ = self.feature(x)
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            h = []
            for i in range(x.shape[0]):
                h.append(x[i, l[i] - 1, :])
            h = torch.stack(h, 0)
        else:
            x = self.feature(x)
            h = x[0][:, -1, :]  # only use the last hidden state
        return h