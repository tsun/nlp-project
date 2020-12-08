# GAN-based multi-style image captioning with unpaired text

## The list of the original source for our code base

There is no official (and unofficial) implementation of our baseline method ([MSCap](https://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_MSCap_Multi-Style_Image_Captioning_With_Unpaired_Stylized_Text_CVPR_2019_paper.pdf)) online, so we implement it from scratch. We refers to the following sources during the implementation:

- data_loader api for MS COCO, FlickrStyle and the logic of the caption generator from StyleNet https://github.com/kacky24/stylenet
- PyTorch translation tutorial https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

## The list of files that we modified and the specific functions within each file we modified for our project

For https://github.com/kacky24/stylenet, we modify the following files
- models.py: we only use the `FactoredLSTM` class to implement the logic of the caption generator. We implement other models.
- train.py: we follow the general organization of training scripts, and add specific losses and optimization for our models. We add utility functions like logger and model saving and loading.
- loss.py: We use the masked_cross_entropy loss and implement other losses.

For https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html, we combine the example codes of the `EncoderRNN` and `AttnDecoderRNN` to be a single back translator module in our own `model.py`, which can efficiently translate stylized captions to the factual captions by a single `forward` function. 

## The list of commands that provide how we train and test our baseline and the systems we built

First of all, we need to build the vocabulary, use the following command
```
python build_vocab.py
```
The vocubulary will be saved to `data/vocab.pkl`.

Before training, we need to pretrain the caption generator for 40000 iterations, using the following command 
```
python pretrain_caption_generator.py --gpu_id 0 --maximum_pretrain_iter 40000
```
and change the `--gpu_id` if necessary. After it finishes, the pretrained caption generator model as well as the word embedding and style embedding will be saved into `checkpoint/xxxx-xx-xx_xx.xx.xx` folder, where `xxxx-xx-xx_xx.xx.xx` is the timestamp of the pretraining. We use the same pretrained generator model and embeddings for the following experiments.

To train the baseline models, run
```
python train.py --gpu_id 0 --load_pretrained --pretrained_timestamp xxxx-xx-xx_xx.xx.xx --maximum_iter 5000 --maximum_pretrain_iter 0
```
and change the `--gpu_id` and `--pretrained_timestamp` if necessary.

To train the baseline model with improved content consistency loss, run 
```
python train_idea1.py --gpu_id 0 --load_pretrained --pretrained_timestamp xxxx-xx-xx_xx.xx.xx --maximum_iter 5000 --maximum_pretrain_iter 0 --sentence_encoder_layer_num 4 --tsl_loss_weight 500
```
and change the `--gpu_id` and `--pretrained_timestamp` if necessary.

To train the baseline model with style consistency loss, append to the command for baseline model
```
--style_loss_weight 0.1
```

To train the baseline model with word duplication loss, append to the command for baseline model
```
--neighbor_loss_weight 0.1
```

Since all training results will be saved into `checkpoint/yyyy-yy-yy_yy.yy.yy` folder, where `yyyy-yy-yy_yy.yy.yy` is the timestamp of the training command, we use unified evaluation protocols for all the training results. During the training process, the bleu@1 and bleu@3 scores will be saved into the corresponding log files (check `log/yyyy-yy-yy_yy.yy.yy.log`), and the sample caption results on test set can be obtained by the following command:
```
python sample_pretrain_caption_generator.py --gpu_id 0 --trained_timestamp yyyy-yy-yy_yy.yy.yy
```
and change the `--gpu_id` and `--trained_timestamp` if necessary.

## The list of the major software requirements that are needed to run our system
- Python 3.x
- PyTorch 1.6.0
- cudatoolkit 10.1.243
- nltk 3.5
- pycocotools
- scikit-image
- tqdm
