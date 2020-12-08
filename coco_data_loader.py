import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import nltk
import re

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from torch.utils.data import Dataset, DataLoader
import skimage.io
import skimage.transform

class COCODataset(Dataset):
    def __init__(self, dataDir, dataType, vocab, require_img=True, transform=None):
        '''
        Args:
            img_dir: Direcutory with all the images
            caption_file: Path to the factual caption file
            vocab: Vocab instance
            transform: Optional transform to be applied
        '''
        self.dataDir = dataDir
        self.dataType = dataType
        self.img_dir = os.path.join(dataDir, dataType)
        self.annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
        self.imgname_caption_list = self._get_imgname_and_caption(self.annFile)
        self.vocab = vocab
        self.transform = transform
        self.require_img = require_img

    def _get_imgname_and_caption(self, annFile):
        # initialize COCO api for caption annotations
        coco_caps = COCO(annFile)
        imgname_caption_list = []

        for anns in coco_caps.imgToAnns.values():
            # anns is 5-element list
            for ann in anns:
                cap = ann['caption'].strip().lower()
                image_id = ann['image_id']
                imgname = "COCO_{}_{:0>12}.jpg".format(self.dataType, image_id)

            img_and_cap = [imgname, cap]
            imgname_caption_list.append(img_and_cap)

        return imgname_caption_list

    def __len__(self):
        return len(self.imgname_caption_list)

    def __getitem__(self, ix):
        '''return one data pair (image and captioin)'''
        img_name = self.imgname_caption_list[ix][0]
        img_name = os.path.join(self.img_dir, img_name)
        caption = self.imgname_caption_list[ix][1]

        if self.require_img:
            image = skimage.io.imread(img_name)
            # convert gray to rgb
            if len(image.shape) == 2:
                image = np.stack((image,)*3, axis=-1)
            if self.transform is not None:
                image = self.transform(image)

        # convert caption to word ids
        r = re.compile("\.")
        tokens = nltk.tokenize.word_tokenize(r.sub("", caption).lower())
        caption = []
        caption.append(self.vocab('<s>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('</s>'))
        caption = torch.Tensor(caption)

        if self.require_img:
            return image, caption
        else:
            return caption


def get_data_loader(dataDir, dataType, vocab, batch_size, require_img=True,
                    transform=None, shuffle=False, num_workers=0):
    '''Return data_loader'''
    if transform is None:
        transform = transforms.Compose([
            Rescale((224, 224)),
            transforms.ToTensor()
            ])

    COCO = COCODataset(dataDir, dataType, vocab, require_img, transform)

    if require_img:
        data_loader = DataLoader(dataset=COCO,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset=COCO,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn_styled)
    return data_loader


class Rescale:
    '''Rescale the image to a given size
    Args:
        output_size(int or tuple)
    '''
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image = skimage.transform.resize(image, (new_h, new_w))

        return image

def collate_fn(data):
    '''create minibatch tensors from data(list of tuple(image, caption))'''
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # images : tuple of 3D tensor -> 4D tensor
    images = torch.stack(images, 0)

    # captions : tuple of 1D Tensor -> 2D tensor
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths)) for cap in captions]
    captions = torch.stack(captions, 0)

    return images, captions, lengths


def collate_fn_styled(captions):
    captions.sort(key=lambda x: len(x), reverse=True)

    # tuple of 1D Tensor -> 2D Tensor
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths)) for cap in captions]
    captions = torch.stack(captions, 0)

    return captions, lengths


def pad_sequence(seq, max_len):
    seq = torch.cat((seq, torch.zeros(max_len - len(seq))))
    return seq

if __name__ == "__main__":

    dataDir='./data/COCO'
    dataType='train2014'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

    # initialize COCO api for instance annotations
    coco=COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
    imgIds = coco.getImgIds(catIds=catIds );
    imgIds = coco.getImgIds(imgIds = [379520])
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]


    # load and display image
    # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
    # use url to load image
    I = io.imread(img['coco_url'])
    plt.axis('off')
    plt.imshow(I)
    plt.show()


    # load and display instance annotations
    plt.imshow(I); plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)


    # initialize COCO api for person keypoints annotations
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
    coco_kps=COCO(annFile)


    # load and display keypoints annotations
    plt.imshow(I); plt.axis('off')
    ax = plt.gca()
    annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco_kps.loadAnns(annIds)
    coco_kps.showAnns(anns)

    # initialize COCO api for caption annotations
    annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    coco_caps=COCO(annFile)


    # load and display caption annotations
    annIds = coco_caps.getAnnIds(imgIds=img['id']);
    anns = coco_caps.loadAnns(annIds)
    coco_caps.showAnns(anns)
    plt.imshow(I); plt.axis('off'); plt.show()