from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2
import random
from .img_aug import get_fun


suffix_list = [".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"]

def get_file_path_list(path):
    cnt_ = 0
    imagePathList = []
    for root, dir, files in os.walk(path):
        for file in files:
            try:
                full_path = os.path.join(root, file)


                suffix_img = full_path.rsplit(".",1)
                if(len(suffix_img)<2):
                    continue
                suffix_img = "." + suffix_img[-1]
                if suffix_img in suffix_list:
                    cnt_ += 1
                    print (cnt_, "  :: ", full_path)
                    imagePathList.append(full_path)

            except IOError:
                continue

    print("=====end get_file_path_list============================\n")
    return imagePathList

def get_label(img_path):
    pos_2 = img_path.rfind(".")
    pos_1 = img_path.rfind("_")
    label = img_path[pos_1+1:pos_2]
    return label

def LstmImgStandardization(img, ratio, stand_w, stand_h):
    img_h, img_w, _ = img.shape
    if img_h < 2 or img_w < 2:
        return
    if 32 == img_h and 320 == img_w:
        return img

    ratio_now = img_w * 1.0 / img_h
    if ratio_now <= ratio:
        mask = np.ones((img_h, int(img_h * ratio), 3), dtype=np.uint8) * 255
        mask[0:img_h,0:img_w,:] = img
    else:
        mask = np.ones((int(img_w*1.0/ratio), img_w, 3), dtype=np.uint8) * 255
        mask[0:img_h, 0:img_w, :] = img

    mask_stand = cv2.resize(mask,(stand_w, stand_h),interpolation=cv2.INTER_AREA)
    return mask_stand

class _OWN(data.Dataset):
    def __init__(self, config, is_train=True):

        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W
        self.ratio_keep_origin = config.DATASET.RATIO_KEEP_ORIGIN

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        dir_img = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']

        imagePathList = get_file_path_list(dir_img)
        random.shuffle(imagePathList)
        len_img = len(imagePathList)
        self.labels = []
        for i in range(len_img):
            imagePath = imagePathList[i]

            label = get_label(imagePath)
            # label = label.replace("@", "/")
            # label = label.replace(".png", "")
            # label = label.replace(".jpg", "")
            # label = label.replace(".PNG", "")
            # label = label.replace(".jpeg", "")
            # label = label.replace(".JPG", "")
            # label = label.replace(".JPEG", "")
            # label = label.replace(" ", "")
            dict_save = {}
            dict_save[imagePath] = label
            self.labels.append(dict_save)

        if is_train:
            print("load train  {} images!".format(self.__len__()))
        else:
            print("load val {} images!".format(self.__len__()))


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_path = list(self.labels[idx].keys())[0]
        # print(img_path,"   ::   ",self.labels[idx].values())
        img_src = cv2.imread(img_path)
        if img_src is None:
            print('img_path is None::: {}'.format(img_path))

            idx = idx + 1
            img_path = list(self.labels[idx].keys())[0]
            img_src = cv2.imread(img_path)

        if random.random() > (1 - self.ratio_keep_origin) :
            # print("================no aug===========")
            img_aug = img_src
        else:
            # print("================aug===========")
            fun_c1 = get_fun()
            img_aug = fun_c1(img_src)
            fun_c2 = get_fun()
            while fun_c2 == fun_c1:
                fun_c2 = get_fun()

            img_aug = fun_c2(img_src)

        img = LstmImgStandardization(img_aug, 10, self.inp_w, self.inp_h)

        img_h, img_w, _ = img.shape
        img = np.reshape(img, (self.inp_h, self.inp_w, 3))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, idx