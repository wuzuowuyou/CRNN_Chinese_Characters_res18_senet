import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import os
import random
import struct


def deal_cfg(path_cfg):
    with open(path_cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    return config


def recognition(config, img, model, converter, device):
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.type(torch.FloatTensor)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)

    print("preds size=",preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    # print('results: {0}'.format(sim_pred))
    return sim_pred


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
    label = label.replace("@", "/")
    return label

if __name__ == '__main__':
    path_cfg = "./lib/config/OWN_config.yaml"
    path_pth = "./checkpoint_115_acc_0.9850.pth"
    path_img_dir = "./myfile/data_sample/val/2/"

    print("cuda?", torch.cuda.is_available())
    config = deal_cfg(path_cfg)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(path_pth))
    checkpoint = torch.load(path_pth)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    list_path_img = get_file_path_list(path_img_dir)
    random.shuffle(list_path_img)


    f = open("crnn_res18.wts", 'w')
    f.write("{}\n".format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
    print("\nSuccessfully generated wts")