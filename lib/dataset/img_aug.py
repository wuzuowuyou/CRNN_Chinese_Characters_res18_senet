import os
import cv2
import numpy as np
import random


def colorjitter(img):
    '''
    ### Different Color Jitter ###
    img: image
    cj_type: {b: brightness, s: saturation, c: constast}

    '''
    # print("==========colorjitter====================")

    list_type = ["b", "s", "c"]
    cj_type = random.choice(list_type)
    if cj_type == "b":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "s":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "c":
        brightness = 10
        contrast = random.randint(40, 100)
        dummy = np.int16(img)
        dummy = dummy * (contrast / 127 + 1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img


def noisy(img):
    '''
    ### Adding Noise ###
    img: image
    cj_type: {gauss: gaussian, sp: salt & pepper}

    '''
    # print("==========noisy====================")

    list_type = ["gauss", "sp"]
    noise_type = random.choice(list_type)

    if noise_type == "gauss":
        image = img.copy()
        mean = 0
        st = 0.7
        gauss = np.random.normal(mean, st, image.shape)
        gauss = gauss.astype('uint8')
        image = cv2.add(image, gauss)
        return image

    elif noise_type == "sp":
        image = img.copy()
        prob = 0.05
        if len(image.shape) == 2:
            black = 0
            white = 255
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        image[probs < (prob / 2)] = black
        image[probs > 1 - (prob / 2)] = white
        return image


def filters(img):
    '''
    ### Filtering ###
    img: image
    f_type: {blur: blur, gaussian: gaussian, median: median}

    '''

    # print("==========filters====================")

    list_type = ["blur", "gaussian", "median"]
    f_type = random.choice(list_type)
    # print(f_type)

    if f_type == "blur":
        image = img.copy()
        fsize = 5
        return cv2.blur(image, (fsize, fsize))

    elif f_type == "gaussian":
        image = img.copy()
        fsize = 5
        return cv2.GaussianBlur(image, (fsize, fsize), 0)

    elif f_type == "median":
        image = img.copy()
        fsize = 5
        return cv2.medianBlur(image, fsize)


def gaussain_noise(img):
    # print("==========gaussain_noise====================")
    img = img.astype(np.uint8)
    h, w, c = img.shape
    list_var = [0.4, 0.38, 0.22, 6, 7, 2, 3, 4, 5, 6, 12, 10, 5, 8, 9]
    var = random.choice(list_var)
    list_mean = [0, 0.5, 0.08, 0.5, 9, 1, 2, 3, 4, 5, 6, 7, 8]
    mean = random.choice(list_mean)
    # print(var,mean)
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (h, w, c))
    gauss = gauss.reshape(h, w, c).astype(np.uint8)
    noisy = img + gauss
    return noisy


def img_contrast(img):
    # print("==========img_contrast====================")
    min_s, max_s, min_v, max_v = 0, 20, 0, 25
    img = img.astype(np.uint8)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _s = random.randint(min_s, max_s)
    _v = random.randint(min_v, max_v)
    if _s >= 0:
        hsv_img[:, :, 1] += _s
    else:
        _s = - _s
        hsv_img[:, :, 1] -= _s
    if _v >= 0:
        hsv_img[:, :, 2] += _v
    else:
        _v = - _v
        hsv_img[:, :, 2] += _v
    out = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return out


def rotate_func(img):
    '''
    like PIL, rotate by degree, not radians
    '''
    # print("==========rotate_func====================")
    fill = (255, 255, 255)
    list_ang = [0.4, -0.4, 0.25, -0.25, 0.3, -0.3, 0.45, -0.45, 0.6, -0.6, 0.7, -0.7, 0.8, -0.8, 0.9, -0.9, 0.1008,
                -0.852, 0.5, -0.5, 1, -1, 0.8, -0.8, 1.2, -1.2, 1.142, -0.952, 1.111, -1.111]
    degree = random.choice(list_ang)
    # print(degree)
    H, W = img.shape[0], img.shape[1]
    center = W / 2, H / 2
    M = cv2.getRotationMatrix2D(center, degree, 1)
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill)
    return out

def get_fun():
    fun_apply_list = [colorjitter, noisy, filters, rotate_func, gaussain_noise, rotate_func, img_contrast, rotate_func]
    fun_apply = random.choice(fun_apply_list)
    return fun_apply


if __name__ == "__main__":

    dir_img = "/data_1/everyday/0507/123/"
    list_img = os.listdir(dir_img)
    for img_name in list_img:
        path_img = dir_img + img_name

        path_img = "/data_1/everyday/0507/123/24_hello.jpg"
        img = cv2.imread(path_img)

        while True:
            fun_apply_list = [colorjitter, noisy, filters, gaussain_noise, img_contrast, rotate_func]
            fun_apply = random.choice(fun_apply_list)

            print(fun_apply)

            img_aug = fun_apply(img)

            cv2.imshow("img_src", img)
            cv2.imshow("img_aug", img_aug)
            cv2.waitKey(0)