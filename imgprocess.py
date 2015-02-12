__author__ = 'dan'

import cv2
import numpy as np
import math


def translateImg(img, vec, option='extend'):
    """
    Function translates image by vector

    :param img: image (2D nparray)
    :param vec: translation vector
    :param option: 'extend' - output image has size depended from scale. 'crop' - output image has fixed size
    :return: new image
    """
    rows, cols = img.shape

    if option == 'extend':
        final_shape = (cols + abs(vec[0]), rows + abs(vec[1]))
    elif option == 'crop':
        final_shape = (cols, rows)
    else:

        return

    M = np.float64([[1, 0, vec[0]], [0, 1, vec[1]]])
    dst = cv2.warpAffine(img, M, final_shape)

    return dst


def translateImgStack(imgs, vec, option='extend'):
    """
    Function translates image stack by vector

    :param imgs: images stack (3D nparray)
    :param vec: translation vector
    :param option: 'extend' - output image stack has size depended from scale. 'crop' - output image stack has fixed size
    :return: new image stack
    """
    rows, cols, num_imgs = imgs.shape

    if option == 'extend':
        new_imgs = np.zeros((rows + abs(vec[1]), cols + abs(vec[0]), num_imgs), dtype=imgs.dtype)
    else:
        new_imgs = np.zeros((rows, cols, num_imgs), dtype=imgs.dtype)

    for i in range(num_imgs):
        new_imgs[:, :, i] = translateImg(imgs[:, :, i], vec, option)

    return new_imgs


def centerImg(img, size):
    """
    Function puts images  to center of output image

    :param img: image (2D nparray)
    :param size: size of output image
    :return: new image
    """
    rows, cols = img.shape

    dst = np.zeros((size[0], size[1]), dtype=img.dtype)

    dst[int(size[0] / 2) - int(rows / 2):int(size[0] / 2) + int(rows / 2),
    int(size[1] / 2) - int(cols / 2):int(size[1] / 2) + int(cols / 2)] = img

    return dst


def centerImgStack(imgs, size):
    """
    Function puts images from image stack to center of output image stack

    :param imgs: images stack (3D nparray)
    :param size: size of output image stack
    :return: new image stack
    """
    rows, cols, num_imgs = imgs.shape

    new_imgs = np.zeros((size[0], size[1], num_imgs), dtype=imgs.dtype)

    for i in range(num_imgs):
        new_imgs[:, :, i] = centerImg(imgs[:, :, i], size)

    return new_imgs


def rotateImg(img, deg, option='extend'):
    """
    Function rotates image

    :param img: image (2D nparray)
    :param deg: rotation degrees
    :param option: 'extend' - output image has size depended from scale. 'crop' - output image has fixed size
    :return: new image
    """
    rows, cols = img.shape
    factor = math.pi / 180

    if option == 'extend':
        final_shape = ( int(math.ceil(abs(cols * math.cos(factor * deg)) + abs(rows * math.sin(factor * deg)))),
                        int(math.ceil(abs(cols * math.sin(factor * deg)) + abs(rows * math.cos(factor * deg)))))
    elif option == 'crop':
        final_shape = (cols, rows)
    else:
        return

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
    dst = cv2.warpAffine(img, M, final_shape)

    return dst


def rotateImgStack(imgs, deg, option='extend'):
    """
    Function rotates images stack

    :param imgs: images stack (3D nparray)
    :param deg: rotation degrees
    :param option: 'extend' - output image stack has size depended from scale. 'crop' - output image stack has fixed size
    :return: new image stack
    """
    rows, cols, num_imgs = imgs.shape
    factor = math.pi / 180

    if option == 'extend':
        new_imgs = np.zeros((math.ceil(abs(cols * math.sin(factor * deg)) + abs(rows * math.cos(factor * deg))),
                             math.ceil(abs(cols * math.cos(factor * deg)) + abs(rows * math.sin(factor * deg))),
                             num_imgs), dtype=imgs.dtype)
    else:
        new_imgs = np.zeros((rows, cols, num_imgs), dtype=imgs.dtype)

    for i in range(num_imgs):
        new_imgs[:, :, i] = rotateImg(imgs[:, :, i], deg, option)

    return new_imgs


def scaleImg(img, scale, option='extend'):
    """
    Function scales (resize) image

    :param img: image (2D nparray)
    :param scale: scale of output image
    :param option: 'extend' - output image has size depended from scale. 'crop' - output image has fixed size
    :return: new image
    """
    rows, cols = img.shape

    if option == 'extend':
        dst = cv2.resize(img, (int(scale * cols), int(scale * rows)), interpolation=cv2.INTER_CUBIC)
    elif option == 'crop':
        dst_full = cv2.resize(img, (int(scale * cols), int(scale * rows)), interpolation=cv2.INTER_CUBIC)
        dst = dst_full[int(scale * rows / 2) - int(rows / 2):int(scale * rows / 2) + int(rows / 2),
              int(scale * cols / 2) - int(cols / 2):int(scale * cols / 2) + int(cols / 2)]
    else:
        return

    return dst


def scaleImgStack(imgs, scale, option='extend'):
    """
    Function scales (resize) image stack

    :param imgs: images stack (3D nparray)
    :param scale: scale of output image stack
    :param option: 'extend' - output image stack has size depended from scale. 'crop' - output image stack has fixed size
    :return: new image stack
    """
    rows, cols, num_imgs = imgs.shape

    if option == 'extend':
        new_imgs = np.zeros((rows * scale, cols * scale, num_imgs), dtype=imgs.dtype)
    elif option == 'crop':
        new_imgs = np.zeros((rows, cols, num_imgs), dtype=imgs.dtype)
    else:
        return

    for i in range(num_imgs):
        new_imgs[:, :, i] = scaleImg(imgs[:, :, i], scale, option)

    return new_imgs


def fillImg(img, size):
    """
    Function put image into bigger empty images (rest of image is zero)

    :param img: image(2D nparray)
    :param size: output size
    :return: output image
    """
    rows, cols = img.shape

    dst = np.zeros(size, dtype=img.dtype)
    dst[0:rows, 0:cols] = img
    return dst


def fillImgStack(imgs, size):
    """
    Function put image stack into bigger empty images (rest of image is zero)

    :param imgs: images stack (3D nparray)
    :param size: output size
    :return: new image stack
    """
    rows, cols, num_imgs = imgs.shape

    new_imgs = np.zeros((size[0], size[1], num_imgs))
    for i in range(num_imgs):
        new_imgs[:, :, i] = fillImg(imgs[:, :, i], size)

    return new_imgs


def resliceImgStack(imgs, num):
    """
    Function performs reslicing based on linear interpolation of image stack

    :param imgs: images stack (3D nparray)
    :param num: number of output images in image stack after reslicing
    :return: new image stack
    """
    rows, cols, num_imgs = imgs.shape

    xx = np.linspace(0, num_imgs - 1, num=num_imgs)
    xi = np.linspace(0, num_imgs, num=num)

    new_imgs = np.zeros((rows, cols, num), dtype=imgs.dtype)

    for r in range(rows):
        for c in range(cols):
            new_imgs[r, c, :] = np.interp(xi, xx, imgs[r, c, :].flatten()).astype(np.int16)
    return new_imgs


def removeEmptyImgs(imgs, thres):
    """
    Function removes empty images from image stack

    :param imgs: images stack (3D nparray)
    :param thres: below threshold image is removed from image stack
    :return:  new image stack
    """
    rows, cols, num_imgs = imgs.shape

    new_imgs = np.copy(imgs)
    bad_list = []
    for n in range(num_imgs):
        if (int(np.max(new_imgs[:, :, n]) < thres)):
            bad_list.append(n)

    new_imgs = np.delete(new_imgs, bad_list, 2)
    return new_imgs



