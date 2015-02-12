__author__ = 'dan'

import medpy.io
import os
import cv2
import numpy as np
import imgprocess as imgp


def nothing(x):
    pass


def openNiFile(file_name):
    """
    Function opens NIfTI file and returns it as nparray

    :param file_name: name of file
    :return: 3D nparray
    """
    image_data, image_header = medpy.io.load(file_name)
    if (np.min(image_data) < 0):
        image_data = image_data - np.min(image_data)
    return np.asarray(image_data, dtype=np.uint16), image_header


def saveNiFile(file_name, image_data, image_header=None):
    """
    Saves nparray to NIfTI file

    :param file_name: name of file
    :param image_data: image stack (3D nparray)
    :param image_header: image header
    """
    medpy.io.save(image_data, file_name, image_header)


def showImg(img, title='image'):
    """
    Displays image using OpenCV API

    :param img: 2D nparray
    :param title: title of window
    """
    img_show = ((img.astype(np.float) - np.min(img)) / np.max(img)) * 255
    cv2.imshow(title, np.uint8(img_show))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showImgStack(imgs, title ='imageStack'):
    """
    Displays image stack, images can be changed using track bar

    :param imgs: image stack (3D nparray)
    :param title: title of window
    """
    cv2.namedWindow(title)
    current_pos = 0
    cv2.createTrackbar('img num', title, 0, imgs.shape[2] - 1, nothing)

    while (1):
        img_show = (imgs[:, :, current_pos].astype(np.float) - np.min(imgs[:, :, current_pos])) / np.max(
            imgs[:, :, current_pos]) * 255
        cv2.imshow(title, np.uint8(img_show))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        current_pos = cv2.getTrackbarPos('img num', title)
    cv2.destroyAllWindows()


def showImgColormap(img, title='image'):
    """
    Displays image using OpenCV API with colormap

    :param img: 2D nparray
    :param title: title of window
    """
    img_show = (img.astype(np.float) - np.min(img)) / np.max(img) * 255
    img_bgr = cv2.cvtColor(np.uint8(img_show), cv2.COLOR_GRAY2BGR)
    colormap_img = cv2.applyColorMap(img_bgr, cv2.COLORMAP_HOT)
    cv2.imshow(title, colormap_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showImgColormapStack(imgs, title='imageStack'):
    """
    Displays image stack with colormap, images can be changed using track bar

    :param imgs: image stack (3D nparray)
    :param title: title of window
    """
    cv2.namedWindow(title)
    current_pos = 0
    cv2.createTrackbar('img num', title, 0, imgs.shape[2] - 1, nothing)

    while (1):
        img_show = (imgs[:, :, current_pos].astype(np.float) - np.min(imgs[:, :, current_pos])) / np.max(
            imgs[:, :, current_pos]) * 255
        img_bgr = cv2.cvtColor(np.uint8(img_show), cv2.COLOR_GRAY2BGR)
        colormap_img = cv2.applyColorMap(img_bgr, cv2.COLORMAP_HOT)
        cv2.imshow(title, colormap_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        current_pos = cv2.getTrackbarPos('img num', title)
    cv2.destroyAllWindows()


def showImgBlend(img1, img2, coef=(0.5, 0.5), title='image'):
    """
    Displays two images blended. Second image is blended with color map

    :param img1: image (2D nparray)
    :param img2: image (2D nparray). This image will be blended with colormap
    :param coef: tuple with blending coefficients
    :param title: title of window
    """
    rows1, cols1 = img1.shape
    rows2, cols2 = img2.shape

    size = (np.max(rows1, rows2), np.max(cols1, cols2))

    img1_filled = imgp.fillImg(img1, size)
    img2_filled = imgp.fillImg(img2, size)

    img_show1 = ((img1_filled.astype(np.float) - np.min(img1_filled)) / np.max(img1_filled)) * 255
    img_show2 = ((img2_filled.astype(np.float) - np.min(img2_filled)) / np.max(img2_filled)) * 255

    img1_bgr = cv2.cvtColor(np.uint8(img_show1), cv2.COLOR_GRAY2BGR)
    img2_bgr = cv2.cvtColor(np.uint8(img_show2), cv2.COLOR_GRAY2BGR)
    img2_bgr = cv2.applyColorMap(img2_bgr, cv2.COLORMAP_HOT)

    dst = cv2.addWeighted(img1_bgr, coef[0], img2_bgr, coef[1], 0)

    cv2.imshow(title, dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showImgBlendStack(imgs1, imgs2, coef=(0.5, 0.5), title='imageStack'):
    """
    Displays two image stacks blended. Second image stack is blended with color map

    :param img1: image stack (3D nparray)
    :param img2: image stack (3D nparray). This image stack will be blended with colormap
    :param coef: tuple with blending coefficients
    :param title: title of window
    """
    cv2.namedWindow(title)
    current_pos = 0
    cv2.createTrackbar('img num', title, 0, imgs1.shape[2] - 1, nothing)

    while (1):
        rows1, cols1 = imgs1[:, :, current_pos].shape
        rows2, cols2 = imgs2[:, :, current_pos].shape

        size = (max(rows1, rows2), max(cols1, cols2))

        img1_filled = imgp.fillImg(imgs1[:, :, current_pos], size)
        img2_filled = imgp.fillImg(imgs2[:, :, current_pos], size)

        img_show1 = ((img1_filled.astype(np.float) - np.min(img1_filled)) / np.max(img1_filled)) * 255
        img_show2 = ((img2_filled.astype(np.float) - np.min(img2_filled)) / np.max(img2_filled)) * 255

        img1_bgr = cv2.cvtColor(np.uint8(img_show1), cv2.COLOR_GRAY2BGR)
        img2_bgr = cv2.cvtColor(np.uint8(img_show2), cv2.COLOR_GRAY2BGR)
        img2_bgr = cv2.applyColorMap(img2_bgr, cv2.COLORMAP_HOT)

        dst = cv2.addWeighted(img1_bgr, coef[0], img2_bgr, coef[1], 0)
        cv2.imshow(title, dst)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        current_pos = cv2.getTrackbarPos('img num', title)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    folder = 'data/patient1'
    filename1 = 'CTofPET-crop256.nii'

    image_data, image_header = openNiFile(os.path.join(folder, filename1))
    print medpy.io.get_pixel_spacing(image_header)
    print medpy.io.get_offset(image_header)
    showImgStack(image_data)
    #show3DImg(image_data)






