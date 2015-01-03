__author__ = 'dan'

import medpy.io
import os
import cv2
import numpy as np

def nothing(x):
    pass

def openNiFile(file_name):
    image_data, image_header = medpy.io.load(file_name)
    return image_data, image_header

def saveNiFile(file_name, image_data, image_header = None):
    medpy.io.save(image_data, file_name, image_header)

def showImg(img):
    cv2.imshow('image',np.uint8(np.clip(img, 0, 255)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showImgStack(imgs):
    cv2.namedWindow('imageStack')
    current_pos = 0
    cv2.createTrackbar('img num','imageStack',0,imgs.shape[2]-1,nothing)

    while(1):
        cv2.imshow('imageStack',np.uint8(np.clip( imgs[:,:,current_pos], 0, 255)))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        current_pos = cv2.getTrackbarPos('img num','imageStack')
    cv2.destroyAllWindows()



if __name__ == "__main__":
    folder = 'data/patient2'
    filename1 = 'CTofPET.nii'

    image_data, image_header = openNiFile(os.path.join(folder,filename1))
    showImgStack(image_data)






