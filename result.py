__author__ = 'dan'


import os
import numpy as np
import utilis as utl
import imgprocess as imgp
import cv2


if __name__ == "__main__":
    folder = 'data/patient2'
    filename2 = 'MR.nii'

    folder_save = 'data/patient2_result'

    # Opening files MRI and PET
    image_data2, image_header2 = utl.openNiFile(os.path.join(folder, filename2))

    image_data2 = imgp.centerImgStack(image_data2, (500,500))
    utl.showImgStack(image_data2)

    print 'Patient 2 results \n'
    print 'Matching CT of PET with MR:'

    image_data1 = np.load(os.path.join('data/CTofPET_iter_best','rot_trans_PET_matrix_iter5.npy'))
    utl.showImgStack(image_data1)

    print 'Blended results of CT of PET with MR:'
    utl.showImgBlendStack(image_data2,image_data1)



    rows, clos, num_imgs = image_data2.shape

    for i in range(num_imgs):
        rows1,cols1 = image_data2[:,:,i].shape
        rows2,cols2 = image_data1[:,:,i].shape

        size = (max(rows1,rows2),max(cols1,cols2))

        img1_filled = imgp.fillImg(image_data2[:,:,i],size)
        img2_filled = imgp.fillImg(image_data1[:,:,i],size)

        img_show1 = ((img1_filled.astype(np.float) - np.min(img1_filled))/np.max(img1_filled))*255
        img_show2 = ((img2_filled.astype(np.float) - np.min(img2_filled))/np.max(img2_filled))*255

        img1_bgr = cv2.cvtColor(np.uint8(img_show1), cv2.COLOR_GRAY2BGR)
        img2_bgr = cv2.cvtColor(np.uint8(img_show2), cv2.COLOR_GRAY2BGR)
        img2_bgr = cv2.applyColorMap(img2_bgr, cv2.COLORMAP_HOT)

        dst = cv2.addWeighted(img1_bgr, 0.5, img2_bgr, 0.5, 0)
        cv2.imwrite(os.path.join(folder_save, 'CTofPET' ,'Img_fusion_CTofPET_MR_patient2_' + str(i) + '.png'), dst)



    print 'Matching PET with MR:'
    image_data1 = np.load(os.path.join('data/PET_iter_best','rot_trans_PET_matrix_iter17.npy'))
    utl.showImgStack(image_data1)

    print 'Blended results of PET with MR:'
    utl.showImgBlendStack(image_data2,image_data1)


    rows, clos, num_imgs = image_data2.shape

    for i in range(num_imgs):
        rows1,cols1 = image_data2[:,:,i].shape
        rows2,cols2 = image_data1[:,:,i].shape

        size = (max(rows1,rows2),max(cols1,cols2))

        img1_filled = imgp.fillImg(image_data2[:,:,i],size)
        img2_filled = imgp.fillImg(image_data1[:,:,i],size)

        img_show1 = ((img1_filled.astype(np.float) - np.min(img1_filled))/np.max(img1_filled))*255
        img_show2 = ((img2_filled.astype(np.float) - np.min(img2_filled))/np.max(img2_filled))*255

        img1_bgr = cv2.cvtColor(np.uint8(img_show1), cv2.COLOR_GRAY2BGR)
        img2_bgr = cv2.cvtColor(np.uint8(img_show2), cv2.COLOR_GRAY2BGR)
        img2_bgr = cv2.applyColorMap(img2_bgr, cv2.COLORMAP_HOT)

        dst = cv2.addWeighted(img1_bgr, 0.5, img2_bgr, 0.5, 0)
        cv2.imwrite(os.path.join(folder_save, 'PET' ,'Img_fusion_PET_MR_patient2_' + str(i) + '.png'), dst)


    folder = 'data/patient1'
    filename2 = 'FLAIR.nii'

    folder_save = 'data/patient1_result'

    # Opening files MRI and PET
    image_data2, image_header2 = utl.openNiFile(os.path.join(folder, filename2))

    image_data2 = imgp.centerImgStack(image_data2, (500,500))
    utl.showImgStack(image_data2)

    print 'Patient 1 results \n'
    print 'Matching CT of PET with MR:'

    image_data1 = np.load(os.path.join('data/CTofPET_iter_best_pat1','rot_trans_PET_matrix_iter5.npy'))
    utl.showImgStack(image_data1)

    print 'Blended results of CT of PET with MR:'
    utl.showImgBlendStack(image_data2,image_data1)



    rows, clos, num_imgs = image_data2.shape

    for i in range(num_imgs):
        rows1,cols1 = image_data2[:,:,i].shape
        rows2,cols2 = image_data1[:,:,i].shape

        size = (max(rows1,rows2),max(cols1,cols2))

        img1_filled = imgp.fillImg(image_data2[:,:,i],size)
        img2_filled = imgp.fillImg(image_data1[:,:,i],size)

        img_show1 = ((img1_filled.astype(np.float) - np.min(img1_filled))/np.max(img1_filled))*255
        img_show2 = ((img2_filled.astype(np.float) - np.min(img2_filled))/np.max(img2_filled))*255

        img1_bgr = cv2.cvtColor(np.uint8(img_show1), cv2.COLOR_GRAY2BGR)
        img2_bgr = cv2.cvtColor(np.uint8(img_show2), cv2.COLOR_GRAY2BGR)
        img2_bgr = cv2.applyColorMap(img2_bgr, cv2.COLORMAP_HOT)

        dst = cv2.addWeighted(img1_bgr, 0.5, img2_bgr, 0.5, 0)
        cv2.imwrite(os.path.join(folder_save, 'CTofPET' ,'Img_fusion_CTofPET_MR_patient1_' + str(i) + '.png'), dst)



    print 'Matching PET with MR:'
    image_data1 = np.load(os.path.join('data/PET_iter_best_pat1','rot_trans_PET_matrix_iter6.npy'))
    utl.showImgStack(image_data1)

    print 'Blended results of PET with MR:'
    utl.showImgBlendStack(image_data2,image_data1)


    rows, clos, num_imgs = image_data2.shape

    for i in range(num_imgs):
        rows1,cols1 = image_data2[:,:,i].shape
        rows2,cols2 = image_data1[:,:,i].shape

        size = (max(rows1,rows2),max(cols1,cols2))

        img1_filled = imgp.fillImg(image_data2[:,:,i],size)
        img2_filled = imgp.fillImg(image_data1[:,:,i],size)

        img_show1 = ((img1_filled.astype(np.float) - np.min(img1_filled))/np.max(img1_filled))*255
        img_show2 = ((img2_filled.astype(np.float) - np.min(img2_filled))/np.max(img2_filled))*255

        img1_bgr = cv2.cvtColor(np.uint8(img_show1), cv2.COLOR_GRAY2BGR)
        img2_bgr = cv2.cvtColor(np.uint8(img_show2), cv2.COLOR_GRAY2BGR)
        img2_bgr = cv2.applyColorMap(img2_bgr, cv2.COLORMAP_HOT)

        dst = cv2.addWeighted(img1_bgr, 0.5, img2_bgr, 0.5, 0)
        cv2.imwrite(os.path.join(folder_save, 'PET' ,'Img_fusion_PET_MR_patient1_' + str(i) + '.png'), dst)





