
import numpy as np
import os
import medpy
import cv2
import imgprocess as imgp


def compareImg(img1, img2, method = 'mutual'):

    """
    Function returns information about two images similarity

    :param img1: image1(2D nparray)
    :param img2: image2(2D nparray)
    :param method: method of comparison images
    :return: dependent from method
        mutual - mutual information coefficient
    """
    if method == 'mutual':
        return medpy.metric.mutual_information(img1,img2)


def saveCompareMatrix(matrix, filename):
    """
    Function saves matrix with similarity measure for two stack of images stored in matrix

    :param matrix: similarity matrix
    :param filename: output filename
    """
    img_save = (matrix.astype(np.float) - np.min(matrix))/np.max(matrix)*255
    img_bgr = cv2.cvtColor(np.uint8(img_save), cv2.COLOR_GRAY2BGR)
    colormap_img = cv2.applyColorMap(img_bgr, cv2.COLORMAP_JET)
    cv2.imwrite(filename, colormap_img)

def saveImage(matrix, filename):
    """
    Function saves image to file

    :param matrix: input image matrix
    :param filename: output filename
    """
    img_save = (matrix.astype(np.float) - np.min(matrix))/np.max(matrix)*255
    img_bgr = cv2.cvtColor(np.uint8(img_save), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(filename, img_bgr)

def saveBlendedImages(image_data1,image_data2, folder_output):

    """
    Saves blended image stacks to PNG files

    :param image_data1: images stack (3D nparray)
    :param image_data2: images stack (3D nparray)
    :param folder_output: output folder
    """
    if not os.path.exists(os.path.join(folder_output,'PET_MRI_results')):
        os.makedirs(os.path.join(folder_output,'PET_MRI_results'))

    rows, cols, num_imgs = image_data1.shape

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
        cv2.imwrite(os.path.join(folder_output,'PET_MRI_results', 'Img_fusion_' + str(i) + '.png'), dst)




def iterativeAligment(image_data_PET, image_data_MRI, folder_output, save_arrays = True, save_iters = True):

    """
    Iterative approach to find bes image alignment of two image stacks

    :param image_data_PET: images stack (3D nparray)
    :param image_data_MRI: images stack (3D nparray)
    :param folder_output: folder output for snapshots after each iteration
    :param save_arrays: flag indicating saving arrays or not
    :param save_iters: flag indicating saving image for each iteration or not
    """
    rows_PET, cols_PET, num_PET = image_data_PET.shape
    rows_MRI, cols_MRI, num_MRI = image_data_MRI.shape
    rot_trans_matrix = np.zeros((num_PET, num_MRI), dtype=np.float32)

    if not os.path.exists(os.path.join(folder_output,'PET_iteration')):
        os.makedirs(os.path.join(folder_output,'PET_iteration'))

    rotation_list = [-5, 0, 5]
    translation_x = [-5, 0, 5]
    translation_y = [-5, 0, 5]

    diff = True
    iter_counter = 0
    back_params = (0, 0, 0)

    while(diff):
        trace_max = 0
        params_max = (0, 0, 0)
        for rot in rotation_list:
            image_data_PET_rot_trans = imgp.rotateImgStack(image_data_PET, rot, 'crop')
            for tr_x in translation_x:
                for tr_y in translation_y:
                    image_data_PET_rot_trans = imgp.translateImgStack(image_data_PET_rot_trans,(tr_x,tr_y),option='crop')
                    for im_1 in range(num_PET):
                            rot_trans_matrix[im_1,im_1] = compareImg(image_data_PET_rot_trans[:,:,im_1], image_data_MRI[:,:,im_1])

                    mat_trace = np.trace(rot_trans_matrix)
                    if mat_trace > trace_max:
                        trace_max = mat_trace
                        params_max = (rot, tr_x, tr_y)

        if params_max == (0, 0, 0) or params_max == back_params:
            diff = False
        back_params = (params_max[0]*(-1),params_max[1]*(-1),params_max[2]*(-1))

        image_data_PET_rot_trans= imgp.rotateImgStack(image_data_PET, params_max[0], option= 'crop')
        image_data_PET = imgp.translateImgStack(image_data_PET_rot_trans,(params_max[1], params_max[2]), option='crop')
        iter_counter += 1
        print 'Best parameters in iteration: ' + str(iter_counter)
        print 'Rotation: ' + str(params_max[0])
        print 'Translation x: ' + str(params_max[1])
        print 'Translation y: ' + str(params_max[2])

        if save_arrays:
            np.save(os.path.join(folder_output,'PET_iteration','rot_trans_PET_matrix_iter'+
                                                                            str(iter_counter)), image_data_PET)
        if save_iters:
            saveImage(image_data_PET[:,:,0], os.path.join(folder_output,'PET_iteration',
                                                                       'rot_trans_PET_img_iter_'+ str(iter_counter) + '.png'))
    saveBlendedImages(image_data_PET, image_data_MRI, folder_output)