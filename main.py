import argparse

__author__ = 'dan'

import medpy.io
import medpy.metric
import os
import utilis as utl
import imgprocess as imgp
import aligment as alig


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PET and MRI images aligment. Goal of this application is to align PET'
                                                 'and MRI images. To achive this goal, images are centred, resliced for'
                                                 'easier processing. After that iterative algorithm was proposed based '
                                                 'on mutual information computation for whole images stack.')
    parser.add_argument('-imgPET', help='Name of NIfTI image of PET.', required=True)
    parser.add_argument('-imgMRI', help='Name of NIfTI image of MRI.', required=True)
    parser.add_argument('-inputFold', help='Path to folder were input files (PET & MRI) are stored', required=True)
    parser.add_argument('-outputFold', help='Path to folder were output files (PET images after every iteration and '
                                            'blended PET and MRI images) are stored', required=True)

    args = parser.parse_args()

    folder_input = args.inputFold
    folder_output = args.outputFold
    filename_PET = args.imgPET
    filename_MRI = args.imgMRI

    # Opening files MRI and PET
    image_data_PET, image_header_PET = utl.openNiFile(os.path.join(folder_input, filename_PET))
    image_data_MRI, image_header_MRI = utl.openNiFile(os.path.join(folder_input, filename_MRI))

    # Show file1 and detalis
    pix_spacing1 = medpy.io.get_pixel_spacing(image_header_PET)
    print "Pixel spacing for PET: " + str(pix_spacing1)
    utl.showImgStack(image_data_PET)

    # Show file2 and detalis
    pix_spacing2 = medpy.io.get_pixel_spacing(image_header_MRI)
    print "Pixel spacing for MR: " + str(pix_spacing2)
    utl.showImgStack(image_data_MRI)

    # Remove empty images
    image_data_PET = imgp.removeEmptyImgs(image_data_PET, 30)
    utl.showImgStack(image_data_PET)


    # Reslicing images
    image_data_PET = imgp.resliceImgStack(image_data_PET, image_data_MRI.shape[2])
    utl.showImgStack(image_data_PET)

    #Correct scale assuming correct values in header
    scale = pix_spacing1[0]/pix_spacing2[0]
    print "Correct scale from header = " + str(scale)

    # Centering images
    image_data_PET = imgp.centerImgStack(image_data_PET, (500,500))
    utl.showImgStack(image_data_PET)

    image_data_MRI = imgp.centerImgStack(image_data_MRI, (500,500))
    utl.showImgStack(image_data_MRI)

    # Scaling image stack
    image_data_PET = imgp.scaleImgStack(image_data_PET, scale, option='crop')
    utl.showImgStack(image_data_PET)

    # Looking for the best aligment
    alig.iterativeAligment(image_data_PET,image_data_MRI, folder_output)

