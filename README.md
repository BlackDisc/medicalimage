medicalimage
============

Project for medical image processing coures

Goal of this project is to realign PET to MRI.

==============================================

Dependencies: 

[Numpy](http://www.scipy.org/install.html)
[MedPy](https://pypi.python.org/pypi/MedPy)
[OpenCV](http://opencv.org/)

==============================================

# Files description

## main.py

Main file responsible for runing whole alignment process. Avaliable options:
* [-h]  - displays help message 
* [-imgPET] - name of NIfTI image of PET 
* [-imgMRI] - name of NIfTI image of MRI
* [-inputFold] - path to folder were input files (PET & MRI) are stored
* [-outputFold] - path to folder were output files (PET images after every iteration and 
                  blended PET and MRI images) are stored

Example of usage:
'''
python main.py -imgPET PET.nii -imgMRI MRI.nii -inputFold /home/login/data/patient1/ -outputFold /home/login/data/patient1_results/
'''

## results.py

File responsible for results presentation. It assumes that 

## alignment.py



## imgprocess.py

Module with simple image processing functions (image translation, rotation, resize)

## utilis.py

Module with functions responsible for opening/closing files and displaying nparrays





