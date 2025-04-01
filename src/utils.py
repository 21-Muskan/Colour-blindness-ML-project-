import numpy as np
from skimage import color

def rgb_to_lms(rgb_img):
    return color.rgb2lms(rgb_img)

def lms_to_rgb(lms_img):
    return color.lms2rgb(lms_img)