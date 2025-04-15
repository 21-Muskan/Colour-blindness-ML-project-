import cv2
import numpy as np

def rgb_to_lms(rgb_img):
    return cv2.transform(rgb_img, np.array([
        [17.8824, 43.5161, 4.11935],
        [3.45565, 27.1554, 3.86714],
        [0.0299566, 0.184309, 1.46709]
    ]))

def lms_to_rgb(lms_img):
    return cv2.transform(lms_img, np.array([
        [0.0809, -0.1305, 0.1167],
        [-0.0102, 0.0540, -0.1136],
        [-0.0004, -0.0041, 0.6935]
    ]))

def apply_protanopia_simulation(rgb_img):
    lms = rgb_to_lms(rgb_img)
    simulated = cv2.transform(lms, np.array([
        [0, 2.02344, -2.52581],
        [0, 1, 0],
        [0, 0, 1]
    ]))
    return lms_to_rgb(simulated)

def apply_deuteranopia_simulation(rgb_img):
    lms = rgb_to_lms(rgb_img)
    simulated = cv2.transform(lms, np.array([
        [1, 0, 0],
        [0.494207, 0, 1.24827],
        [0, 0, 1]
    ]))
    return lms_to_rgb(simulated)

def apply_tritanopia_simulation(rgb_img):
    lms = rgb_to_lms(rgb_img)
    simulated = cv2.transform(lms, np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-0.395913, 0.801109, 0]
    ]))
    return lms_to_rgb(simulated)