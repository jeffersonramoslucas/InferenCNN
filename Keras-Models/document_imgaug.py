import os
import cv2
import numpy as np
from imgaug import augmenters as iaa

_rotation_plus = iaa.Affine(rotate=3)
_rotation_minus = iaa.Affine(rotate=-3)
_translate_plus = iaa.Affine(translate_percent={"x": 0, "y": 0.2})
_translate_minus = iaa.Affine(translate_percent={"x": 0, "y": -0.2})
_scale_up = iaa.Affine(scale=1.02, order=[0, 1])
_scale_down = iaa.Affine(scale=0.8, order=[0, 1])

_brightness = iaa.Multiply(0.6)
_dropout = iaa.Dropout(p=0.09, per_channel=True)
_gaussian_noise = iaa.AdditiveGaussianNoise(scale=30, per_channel=True)
_gaussian_blur = iaa.GaussianBlur(sigma=(1.5, 1.6))
_pepper = iaa.Pepper(0.05)
_hue_and_saturation = iaa.AddToHueAndSaturation((-25, 15))
_contrast = iaa.ContrastNormalization((0.4, 0.7))

def _apply_perspective(img):
    rows, cols, ch = img.shape

    param = 0.10 * rows

    pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    pts2 = np.float32([[param, param], [cols - param, param], [cols, rows - param], [0, rows - param]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))

    return dst

_transformations = [
    iaa.Noop(),

    iaa.OneOf([
        _rotation_plus,
        _rotation_minus,
        _gaussian_blur,
        _brightness

    ]),

    #_translate_plus,
    
    #iaa.Sequential([
    #    _gaussian_noise,
    #   
    #]),

    #_brightness,

    

    #iaa.Sequential([
    #    _hue_and_saturation,
    #    iaa.Sometimes(0.8, _gaussian_noise)
    #]),
    #iaa.Sequential([
    #    iaa.Sometimes(0.6, _scale_down),
    #    _translate_minus
    #], random_order=False),

    #iaa.Sequential([
    #    iaa.Sometimes(0.6, _scale_up),
    #    _translate_minus
    #], random_order=False),

    #iaa.Sequential([
    #    _contrast,
    #    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=1.2))
    #])
]
 
    

def _augment_image(img, results):
    for t in _transformations:
        if isinstance(t, iaa.Augmenter):
            results.append(t.augment_image(img))
        else:
            results.append(t(img))

def resize_with_padding(img, size):
    width, height = img.shape[1], img.shape[0]

    target_h, target_w = size

    if width > height:
        ratio = float(target_w) / float(width)

        if height * ratio <= target_h:
            padding = int((target_h - ratio * height) / 2) + 1
            top = padding
            bottom = padding
            left = 0
            right = 0
            shape = (target_w, int(ratio * height))
            offset = (0, padding)
        else:
            ratio = float(target_h) / float(height)
            padding = int((target_w - ratio * width) / 2) + 1
            top = 0
            bottom = 0
            left = padding
            right = padding
            shape = (int(width * ratio), target_h)
            offset = (padding, 0)

    else:
        ratio = float(target_h) / float(height)

        if width * ratio <= target_w:
            padding = int((target_w - ratio * width) / 2) + 1
            top = 0
            bottom = 0
            left = padding
            right = padding
            shape = (int(ratio * width), target_h,)
            offset = (padding, 0)
        else:
            ratio = float(target_w) / float(width)
            padding = int((target_h - ratio * height) / 2) + 1
            top = padding
            bottom = padding
            left = 0
            right = 0
            shape = (target_w, int(ratio * height))
            offset = (padding, 0)

    resized_img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT)

    padded = padded[0:target_h, 0:target_w, :]

    return padded, ratio, offset

def document_imgaug( x , y ):

    x_generator = []
    y_generator = []
    # print('Aug generator')
    for i, img_array in enumerate(x):
        #img_array = np.asarray(img_array)
        #img_array = img_array.astype('float32')

        if(img_array.shape[2] == 3):
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img = img_array

        _augment_image(img, x_generator)
        

        for c in range(0, len(_transformations)):
            y_generator.append(y[i])
        # print(i)
        # print(i,'  --  ', y[i],'  --  ',len(x_generator))
    # print('Aug generated')

    return np.array(x_generator), np.array(y_generator)


