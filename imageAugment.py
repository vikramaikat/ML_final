'''
####################################################
#              Methods to Augment Data             #
####################################################
Depends on:
pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely
pip install pip install imgaug

'batch' inputs are lists of numpy array images
####################################################
'''
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa

# Represents position errors that lead to anatomy cutoff
# min_scale, max_scale: range of image zoom (1+)
# shift: max amount images shift by
def aug_position_error(batch, min_scale, max_scale, shift):
    seq = iaa.Sequential([iaa.Affine(translate_percent={"x": (-shift, shift), "y": (-shift, shift)}),
                          iaa.Affine(scale=(min_scale, max_scale))])
    return seq(images=batch)

# Add artifacts that occulde parts of the image
# min_drop, max_drop represent percent of pixels to remove (0-1)
# size is dispurtion factor (smaller numbers are bigger artifacts)
def aug_artifact_error(batch, min_drop, max_drop, size):
    aug = iaa.CoarseDropout((min_drop, max_drop), size_percent=size)
    return aug(images=batch)

# Creates a motion blur effect (0+)
# blur_amount: magnitude of bluring (higher numbers more bluring)
def aug_motion_blur(batch, blur_amount):
    aug = iaa.MotionBlur(k=blur_amount, angle=[-45, 45])
    return aug(images=batch)

# Create a ghost image effect (from lateral motion)
# min_alpha, max_alpha: range of mixing coefficients (0-1)
# shift: max percent ghost image is shifted by (0-1)
def aug_motion_ghost(batch, min_alpha, max_alpha, shift):
    seq = iaa.Alpha(
        factor=(min_alpha, max_alpha),
        first=iaa.Affine(translate_percent={"x": (-shift, shift), "y": (-shift, shift)}),
        per_channel=False)

    return seq(images=batch)

# Changes the exposure of the image
# Exposures greater than 1 overexpose
# Exposures less than 1 overexpose
def aug_machine_expose(batch, min_expose, max_expose):
    aug = iaa.Multiply((min_expose, max_expose))
    return aug(images=batch)
    
# Adds a Gaussina blur (represents out of focus image)
# Mean of 0 is no blur
# Sigma is standard deviation
def aug_machine_focus(batch, mean, sigma):
    aug = iaa.GaussianBlur(sigma=(mean, sigma))
    return aug(images=batch)
