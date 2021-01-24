from PIL import Image
from skimage.transform import resize

import numpy as np

from src.config import TARGET_SIZE


def preprocess(image, cnn_input_size=TARGET_SIZE):
  padded_image = pad_to_square(image)
  resized_image = resize(padded_image, cnn_input_size)
  rgb_image = l_to_rgb(resized_image)
  return rgb_image


def pad_to_square(image):
  dim0, dim1 = image.shape
  
  if dim0 == dim1:
    return image

  if dim0 > dim1:
    pad_width = ((dim0-dim1)//2, (dim0-dim1)//2+(dim0-dim1)%2)
    return np.pad(image, ((0,0), pad_width), "edge")
  else:
    pad_width = ((dim1-dim0)//2, (dim1-dim0)//2)
    return np.pad(image, (pad_width, (0,0)), "edge")


def l_to_rgb(image):
  image = np.stack((image,) * 3, axis=-1) * 255
  image = image.astype("uint8")
  return image
