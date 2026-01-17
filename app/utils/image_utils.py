import numpy as np
from PIL import Image
from io import BytesIO

def load_image_from_bytes(image_bytes: bytes) -> np.array:
    """
    Load an image from bytes and convert it to a numpy array.

    :param image_bytes: Image in bytes
    :type image_bytes: bytes
    :return: Numpy array representation of the image
    :rtype: np.array
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(image)