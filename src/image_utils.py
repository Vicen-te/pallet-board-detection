import cv2
import numpy as np

def load_image(path: str):
    """
    Load a 16-bit TIFF image from the specified path.
    
    Args:
        path: Path to the image file.
    
    Returns:
        Loaded image as a NumPy array.
    """
    image = cv2.imread(path, -1)  # Use -1 to preserve original bit depth (16-bit)
    return image


def normalize_image(image: np.ndarray):
    """
    Normalize an image to the 0-255 range and convert to 8-bit.
    
    Args:
        image: Input image (any depth).
    
    Returns:
        8-bit normalized image.
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def save_image(image: np.ndarray, path: str):
    """
    Save an image to disk.
    
    Args:
        image: Image to save.
        path: Path where the image will be saved.
    """
    cv2.imwrite(path, image)
