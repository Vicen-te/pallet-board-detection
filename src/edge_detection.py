import cv2
import numpy as np

def sobel_edges(image: np.ndarray, scale: float = 5):
    """
    Compute Sobel gradients in the X and Y directions and return their scaled magnitudes.
    The output highlights vertical (X) and horizontal (Y) edges in the image.
    """
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)   #< Vertical edge detection
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)   #< Horizontal edge detection

    # Scale magnitudes, clip to [0,255], and convert to 8-bit
    magnitude_x = np.uint8(np.clip(np.abs(sobelx) * scale, 0, 255))
    magnitude_y = np.uint8(np.clip(np.abs(sobely) * scale, 0, 255))

    return magnitude_x, magnitude_y 

def threshold_edges(magnitude, threshold: int = 25):
    """
    Convert a gradient magnitude map into a binary mask.
    Pixels above the threshold are set to 255, others to 0.
    """
    _, mask = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
    return mask

def extract_lines(image, vertical_range=(215,255), horizontal_range=(1,215)):
    """
    Vertical structures are extracted by selecting high intensity values.
    Horizontal structures are extracted from the lower intensity range.
    Returns two binary masks: one for vertical lines and one for horizontal lines.
    """
    vertical_lines = cv2.inRange(image, vertical_range[0], vertical_range[1])
    horizontal_lines = cv2.inRange(image, horizontal_range[0], horizontal_range[1])

    return vertical_lines, horizontal_lines

def clean_lines(lines, kernel_size=(1,1), edge_mask=None):
    """
    Refine line masks by applying morphological opening to remove noise.
    If an edge mask is provided, those edges are subtracted to keep only clean structures.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    lines_mor = cv2.morphologyEx(lines, cv2.MORPH_OPEN, kernel, iterations=1)
    
    if edge_mask is not None:
        # Keep only pixels that are in 'lines_mor' but not in 'edge_mask'
        lines_clean = cv2.bitwise_and(lines_mor, cv2.bitwise_not(edge_mask))
        return lines_clean
    
    return lines_mor
