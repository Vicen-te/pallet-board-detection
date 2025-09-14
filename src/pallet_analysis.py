import cv2
import numpy as np

def connect_fragments(lines: np.ndarray, ksize: tuple, morph_type: int, min_area: int):
    """
    Connect fragmented line segments using morphological operations
    and return only contours larger than a minimum area.
    
    Args:
        lines: Binary image with detected line structures.
        ksize: Kernel size for morphology (width, height).
        morph_type: Morphological operation type (e.g., cv2.MORPH_OPEN, cv2.MORPH_CLOSE).
        min_area: Minimum contour area to keep.
    
    Returns:
        A list of contours that are sufficiently large after connecting fragments.
    """
    # Create a rectangular structuring element for morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    
    # Apply the morphological operation to connect fragmented lines
    connected_lines = cv2.morphologyEx(lines, morph_type, kernel)
    
    # Find external contours in the connected lines image
    contours_fixed, _ = cv2.findContours(connected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours based on minimum area
    filtered_contours = [cnt for cnt in contours_fixed if cv2.contourArea(cnt) >= min_area]
    
    return filtered_contours


def remove_nested_boxes(boxes, overlap_threshold=0.5):
    """
    Remove boxes that are mostly contained inside other boxes.
    
    Args:
        boxes: List of bounding boxes (x, y, w, h).
        overlap_threshold: Fraction of area overlap to consider a box as nested.
    
    Returns:
        Filtered list of boxes with nested boxes removed.
    """
    keep = []
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        area1 = w1*h1
        rect1 = (x1, y1, x1+w1, y1+h1)
        inside = False
        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i == j: 
                continue
            rect2 = (x2, y2, x2+w2, y2+h2)
            
            # Compute intersection rectangle
            ix1 = max(rect1[0], rect2[0])
            iy1 = max(rect1[1], rect2[1])
            ix2 = min(rect1[2], rect2[2])
            iy2 = min(rect1[3], rect2[3])
            iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
            
            # If the intersection is larger than the threshold, mark as nested
            if (iw*ih)/area1 >= overlap_threshold:
                inside = True
                break
        if not inside:
            keep.append((x1, y1, w1, h1))
    return keep


def merge_broken_verticals(boxes, avg_height, y_tolerance=10, x_tolerance=0.25):
    """
    Merge fragmented vertical boxes that likely belong to the same board.
    
    Args:
        boxes: List of vertical bounding boxes (x, y, w, h).
        avg_height: Average height of vertical boards (used for merging criteria).
        y_tolerance: Maximum horizontal distance to consider boxes aligned vertically.
        x_tolerance: Maximum relative height difference to merge boxes.
    
    Returns:
        List of merged vertical boxes.
    """
    merged = []
    used = set()
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        if i in used:
            continue
        center_x1 = x1 + w1//2
        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if j <= i or j in used:
                continue
            center_x2 = x2 + w2//2
            
            # Check if boxes are aligned horizontally
            if abs(center_x1 - center_x2) < y_tolerance:
                combined_height = h1 + h2
                
                # Merge if combined height is close to average board height
                if abs(combined_height - avg_height)/avg_height < x_tolerance:
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = max(x1+w1, x2+w2) - x
                    h = max(y1+h1, y2+h2) - y
                    merged.append((x, y, w, h))
                    used.update([i, j])
                    break
        else:
            # Keep boxes that were not merged
            merged.append((x1, y1, w1, h1))
            used.add(i)
    return merged
