import cv2
import numpy as np
from src.pallet_analysis import merge_broken_verticals, remove_nested_boxes, connect_fragments

def extract_vertical_boards(vertical_lines_no_edges):
    """
    Extract vertical boards, adjust large boxes, merge fragments,
    fill missing gaps, and return a labeled image and list of boxes.
    
    Args:
        vertical_lines_no_edges: Binary image with detected vertical lines, cleaned of edges.
    
    Returns:
        labeled_image: BGR image with rectangles and labels for each detected board.
        all_boxes_sorted: List of final bounding boxes (x, y, w, h) for vertical boards.
    """

    # Filter contours by minimum area and connect fragmented vertical lines
    min_area = 100
    filtered_contours_vertical = connect_fragments(vertical_lines_no_edges, (5, 25), cv2.MORPH_OPEN, min_area) 

    # Convert contours to bounding boxes and compute their areas
    boxes = [cv2.boundingRect(cnt) for cnt in filtered_contours_vertical]
    areas = [w*h for (_,_,w,h) in boxes]

    avg_area = np.mean(areas)
    avg_width = np.mean([w for (_,_,w,_) in boxes])
    avg_height = np.mean([h for (_,_,_,h) in boxes])

    # Divide large boxes that are significantly bigger than average
    adjusted_boxes = []
    scale = 1.5
    
    for (x,y,w,h), area in zip(boxes, areas):
        if area > scale*avg_area:
            if h > w and h > 10:  # Split vertically
                w2 = w // 2
                adjusted_boxes.extend([(x,y,w2,h),(x+w2,y,w-w2,h)])
            elif w >= h and w > 10:  # Split horizontally
                h2 = h // 2
                adjusted_boxes.extend([(x,y,w,h2),(x,y+h2,w,h-h2)])
            else:
                adjusted_boxes.append((x,y,w,h))
        else:
            adjusted_boxes.append((x,y,w,h))

    # Remove boxes fully inside other boxes
    adjusted_boxes = remove_nested_boxes(adjusted_boxes, overlap_threshold=0.5)

    # Merge fragmented vertical boxes that should belong to the same board
    adjusted_boxes = merge_broken_verticals(adjusted_boxes, avg_height, y_tolerance=50, x_tolerance=0.3)

    # Print statistics of detected boxes
    print("\nAdjusted pallet areas:")
    for i,(x,y,w,h) in enumerate(adjusted_boxes, start=1):
        print(f"Pallet {i}: area={w*h}, width={w}, height={h}")
    print(f"Average area={avg_area:.2f}, width={avg_width:.2f}, height={avg_height:.2f}")
    print(f"Total pallets after cleaning: {len(adjusted_boxes)}")

    # Fill gaps between boards by estimating missing pallets
    vertical_boxes_sorted = sorted(adjusted_boxes, key=lambda b: b[0], reverse=True)
    average_width = np.mean([w for (_,_,w,_) in vertical_boxes_sorted])
    extended_boxes = [vertical_boxes_sorted[0]]

    for i in range(len(vertical_boxes_sorted)-1):
        x1,y1,w1,h1 = vertical_boxes_sorted[i]
        x2,y2,w2,h2 = vertical_boxes_sorted[i+1]
        
        current_left = x1
        next_right = x2 + w2
        separation = current_left - next_right

        print(f"x1: {x1}, x2: {x2}, separation: {separation:.2f}")

        if separation <= 0:
            extended_boxes.append(vertical_boxes_sorted[i+1])
            continue

        estimated_pallets = separation / average_width
        estimated_full_pallets = int(estimated_pallets)

        print(f"Between pallet {i + 1} and {i + 2}: separation = {separation:.2f}, estimated = {estimated_pallets:.2f}")

        if estimated_full_pallets >= 1:
            print(f"👉 Can add {estimated_full_pallets} pallets between {i + 1} and {i + 2}")
            for n in range(estimated_full_pallets):
                gap_x = current_left - (n+1)*average_width
                virtual_box = (int(gap_x), y1, int(average_width), h1)
                extended_boxes.append(virtual_box)

        extended_boxes.append(vertical_boxes_sorted[i+1])

    # Sort all boxes from right to left
    all_boxes_sorted = sorted(extended_boxes, key=lambda b: b[0], reverse=True)

    # Generate a labeled image (optional visualization)
    labeled_image = cv2.cvtColor(vertical_lines_no_edges, cv2.COLOR_GRAY2BGR)

    for i,(x,y,w,h) in enumerate(all_boxes_sorted):
        font_scale = max(w,h)/500
        font_scale = max(font_scale,1)

        text = str(i+1)
        text_size,_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        text_w, text_h = text_size
        text_x = x + (w-text_w)//2
        text_y = y + (h+text_h)//2

        # Color green for real boxes, orange for virtual boxes
        color = (255,125,0) if (x,y,w,h) not in vertical_boxes_sorted else (0,255,0)
        
        cv2.rectangle(labeled_image, (x,y), (x+w,y+h), color, 2)
        cv2.putText(labeled_image, text, (text_x,text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    print(f"Total boards: {len(all_boxes_sorted)}")

    return labeled_image, all_boxes_sorted
