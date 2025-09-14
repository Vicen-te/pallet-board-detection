import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_horizontal_boards(horizontal_lines_no_edges):
    """
    Extract horizontal boards from a preprocessed mask, group them into horizontal lines,
    and return both a labeled visualization and the list of groups.
    """

    # Find contours in the horizontal line mask
    contours_horizontal, _ = cv2.findContours(horizontal_lines_no_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours into bounding boxes
    horizontal_boxes = [cv2.boundingRect(cnt) for cnt in contours_horizontal]

    # Filter boxes based on area relative to the average
    areas = [w*h for (x,y,w,h) in horizontal_boxes]
    mean_area = np.mean(areas)
    area_threshold_ratio = 0.1
    filtered_boxes = [box for box in horizontal_boxes if box[2]*box[3] >= mean_area*area_threshold_ratio]

    # Sort boxes from bottom to top using the y-coordinate
    horizontal_boxes_sorted = sorted(filtered_boxes, key=lambda b: b[1], reverse=True)

    # Group boxes that belong to the same horizontal line based on vertical proximity
    line_groups = []
    y_threshold = 75
    
    for box in horizontal_boxes_sorted:
        x, y, w, h = box
        center_y = y + h//2
        found_group = False
        for group in line_groups:
            if abs(center_y - group['y_mean']) < y_threshold:
                group['boxes'].append(box)
                all_ys = [b[1] + b[3]//2 for b in group['boxes']]
                group['y_mean'] = int(np.mean(all_ys))  # update average vertical position
                found_group = True
                break
        if not found_group:
            line_groups.append({'y_mean': center_y, 'boxes':[box]})

    # Estimate the average number of pallets per line
    counts = [len(g['boxes']) for g in line_groups]
    mean_count = np.mean(counts)

    # Sort line groups from bottom to top
    line_groups_sorted = sorted(line_groups, key=lambda g: g['y_mean'], reverse=True)

    # Keep only groups with a number of pallets close to the average (>=80%)
    valid_groups = [g for g in line_groups_sorted if len(g['boxes']) >= 0.8*mean_count]

    # Draw rectangles and labels for each valid horizontal group
    labeled_image = cv2.cvtColor(horizontal_lines_no_edges, cv2.COLOR_GRAY2BGR)
    for idx, group in enumerate(valid_groups):
        boxes = group['boxes']
        x_coords = [b[0] for b in boxes]
        y_coords = [b[1] for b in boxes]
        x_ends = [b[0] + b[2] for b in boxes]
        y_ends = [b[1] + b[3] for b in boxes]

        # Bounding box enclosing all boards in this horizontal group
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_ends), max(y_ends)

        cv2.rectangle(labeled_image, (x_min,y_min), (x_max,y_max), (0,255,0), 2)

        # Draw the group index number at the center of the bounding box
        text = str(idx+1)
        font_scale = 1.5
        thickness = 3
        text_size,_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_w, text_h = text_size
        center_x = (x_min+x_max)//2
        center_y = (y_min+y_max)//2
        text_x = center_x - text_w//2
        text_y = center_y + text_h//2
        cv2.putText(labeled_image, text, (text_x,text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)

        print(f"Line {idx+1}: y≈{group['y_mean']}, pallets={len(boxes)}")

    print(f"Total horizontal boards: {len(valid_groups)}")

    return labeled_image, valid_groups
