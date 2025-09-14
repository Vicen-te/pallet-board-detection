import cv2
from pathlib import Path
from config import OUTPUT_FOLDER, INPUT_FOLDER
from src import image_utils, edge_detection
from src.horizontal_boards import extract_horizontal_boards
from src.vertical_boards import extract_vertical_boards

# Make sure the output folder exists
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# Iterate over all TIFF images in the input folder
for img_path in Path(INPUT_FOLDER).glob("*.tif"):

    img_name = img_path.stem
    img_folder = Path(OUTPUT_FOLDER) / img_name
    img_folder.mkdir(exist_ok=True)

    print(f"\nProcessing {img_name}")

    # Load the input image and apply normalization
    img = image_utils.load_image(str(img_path))
    norm = image_utils.normalize_image(img)
    cv2.imwrite(str(img_folder / f"{img_name}_normalized.png"), norm)

    # Compute Sobel edge maps for vertical and horizontal structures
    mag_x, mag_y = edge_detection.sobel_edges(norm)

    # Generate rough vertical and horizontal line masks
    vert_mask, hor_mask = edge_detection.extract_lines(norm)

    # Refine the line masks by removing noise and irrelevant edges
    vert_clean = edge_detection.clean_lines(
        vert_mask, kernel_size=(1, 13), edge_mask=mag_x
    )
    hor_clean = edge_detection.clean_lines(
        hor_mask, kernel_size=(13, 3), edge_mask=mag_y
    )

    # Detect horizontal boards and save a labeled image
    labeled_hor, hor_groups = extract_horizontal_boards(hor_clean)
    cv2.imwrite(str(img_folder / f"{img_name}_horizontal.png"), labeled_hor)

    # Detect vertical boards and save a labeled image
    labeled_vert, vert_boxes = extract_vertical_boards(vert_clean)
    cv2.imwrite(str(img_folder / f"{img_name}_vertical.png"), labeled_vert)

    # Create a combined visualization on top of the normalized image
    combined = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    # Draw horizontal board groups in magenta
    for idx, group in enumerate(hor_groups):
        boxes = group['boxes']
        x_coords = [b[0] for b in boxes]
        y_coords = [b[1] for b in boxes]
        x_ends = [b[0] + b[2] for b in boxes]
        y_ends = [b[1] + b[3] for b in boxes]

        # Bounding box enclosing all boards in this group
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_ends), max(y_ends)

        cv2.rectangle(combined, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

    # Draw vertical boards in green
    for i, (x, y, w, h) in enumerate(vert_boxes):
        cv2.rectangle(combined, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the combined result
    cv2.imwrite(str(img_folder / f"{img_name}_combined_boards.png"), combined)

    print(f"Saved results for {img_name} in {OUTPUT_FOLDER}\n")
