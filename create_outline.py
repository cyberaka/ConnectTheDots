import cv2
import numpy as np

def remove_background(input_path, output_no_background_path,
                      blur_kernel=(5, 5), threshold_value=240, morph_kernel_size=(5, 5)):
    """
    Removes the background by isolating the largest object in the image.
    Saves the result to output_no_background_path.
    """
    # 1. Load the image.
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image from {input_path}")
        return None

    # 2. Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Apply Gaussian blur to reduce noise.
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # 4. Threshold the image to separate foreground from background.
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # 5. Morphological closing to fill small holes and gaps.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 6. Find contours in the binary image.
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found! Try adjusting the threshold or morphological parameters.")
        return None

    # 7. Assume the largest contour corresponds to the main object.
    largest_contour = max(contours, key=cv2.contourArea)

    # 8. Create a mask for the largest object.
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # 9. Create an output image with a white background.
    no_bg = np.ones_like(img) * 255
    no_bg[mask == 255] = img[mask == 255]

    # 10. Save the result.
    cv2.imwrite(output_no_background_path, no_bg)
    print(f"Saved image without background to {output_no_background_path}")
    return no_bg

def generate_outline(input_path, output_outline_path,
                     blur_kernel=(5, 5), canny_lower=50, canny_upper=150,
                     morph_kernel_size=(5, 5), contour_thickness=2,
                     simplify_factor=0.02):
    """
    Generates a simplified outline from the image at input_path (expected to be a no-background image).
    The outline is simplified using contour approximation (approxPolyDP) and drawn on a white canvas.
    The result is saved to output_outline_path.

    Parameters:
      - simplify_factor: A multiplier for the contour's arc length to determine the approximation precision.
                         Increase this value to simplify the contour further.
    """
    # 1. Load the image.
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image from {input_path}")
        return

    # 2. Convert to grayscale and apply Gaussian blur.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # 3. Apply Canny edge detection.
    edges = cv2.Canny(blurred, canny_lower, canny_upper)

    # 4. Perform morphological closing to connect broken edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 5. Find contours in the edge map.
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found for outline generation!")
        return

    # 6. Select the largest contour (assumed to be the main object).
    largest_contour = max(contours, key=cv2.contourArea)

    # 7. Simplify the contour using approxPolyDP.
    epsilon = simplify_factor * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 8. Create a blank white canvas.
    height, width = img.shape[:2]
    outline_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 9. Draw the simplified contour as the outline.
    cv2.drawContours(outline_canvas, [approx_contour], -1, (0, 0, 0), contour_thickness)

    # 10. Save the outline image.
    cv2.imwrite(output_outline_path, outline_canvas)
    print(f"Simplified outline saved to {output_outline_path}")

def generate_dotted_from_outline(input_outline_path, output_dotted_path,
                                 dot_radius=4, font_scale=1.0, contour_thickness=2,
                                 min_spacing=60, max_dots=50):
    """
    Generates a dotted image from the outline image at input_outline_path.
    This function computes the contour's total arc length and then places dots
    evenly along the contour so that the distance between successive dots is at least min_spacing.
    Additionally, it limits the total number of dots to max_dots.
    It overlays dots and sequential numbers on the outline and saves the result as output_dotted_path.

    Parameters:
      - min_spacing: The minimal distance (in pixels) between successive dots.
      - max_dots: The maximum number of dots allowed.
    """
    # 1. Load the outline image.
    img = cv2.imread(input_outline_path)
    if img is None:
        print(f"Error: Could not load image from {input_outline_path}")
        return

    # 2. Convert to grayscale and threshold to obtain a binary image.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 3. Find contours in the binary image.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found in the outline image!")
        return

    # 4. Select the largest contour (assumed to be the drawn outline).
    largest_contour = max(contours, key=cv2.contourArea)

    # 5. Flatten the contour points.
    points = largest_contour.squeeze()
    if points.ndim == 1:
        points = np.expand_dims(points, axis=0)

    # 6. Ensure the contour is treated as a closed loop.
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    # 7. Compute cumulative arc length along the contour.
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative = np.concatenate(([0], np.cumsum(distances)))
    total_length = cumulative[-1]

    # 8. Determine the number of dots based on total length, min_spacing, and max_dots.
    computed_dots = int(total_length / min_spacing)
    if computed_dots < 2:
        num_dots = 2
    else:
        num_dots = min(computed_dots, max_dots)

    # 9. Determine the desired arc distances for the dots.
    desired_distances = np.linspace(0, total_length, num=num_dots, endpoint=False)

    # 10. Interpolate to find the exact (x, y) coordinates for each dot.
    dot_positions = []
    for d in desired_distances:
        idx = np.searchsorted(cumulative, d) - 1
        if idx < 0:
            idx = 0
        if idx >= len(points) - 1:
            idx = len(points) - 2
        segment_length = cumulative[idx+1] - cumulative[idx]
        t = 0 if segment_length == 0 else (d - cumulative[idx]) / segment_length
        pt = (1 - t) * points[idx] + t * points[idx+1]
        dot_positions.append(pt.astype(int))

    # 11. Optionally, remove the last dot if it overlaps the first.
    if len(dot_positions) > 1:
        first = np.array(dot_positions[0])
        last = np.array(dot_positions[-1])
        if np.linalg.norm(first - last) < dot_radius * 2:
            dot_positions.pop()

    # 12. Create a copy of the outline image to overlay dots.
    dotted_img = img.copy()
    cv2.drawContours(dotted_img, [largest_contour], -1, (0, 0, 0), contour_thickness)

    # 13. Draw the dots and number them.
    for idx, point in enumerate(dot_positions):
        x, y = int(point[0]), int(point[1])
        cv2.circle(dotted_img, (x, y), dot_radius, (0, 0, 255), -1)
        cv2.putText(dotted_img, str(idx + 1), (x + dot_radius + 2, y - dot_radius - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)

    # 14. Save the final dotted image.
    cv2.imwrite(output_dotted_path, dotted_img)
    print(f"Dotted outline saved to {output_dotted_path}")

if __name__ == "__main__":
    # Define file paths.
    input_file = "input.jpeg"
    no_background_file = "no_background.jpeg"
    outline_file = "outline.jpeg"
    dotted_file = "dotted.jpeg"

    # Step 1: Remove the background and save to no_background.jpeg.
    remove_background(input_file, no_background_file,
                      blur_kernel=(5, 5), threshold_value=240, morph_kernel_size=(5, 5))

    # Step 2: Generate the simplified outline from the no-background image and save to outline.jpeg.
    generate_outline(no_background_file, outline_file,
                     blur_kernel=(5, 5), canny_lower=50, canny_upper=150,
                     morph_kernel_size=(5, 5), contour_thickness=2,
                     simplify_factor=0.01)  # Adjust simplify_factor as needed.

    # Step 3: Generate the dotted image from outline.jpeg and save to dotted.jpeg.
    generate_dotted_from_outline(outline_file, dotted_file,
                                 dot_radius=4, font_scale=1.0, contour_thickness=2,
                                 min_spacing=60, max_dots=50)
