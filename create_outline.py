import cv2
import numpy as np

def remove_background(input_path, output_no_background_path,
                      blur_kernel=(5, 5), threshold_value=240, morph_kernel_size=(5, 5)):
    """
    Removes the background by isolating the largest object in the image.
    Saves the result to output_no_background_path.
    """
    # 1. Load the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image from {input_path}")
        return None

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # 4. Threshold the image to separate foreground from background.
    #    THRESH_BINARY_INV assumes a light background.
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
    The outline is simplified using contour approximation and drawn on a white canvas.
    The result is saved to output_outline_path.

    Parameters:
      - simplify_factor: A multiplier for the contour's arc length to determine the approximation precision.
                         Increase this value to simplify the contour further.
    """
    # 1. Load the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image from {input_path}")
        return

    # 2. Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # 3. Apply Canny edge detection
    edges = cv2.Canny(blurred, canny_lower, canny_upper)

    # 4. Perform morphological closing to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 5. Find contours in the edge map
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found for outline generation!")
        return

    # 6. Select the largest contour (assumed to be the main object)
    largest_contour = max(contours, key=cv2.contourArea)

    # 7. Simplify the contour using approxPolyDP
    epsilon = simplify_factor * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 8. Create a blank white canvas
    height, width = img.shape[:2]
    outline_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 9. Draw the simplified contour as the outline
    cv2.drawContours(outline_canvas, [approx_contour], -1, (0, 0, 0), contour_thickness)

    # 10. Save the outline image
    cv2.imwrite(output_outline_path, outline_canvas)
    print(f"Simplified outline saved to {output_outline_path}")

def generate_dotted_from_outline(input_outline_path, output_dotted_path,
                                 max_dots=50, dot_radius=4, font_scale=1.0,
                                 contour_thickness=2):
    """
    Generates a dotted image from the outline image at input_outline_path.
    It detects the contour from the outline image, samples up to max_dots points,
    draws dots and numbers them, and saves the result as output_dotted_path.
    """
    # 1. Load the outline image
    img = cv2.imread(input_outline_path)
    if img is None:
        print(f"Error: Could not load image from {input_outline_path}")
        return

    # 2. Convert to grayscale and threshold to get binary outline.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Since the outline is black on white, threshold at 127 works well.
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 3. Find contours in the thresholded image.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found in the outline image!")
        return

    # 4. Select the largest contour (the drawn outline).
    largest_contour = max(contours, key=cv2.contourArea)

    # 5. Flatten the contour points.
    points = largest_contour.squeeze()
    if points.ndim == 1:  # Ensure we have at least a 2D array of points.
        points = np.expand_dims(points, axis=0)

    num_points = len(points)
    # 6. Sample evenly if more than max_dots points.
    if num_points > max_dots:
        indices = np.linspace(0, num_points - 1, max_dots, dtype=int)
        sampled_points = points[indices]
    else:
        sampled_points = points

    # 7. Check if the first and last points are nearly identical.
    if len(sampled_points) > 1:
        first_point = np.array(sampled_points[0], dtype=float)
        last_point = np.array(sampled_points[-1], dtype=float)
        dist = np.linalg.norm(first_point - last_point)
        if dist < dot_radius * 2:
            sampled_points = sampled_points[:-1]

    # 8. Create a copy of the outline image to overlay dots.
    dotted_img = img.copy()

    # 9. Optionally, redraw the outline to ensure it's visible.
    cv2.drawContours(dotted_img, [largest_contour], -1, (0, 0, 0), contour_thickness)

    # 10. Draw dots and number them with increased font size and thickness.
    for idx, point in enumerate(sampled_points):
        x, y = int(point[0]), int(point[1])
        cv2.circle(dotted_img, (x, y), dot_radius, (0, 0, 255), -1)
        cv2.putText(dotted_img, str(idx + 1), (x + dot_radius + 2, y - dot_radius - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)

    # 11. Save the final dotted image.
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
                     simplify_factor=0.002)

    # Step 3: Generate the dotted image from outline.jpeg and save to dotted.jpeg.
    generate_dotted_from_outline(outline_file, dotted_file,
                                 max_dots=30, dot_radius=4, font_scale=1.0, contour_thickness=2)
