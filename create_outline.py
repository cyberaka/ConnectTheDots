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

    # 4. Threshold the image to separate foreground from background
    #    Using THRESH_BINARY_INV assumes the background is light and the object is darker.
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # 5. Morphological closing to fill small holes and gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 6. Find contours from the binary image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found! Try adjusting the threshold or morphological parameters.")
        return None

    # 7. Assume the largest contour corresponds to the main object (e.g., the hen)
    largest_contour = max(contours, key=cv2.contourArea)

    # 8. Create a mask for the largest object
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # 9. Create an output image with a white background
    no_bg = np.ones_like(img) * 255
    no_bg[mask == 255] = img[mask == 255]

    # 10. Save the result
    cv2.imwrite(output_no_background_path, no_bg)
    print(f"Saved image without background to {output_no_background_path}")
    return no_bg

def generate_outline(input_path, output_outline_path,
                     blur_kernel=(5, 5), canny_lower=50, canny_upper=150,
                     morph_kernel_size=(5, 5), contour_thickness=2):
    """
    Generates an outline from the image at input_path.
    The outline is saved on a white background to output_outline_path.
    """
    # 1. Load the image (should be the no_background image)
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image from {input_path}")
        return

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Blur the image to reduce noise
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # 4. Apply Canny edge detection
    edges = cv2.Canny(blurred, canny_lower, canny_upper)

    # 5. Use morphological closing to connect any broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 6. Find contours on the edge map
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found in outline generation!")
        return

    # 7. Select the largest contour (the main object outline)
    largest_contour = max(contours, key=cv2.contourArea)

    # 8. Create a blank white canvas
    height, width = img.shape[:2]
    outline_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 9. Draw the contour as the outline
    cv2.drawContours(outline_canvas, [largest_contour], -1, (0, 0, 0), contour_thickness)

    # 10. Save the outline image
    cv2.imwrite(output_outline_path, outline_canvas)
    print(f"Outline saved to {output_outline_path}")

if __name__ == "__main__":
    # File paths
    input_file = "input.jpeg"
    no_background_file = "no_background.jpeg"
    outline_file = "outline.jpeg"

    # Step 1: Remove the background and save to no_background.jpeg
    remove_background(input_file, no_background_file,
                      blur_kernel=(5, 5), threshold_value=240, morph_kernel_size=(5, 5))

    # Step 2: Generate the outline from no_background.jpeg and save to outline.jpeg
    generate_outline(no_background_file, outline_file,
                     blur_kernel=(5, 5), canny_lower=50, canny_upper=150,
                     morph_kernel_size=(5, 5), contour_thickness=2)
