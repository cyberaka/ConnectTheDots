import cv2
import numpy as np

def create_outline(
    image_path,
    output_path,
    blur_kernel=(5, 5),
    canny_lower=50,
    canny_upper=150,
    morph_kernel_size=(5, 5),
    contour_thickness=2
):
    """
    Creates an outline (silhouette) of the largest object in the image.
    Saves the outline on a white background.
    """

    # 1. Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Blur to reduce noise (optional)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # 4. Use Canny edge detection
    edges = cv2.Canny(blurred, canny_lower, canny_upper)

    # 5. Morphological closing to connect fragmented edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 6. Find contours
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found. Try adjusting Canny thresholds or morphological parameters.")
        return

    # 7. Select the largest contour (likely your main subject)
    largest_contour = max(contours, key=cv2.contourArea)

    # 8. Create a blank (white) canvas
    height, width = img.shape[:2]
    outline_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 9. Draw the largest contour as the outline on the canvas
    cv2.drawContours(outline_canvas, [largest_contour], -1, (0, 0, 0), contour_thickness)

    # 10. Save the result
    cv2.imwrite(output_path, outline_canvas)
    print(f"Outline saved to {output_path}")

if __name__ == "__main__":
    # Example usage with input.jpeg and outline.jpeg
    create_outline(
        image_path="input.jpeg",
        output_path="outline.jpeg",
        blur_kernel=(5, 5),
        canny_lower=50,
        canny_upper=150,
        morph_kernel_size=(5, 5),
        contour_thickness=2
    )
