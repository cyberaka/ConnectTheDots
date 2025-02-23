import cv2
import numpy as np

def generate_dot_to_dot(image_path, output_path,
                         blur_kernel=(5, 5), canny_thresh=(50, 150),
                         epsilon_ratio=0.01, dot_radius=5, font_scale=0.5, line_thickness=1):
    # 1. Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image from", image_path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    edges = cv2.Canny(blurred, canny_thresh[0], canny_thresh[1])

    # 2. Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found in the image!")
        return

    # For demonstration, choose the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 3. Simplify the contour to reduce the number of points
    epsilon = epsilon_ratio * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 4. Create a blank white canvas (or you can use a copy of the original)
    output = np.ones_like(img) * 255

    # Draw dots and numbers
    for idx, point in enumerate(approx):
        x, y = point[0]
        # Draw a circle for each dot
        cv2.circle(output, (x, y), dot_radius, (0, 0, 255), -1)  # red dot
        # Put a sequential number next to the dot
        cv2.putText(output, str(idx + 1), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    # Optionally, draw lines connecting the dots
    for i in range(len(approx) - 1):
        x1, y1 = approx[i][0]
        x2, y2 = approx[i + 1][0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 0), line_thickness)
    # Optionally, connect the last point to the first if the contour is closed:
    if len(approx) > 2:
        x1, y1 = approx[-1][0]
        x2, y2 = approx[0][0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 0), line_thickness)

    # 5. Save the resulting image
    cv2.imwrite(output_path, output)
    print("Connect-the-dots image saved to", output_path)

# Example usage:
if __name__ == '__main__':
    input_image = 'input.jpg'       # Replace with your image path
    output_image = 'dot_to_dot.jpg' # Desired output image file name
    generate_dot_to_dot(input_image, output_image)
