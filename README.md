## Connect-the-Dots Image Generator with Java API Wrapper

This project provides a secure API layer for a connect-the-dots image generation system. The core image processing functionality is implemented in Python, while a Java-based wrapper (built as a Maven project) exposes REST endpoints and provides security. This design allows you to integrate robust Python-powered image processing into a secure, enterprise-ready Java backend.

## Features

- **Python Image Processing:**
    - **Aggressive Background Removal:** Uses k-means clustering on a downscaled image to detect and remove the dominant background color/gradient.
    - **Outline Extraction:** Applies edge detection and contour simplification (using `approxPolyDP`) to generate a simplified outline.
    - **Dot Placement:** Automatically computes dot positions along the outline based on a specified minimum spacing and limits the total number of dots.
    - **Output Images:**
        - `no_background.jpeg`: Image with background removed.
        - `outline.jpeg`: Simplified outline of the main object.
        - `dotted.jpeg`: Outline image overlaid with dots and sequential numbers.
        - `connect-dots.jpeg`: Blank white canvas with only dots and numbers.

- **Java API Wrapper:**
    - Exposes REST endpoints to invoke the Python image processing scripts.
    - Provides additional security and integration into larger applications.
    - Built using Maven for dependency management and build automation.

## Repository Structure

```
pom.xml                   - Maven build file for the Java project
src/
  main/
    java/                 - Java source code for REST endpoints and API wrapper
    resources/            - Configuration files and Python scripts
python/
  requirements.txt        - Python dependencies
  connect_dots.py         - Main Python script (image processing)
  input.jpeg              - Sample input image
  no_background.jpeg      - Output: Image with background removed
  outline.jpeg            - Output: Simplified outline of the main object
  dotted.jpeg             - Output: Outline with dots and sequential numbers
  connect-dots.jpeg       - Output: Blank canvas with only dots and numbers
README.txt                - This file
```

## Prerequisites

- **Java:** JDK 8 or higher
- **Maven:** For building the Java wrapper
- **Python:** Version 3.x
- **Git**

## Python Environment Setup

1. Navigate to the python directory:
   ```sh
   cd python
   ```
2. Create a Python virtual environment:
   ```sh
   python3 -m venv venv
   ```
3. Activate the virtual environment:
    - On Linux/macOS:
      ```sh
      source venv/bin/activate
      ```
    - On Windows:
      ```sh
      venv\Scripts\activate
      ```
4. Install the Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Maven Project Setup and Build

1. Navigate to the repository root (where `pom.xml` is located).

2. Build the Maven project:
   ```sh
   mvn clean install
   ```

3. Run the Java application:
   ```sh
   mvn spring-boot:run
   ```
   (This will expose secure REST endpoints that invoke the Python scripts.)

## Usage

- **REST Endpoints:**
  When the Java application is running, it will expose REST endpoints (for example, `POST /api/connect-dots`) that trigger the Python image processing tasks. Refer to the Java source code and API documentation for details.

- **Running Python Script Directly:**
  You can also run the Python script independently for testing:
  ```sh
  cd python
  python connect_dots.py
  ```
  This will process `input.jpeg` and produce the output images in the `python` directory.

## Customization

- **Python Parameters:**
    - Background Removal: Adjust parameters such as `downscale_factor`, number of clusters (`k`), and attempts in the k-means based background removal function.
    - Outline Generation: Fine-tune parameters such as `blur_kernel`, `canny_lower`, `canny_upper`, and `simplify_factor` to adjust the level of detail.
    - Dot Placement: Modify `min_spacing`, `max_dots`, `dot_radius`, and `font_scale` as needed.

- **Java API Customization:**
  The Java code in `src/main/java` can be modified to change endpoint paths, security settings, or how the Python scripts are invoked.

## License

MIT License

## Acknowledgements

- OpenCV & NumPy: For robust image processing.
- Spring Boot & Java: For providing the secure API framework.
- Special thanks to the open source community for their contributions.