"""
Author: Omar Abdeen
CP 467 Assignment 3
11/04/2024
"""
import numpy as np
import cv2

def detect_circles(image, min_radius, max_radius):
    """
    Detect circles in a grayscale image.
    """
    # Canny Edge
    edges = cv2.Canny(image, 20, 60)
    
    # Apply 'Hough Transform - Circle' with stricter parameters
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1,minDist=max_radius, param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
    
    return edges, circles

def preprocess_image(image):

    # Convert to grayscale if the image is colored
    if len(image.shape) != 3:
        gray_image = image.copy()
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur with kernel size 7x7
    gaus_blur = cv2.GaussianBlur(gray_image, (7, 7), 0)
    
    return gaus_blur

def select_best_circle(circles, image_center):
    """
    Select the best circle based on proximity to image center and circle properties.
    Returns the single best circle or None if no circles found.
    """
    if circles is None:
        return None
    
    circles = circles[0]
    if len(circles) == 0:
        return None
    
    scores = []
    for circle in circles:
        center_x, center_y, radius = circle
        distance_to_center = np.sqrt((center_x - image_center[0])**2 + 
                                   (center_y - image_center[1])**2)
        
        score = distance_to_center
        scores.append(score)
    
    # Select the circle with the lowest score (least distance to center)
    best_circle = circles[np.argmin(scores)]
    return np.array([[best_circle]])

def process_eye_image(image_path):
    """
    Process a single eye image to detect pupil, iris, and sclera boundaries.
    """
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return None, None
    
    height, width = image.shape[:2]
    image_center = (width // 2, height // 2)
    
    # Copy for manipulation
    result_image = image.copy()
    
    # Preprocess the image
    preprocessed = preprocess_image(image)
    
    # Pupil
    pupil_edges, pupil_circles = detect_circles(preprocessed,min_radius=int(width * 0.05),max_radius=int(width * 0.2))
    
    # Iris
    iris_edges, iris_circles = detect_circles(preprocessed,min_radius=int(width * 0.2),max_radius=int(width * 0.4))
    
    
    # Select the best circles for the segmented images
    pupil_circles = select_best_circle(pupil_circles, image_center)
    iris_circles = select_best_circle(iris_circles, image_center)
    
    # Combine all the edge maps
    edge_map = cv2.addWeighted(pupil_edges, 1, iris_edges, 1, 0)
    
    # Draw the pupil circle in red
    if pupil_circles is not None:
        pupil_circles = np.uint16(np.around(pupil_circles))
        for i in pupil_circles[0, :]:
            cv2.circle(result_image, (i[0], i[1]), i[2], (0, 0, 255), 2)
    
    # Draw the iris circle in blue
    if iris_circles is not None:
        iris_circles = np.uint16(np.around(iris_circles))
        for i in iris_circles[0, :]:
            cv2.circle(result_image, (i[0], i[1]), i[2], (255, 0, 0), 2)
    
    return result_image, edge_map

# Process iris1.tif through iris5.tif
for i in range(1, 6):
    image_path = f"iris{i}.tif"

    output_image, edge_map = process_eye_image(image_path)
        
    # Save results if processing was successful
    if output_image is not None and edge_map is not None:
        cv2.imwrite(f"iris{i}_output.tif", output_image)
        cv2.imwrite(f"iris{i}_edge.tif", edge_map)
        print(f"Completed processing iris{i}.tif")
    else:
        print(f"Failure to process {image_path}")