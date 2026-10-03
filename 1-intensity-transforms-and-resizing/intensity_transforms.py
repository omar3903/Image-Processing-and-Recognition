"""
Author: Omar Abdeen
CP 467 Assignment 1
9/30/2024
"""
import numpy as np
import cv2
import os

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")
    
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV failed to load the image {image_path}. It may not be a supported format.")
    
    return image

###################    TASK 1    ###################

def nearest_neighbor_interpolation(image, factor):
    height, width = image.shape[:2]
    new_height = int(height * factor) 
    new_width = int(width * factor)
    
    if len(image.shape) == 2:
        resize = np.zeros((new_height, new_width), dtype=image.dtype)
    else:
        resize = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    
    for i in range(new_height):
        for j in range(new_width):
            resize[i, j] = image[int(i/factor), int(j/factor)]
    
    return resize

def bilinear_interpolation(image, factor):
    height, width = image.shape[:2]
    new_height = int(height * factor) 
    new_width = int(width * factor)
    
    if len(image.shape) == 2:
        resize = np.zeros((new_height, new_width), dtype=image.dtype)
    else:
        resize = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    
    for i in range(new_height):
        for j in range(new_width):
            x, y = i / factor, j / factor
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, height - 1), min(y1 + 1, width - 1)
            
            dx, dy = x - x1, y - y1
            
            if len(image.shape) == 2:
                resize[i, j] = int((1 - dx) * (1 - dy) * image[x1, y1] +
                                    dx * (1 - dy) * image[x2, y1] +
                                    (1 - dx) * dy * image[x1, y2] +
                                    dx * dy * image[x2, y2])
            else:
                for c in range(image.shape[2]):
                    resize[i, j, c] = int((1 - dx) * (1 - dy) * image[x1, y1, c] +
                                           dx * (1 - dy) * image[x2, y1, c] +
                                           (1 - dx) * dy * image[x1, y2, c] +
                                           dx * dy * image[x2, y2, c])
    
    return resize

# File path
curr_dir = os.getcwd()
image_path = os.path.join(curr_dir, "images", "lena.tif")

# Load the image
image = load_image(image_path)

print(f"Successfully loaded the image from {image_path}")

# 1. Nearest neighbor interpolation implementation from scratch
first_resize_nn = nearest_neighbor_interpolation(image, 0.5)
original_resize_nn = nearest_neighbor_interpolation(first_resize_nn, 2)
cv2.imwrite('lena_nearest_scratch.tif', original_resize_nn)

# 2. Nearest neighbor interpolation using OpenCV
first_resize_nn_cv = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
original_resize_nn_cv = cv2.resize(first_resize_nn_cv, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
cv2.imwrite('lena_nearest_cv.tif', original_resize_nn_cv)

# 3. Bilinear interpolation implementation from scratch
first_resize_bl = bilinear_interpolation(image, 0.5)
original_resize_bl = bilinear_interpolation(first_resize_bl, 2)
cv2.imwrite('lena_bilinear_scratch.tif', original_resize_bl)

# 4. Bilinear interpolation using OpenCV
first_resize_bl_cv = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
original_resize_bl_cv = cv2.resize(first_resize_bl_cv, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
cv2.imwrite('lena_bilinear_cv.tif', original_resize_bl_cv)

# 5. Bicubic interpolation using OpenCV
first_resize_bc_cv = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
original_resize_bc_cv = cv2.resize(first_resize_bc_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imwrite('lena_bicubic_cv.tif', original_resize_bc_cv)

print("Task 1 completed successfully.")

###################    END OF TASK 1    ###################

###################    TASK 2    ###################

def power_law_transform(image, gamma):
    # Normalize the image to [0, 1] range
    normalized = image / 255.0
    # Apply power-law transformation
    transformed = np.power(normalized, gamma)
    # factor back to [0, 255] range and convert to uint8
    return np.uint8(transformed * 255)

def negative_image(image):
    return 255 - image

def bit_plane_slice(image, bit):
    # Extract the specified bit plane
    return np.uint8((image & (1 << (bit - 1))) != 0) * 255

# File path
curr_dir = os.getcwd()
image_path = os.path.join(curr_dir, "images", "cameraman.tif")

# Load the image
image = load_image(image_path)

print(f"Successfully loaded image from {image_path}")

# 1. Negative of the image
negative = negative_image(image)
cv2.imwrite('cameraman_negative.tif', negative)

# 2. Power-law transformation
gamma_values = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]
for gamma in gamma_values:
    power_law = power_law_transform(image, gamma)
    cv2.imwrite(f'cameraman_power_gamma_{gamma:.1f}.tif', power_law)

# After manual choosing of the best gamma image, we concluded that gamma value of 0.8 gave us the best picture.

# 3. bit-plane slicing
for bit in range(1, 9):
    bit_plane = bit_plane_slice(image, bit)
    cv2.imwrite(f'cameraman_b{bit}.tif', bit_plane)

print("Task 2 completed successfully.")

###################    END OF TASK 2    ###################

###################    TASK 3    ###################

def compute_cdf(histogram):
    cdf = histogram.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    return cdf_normalized

def compute_histogram(image):
    histogram = np.zeros(256, dtype=int)
    for pixel in image.flat:
        histogram[pixel] += 1
    return histogram

def histogram_equalization(image):
    histogram = compute_histogram(image)
    cdf = compute_cdf(histogram)
    
    equalized = np.interp(image.flatten(), range(256), cdf)
    return equalized.reshape(image.shape).astype(np.uint8)

def histogram_specification(source_image, target_image):
    source_hist = compute_histogram(source_image)
    source_cdf = compute_cdf(source_hist)
    target_hist = compute_histogram(target_image)
    target_cdf = compute_cdf(target_hist)
    
    lookup_table = np.zeros(256, dtype=np.uint8)
    j = 0
    # Ensure valid value
    for i in range(256):
        while j < 256 and target_cdf[j] <= source_cdf[i]:
            j += 1
        lookup_table[i] = max(0, min(255, j)) 
    
    # Apply the lookup table
    specified = lookup_table[source_image]
    return specified

# Image paths
curr_dir = os.getcwd()
einstein_path = os.path.join(curr_dir, "images", "einstein.tif")
xray1_path = os.path.join(curr_dir, "images", "chest_x-ray1.jpeg")
xray2_path = os.path.join(curr_dir, "images", "chest_x-ray2.jpeg")

# 1. Histogram Equalization on Einstein image
einstein_image = load_image(einstein_path)
einstein_equalized = histogram_equalization(einstein_image)
cv2.imwrite('einstein_equalized.tif', einstein_equalized)

print("Histogram Equalization completed for Einstein image.")

# 2. Histogram Specification on chest X-ray images
xray1_image = load_image(xray1_path)
xray2_image = load_image(xray2_path)
xray3_image = histogram_specification(xray1_image, xray2_image)
cv2.imwrite('chest_x-ray3.tif', xray3_image)


print("Task 3 completed successfully.")

###################    END OF TASK 3    ###################

print("Assignment completed successfully")