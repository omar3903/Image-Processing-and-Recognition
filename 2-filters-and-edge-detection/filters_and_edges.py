"""
Author: Omar Abdeen
CP 467 Assignment 2
10/14/2024
"""
import numpy as np
import cv2
# Bonus task
import time
# Figure(s) for the report, uncomment import statement and matplotlib code blocks below to view figures
import matplotlib.pyplot as plt

# Save the image properly to get a working output
def save_image(image, filename):
    # Correct range (0-255) and data type (uint8) for working output
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    cv2.imwrite(filename, image)

def convolution2D(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - m + 1
    result = np.zeros((y,x))
    for i in range(y):
        for j in range(x):
            result[i,j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return result

def gaussian_kernel(size, sigma):
    # Create the grid for the coordinates
    coord_range = np.arange(-size // 2 + 1, size // 2 + 1)
    x_coord, y_coord = np.meshgrid(coord_range, coord_range)
    
    # Compute the Gaussian kernel
    # Common practice to remove the costant 1/ (2*np.pi*sigma**2)
    kernel = np.exp(-(x_coord**2 + y_coord**2) / (2 * sigma**2))
    
    # Normalizing the kernel
    return kernel / np.sum(kernel)

# Load the image
img = cv2.imread('lena.tif', cv2.IMREAD_GRAYSCALE)

# Task 1: Implement from scratch
print("Task 1: Custom Implementations")

# A) Averaging smoothing filter
start_time = time.time()
avg_kernel = np.ones((3,3)) / 9
avg_image = convolution2D(img, avg_kernel)
save_image(avg_image, 't1a.tif')
print(f"Averaging filter time: {time.time() - start_time:.3f} seconds")

# B) Gaussian smoothing filter
start_time = time.time()
gaussian_kernel = gaussian_kernel(7, 1)
gaussian_image = convolution2D(img, gaussian_kernel)
save_image(gaussian_image, 't1b.tif')
print(f"gaussianian filter time: {time.time() - start_time:.3f} seconds")

# C) Sobel sharpening filter
start_time = time.time()
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_x_result = convolution2D(img, sobel_x)
sobel_y_result = convolution2D(img, sobel_y)
sobel_image = np.sqrt(sobel_x_result**2 + sobel_y_result**2)
print(f"Sobel filter time: {time.time() - start_time:.3f} seconds")
save_image(sobel_image, 't1c.tif')

print("=============================        Task 1 Completed Successfully       ===============================")

# Task 2: OpenCV built-in functions
print("\nTask 2: OpenCV Built-in Functions")

# A) Averaging smoothing filter
start_time = time.time()
opencv_avg_image = cv2.blur(img, (3,3))
print(f"OpenCV Averaging filter time: {time.time() - start_time:.3f} seconds")
save_image(opencv_avg_image, 't2a.tif')

# B) Gaussian smoothing filter
start_time = time.time()
opencv_gaussian_image = cv2.GaussianBlur(img, (7,7), 1)
print(f"OpenCV gaussianian filter time: {time.time() - start_time:.3f} seconds")
save_image(opencv_gaussian_image, 't2b.tif')

# C) Sobel sharpening filter
start_time = time.time()
opencv_sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
opencv_sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
opencv_sobel_image = np.sqrt(opencv_sobel_x**2 + opencv_sobel_y**2)
print(f"OpenCV Sobel filter time: {time.time() - start_time:.3f} seconds")
save_image(opencv_sobel_image, 't2c.tif')

"""
Unblock the bottom block of code to see the figure showed in the report
"""

# plt.figure(figsize=(25, 12))
# images = [img, avg_image, gaussian_image, sobel_image, 
#           opencv_avg_image, opencv_gaussian_image, opencv_sobel_image]
# titles = ['Original', 'Averaging Smoothing Filter From Scratch', 'Gaussian Smoothing Filter From Scratch', 'Sobel Sharpening Filter From Scratch',
#           'OpenCV Averaging Smoothing Filter ', 'OpenCV Gaussian', 'OpenCV Sobel']


# # Top row
# for i in range(4):
#     plt.subplot2grid((2, 4), (0, i))
#     plt.imshow(images[i], cmap='gray')
#     plt.title(titles[i])
#     plt.axis('off')

# # Bottom row with one image size to the left
# for i in range(3):
#     plt.subplot2grid((2, 4), (1, i+1))
#     plt.imshow(images[i+4], cmap='gray')
#     plt.title(titles[i+4])
#     plt.axis('off')

# plt.tight_layout()
# plt.show()


print("=============================        Task 2 Completed Successfully       ===============================")

"""
=================================       Tasks 1 & 2 are Done!       ==================================
"""



def marr_hildreth_edge_detector(image, sigma, threshold):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    zero_crossings = np.zeros_like(laplacian, dtype=np.uint8)
    zero_crossings[(laplacian > threshold) & (np.roll(laplacian, 1, axis=0) < -threshold)] = 255
    zero_crossings[(laplacian < -threshold) & (np.roll(laplacian, 1, axis=0) > threshold)] = 255
    zero_crossings[(laplacian > threshold) & (np.roll(laplacian, 1, axis=1) < -threshold)] = 255
    zero_crossings[(laplacian < -threshold) & (np.roll(laplacian, 1, axis=1) > threshold)] = 255
    return zero_crossings


# Custom parameters for Marr-Hildreth
# Changing Sigma then changing Threshold
mh_params = [(1, 1), (2, 1), (3, 1), (2, 0.5), (2, 1.5), (2, 2)]

# Custom parameters for Canny
# Changing lower threshold then changing the higher threshold
canny_params = [(50, 150), (100, 150), (150, 150), (100, 100), (100, 200), (100, 250)]

# Lists to store results and times
mh_results = []
mh_times = []
canny_results = []
canny_times = []

# Marr-Hildreth edge detection
for i, (sigma, threshold) in enumerate(mh_params):
    start_time = time.time()
    mh_edges = marr_hildreth_edge_detector(img, sigma, threshold)
    end_time = time.time()
    mh_results.append(mh_edges)
    mh_times.append(end_time - start_time)
    save_image(mh_edges, f't3a{i+1}.tif')
# Canny edge detection
for i, (low_threshold, high_threshold) in enumerate(canny_params):
    start_time = time.time()
    canny_edges = cv2.Canny(img, low_threshold, high_threshold)
    end_time = time.time()
    canny_results.append(canny_edges)
    canny_times.append(end_time - start_time)
    cv2.imwrite(f't3b{i+1}.tif', canny_edges)
    save_image(canny_edges, f't3b{i+1}.tif')

"""
Uncomment to display results figure and time taken for each computation
"""
# # Display Marr-Hildreth results
# plt.figure(figsize=(20, 15))
# plt.suptitle("Marr-Hildreth Edge Detector", fontsize=17)
# plt.subplot(3, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
# for i in range(6):
#     plt.subplot(3, 3, i+4), plt.imshow(mh_results[i], cmap='gray')
#     plt.title(f'Sigma={mh_params[i][0]}, Thres={mh_params[i][1]}\nTime: {mh_times[i]:.3f}s')
# plt.tight_layout()
# plt.show()

# # Display Canny results
# plt.figure(figsize=(20, 15))
# plt.suptitle("Canny Edge Detector", fontsize=17)
# plt.subplot(3, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
# for i in range(6):
#     plt.subplot(3, 3, i+4), plt.imshow(canny_results[i], cmap='gray')
#     plt.title(f'Low={canny_params[i][0]}, High={canny_params[i][1]}\nTime: {canny_times[i]:.3f}s')
# plt.tight_layout()
# plt.show()

print("Marr-Hildreth Times:")
for i, mh_time in enumerate(mh_times):
    print(f"\tParameters {mh_params[i]}: {mh_time:.3f} seconds")

print("\nCanny Times:")
for i, canny_time in enumerate(canny_times):
    print(f"\tParameters {canny_params[i]}: {canny_time:.3f} seconds")

print("=============================        Task 3 Completed Successfully       ===============================")


def connected_components_labeling(edge_map):
    height, width = edge_map.shape
    labels = np.zeros_like(edge_map, dtype=int)
    curr_label = 1
    
    for y in range(height):
        for x in range(width):
            if edge_map[y, x] == 255 and labels[y, x] == 0:
                # Start a new component while using a list as a queue
                queue = [(y, x)]
                labels[y, x] = curr_label
                # not empty
                while queue:
                    cy, cx = queue.pop(0)
                    # Check 8-connected neighbors
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = cy + dy, cx + dx
                            if (0 <= ny < height and 0 <= nx < width and 
                                edge_map[ny, nx] == 255 and labels[ny, nx] == 0):
                                labels[ny, nx] = curr_label
                                queue.append((ny, nx))
                
                curr_label += 1
    
    return labels

def apply_color_map(labels):
    num_labels = np.max(labels)
    cmap = plt.colormaps['tab20']
    
    colors = cmap(np.linspace(0, 1, num_labels))
    colored_map = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    
    for i in range(1, num_labels + 1):
        colored_map[labels == i] = (colors[i-1][:3] * 255).astype(np.uint8)
    return colored_map

# Process all images
mh_results = []
canny_results = []

for i in range(1, 7):
    # Marr-Hildreth figures
    mh_edges = cv2.imread(f't3a{i}.tif', cv2.IMREAD_GRAYSCALE)
    start_time = time.time()
    mh_labels = connected_components_labeling(mh_edges)
    mh_time = time.time() - start_time
    mh_colored = apply_color_map(mh_labels)
    mh_results.append((mh_edges, mh_colored, np.max(mh_labels), mh_time))
    
    # Canny figures
    canny_edges = cv2.imread(f't3b{i}.tif', cv2.IMREAD_GRAYSCALE)
    start_time = time.time()
    canny_labels = connected_components_labeling(canny_edges)
    canny_time = time.time() - start_time
    canny_colored = apply_color_map(canny_labels)
    canny_results.append((canny_edges, canny_colored, np.max(canny_labels), canny_time))

"""
Uncomment to display results figure and time taken for each computation
"""

# # Display Marr-Hildreth results
# plt.figure(figsize=(20, 30))
# plt.suptitle("Marr-Hildreth Edge Detector Results", fontsize=16)
# for i in range(6):
#     plt.subplot(6, 2, 2*i + 1)
#     plt.imshow(mh_results[i][0], cmap='gray')
#     plt.title(f'Edge Map {i+1}')
#     plt.axis('off')
    
#     plt.subplot(6, 2, 2*i + 2)
#     plt.imshow(mh_results[i][1])
#     plt.title(f'Components: {mh_results[i][2]}\nTime: {mh_results[i][3]:.3f}s')
#     plt.axis('off')

# plt.tight_layout()
# plt.show()

# # Display Canny results
# plt.figure(figsize=(20, 30))
# plt.suptitle("Canny Edge Detector Results", fontsize=16)
# for i in range(6):
#     plt.subplot(6, 2, 2*i + 1)
#     plt.imshow(canny_results[i][0], cmap='gray')
#     plt.title(f'Edge Map {i+1}')
#     plt.axis('off')
    
#     plt.subplot(6, 2, 2*i + 2)
#     plt.imshow(canny_results[i][1])
#     plt.title(f'Components: {canny_results[i][2]}\nTime: {canny_results[i][3]:.3f}s')
#     plt.axis('off')

# plt.tight_layout()
# plt.show()

# Save the results and print the summary of the results
for i in range(1, 7):
    cv2.imwrite(f't4a{i}.tif', cv2.cvtColor(mh_results[i-1][1], cv2.COLOR_RGB2BGR))
    cv2.imwrite(f't4b{i}.tif', cv2.cvtColor(canny_results[i-1][1], cv2.COLOR_RGB2BGR))
    print(f"Image set {i}:")
    print(f"\tMarr-Hildreth: {mh_results[i-1][2]} components, {mh_results[i-1][3]:.3f} seconds")
    print(f"\tCanny: {canny_results[i-1][2]} components, {canny_results[i-1][3]:.3f} seconds")

print("=============================        Task 4 Completed Successfully       ===============================")

