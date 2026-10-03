# Image Processing and Recognition

Classic image processing in Python, mostly implemented from scratch with NumPy and compared against OpenCV's built-in versions. Individual work from my Computer Science degree at Wilfrid Laurier University (2024).

The fourth part of this work, transfer learning with a ResNet18 on my own handwritten digits, has its own repository: [Handwritten-Digit-Recognition](https://github.com/omar3903/Handwritten-Digit-Recognition).

## 1. Intensity transforms and resizing (`1-intensity-transforms-and-resizing/`)

- Image resizing with **nearest-neighbour** and **bilinear interpolation** written from scratch, compared with OpenCV's nearest, bilinear and bicubic resizing.
- **Negative**, **power-law (gamma)** transforms at 8 gamma values, and **bit-plane slicing**.
- **Histogram equalization** and **histogram specification** (matching one X-ray's histogram to another), from scratch.

Outputs are in `results/`.

```bash
cd 1-intensity-transforms-and-resizing
python intensity_transforms.py
```

## 2. Filters and edge detection (`2-filters-and-edge-detection/`)

- **2D convolution** with averaging, **Gaussian** and **Sobel** filters from scratch in NumPy, each timed against OpenCV's equivalent.
- A **Marr-Hildreth** edge detector with my own zero-crossing test, compared with OpenCV's **Canny** across 6 parameter settings each.
- **Connected-component labelling** (8-connected, breadth-first search) written from scratch, with colour-coded component maps (`results/Task 4a.png`, `results/Task 4b.png`).

```bash
cd 2-filters-and-edge-detection
python filters_and_edges.py
```

## 3. Iris and pupil detection (`3-iris-and-pupil-detection/`)

Finds the pupil and iris boundaries in eye images with Gaussian blurring, Canny edges and OpenCV's **Hough circle transform**, choosing the best circle for each by its distance from the image centre.

The eye images are not included, because they are biometric images from a course dataset. To run it, place images named `iris1.tif` to `iris5.tif` in this folder:

```bash
cd 3-iris-and-pupil-detection
python iris_detection.py
```

## Setup

```bash
pip install -r requirements.txt
```
