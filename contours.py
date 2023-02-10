import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt


def region_growing_segmentation(img_gray: np.ndarray, seed_pixel: tp.Tuple[int, int]) -> np.ndarray:
    """Segment the image by considering a region-growing method."""
    # Smooth image to improve results.
    img_gray = img_gray.astype('float')
    img_gray = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)
    # Initialize mask.
    segmentation = np.zeros_like(img_gray)

    pixels_to_process = [seed_pixel]
    while pixels_to_process:
        pixel = pixels_to_process.pop()
        # Add the pixel to the segmentation
        segmentation[pixel] = 1

        # Add [some of] its neighbours to the list of current pixels
        candidates = [(pixel[0] + 1, pixel[1]), (pixel[0] - 1, pixel[1]), (pixel[0], pixel[1] + 1), (pixel[0], pixel[1] - 1)]
        # Filter out pixels that are out of bounds
        candidates = [c for c in candidates
                      if 0 <= c[0] < img_gray.shape[0] and 0 <= c[1] < img_gray.shape[1]]
        # Filter out pixels that are already in the segmentation
        candidates = [c for c in candidates
                      if segmentation[c] == 0]
        # Filter out pixels that are too white
        candidates = [c for c in candidates
                      if img_gray[c] < 150]
        # Filter out pixels that are too different
        candidates = [c for c in candidates if
                      np.abs(img_gray[c] - img_gray[pixel]) < 50]
        # Add neighbours to the list of pixels to process
        pixels_to_process = candidates + pixels_to_process

        # Check for an ending condition
        if np.sum(segmentation) > 0.1 * img_gray.size:
            break

    return segmentation


if __name__ == "__main__":
    img_bgr = cv2.imread('../samples/Image.jpg', cv2.IMREAD_COLOR)
    img_bgr = cv2.resize(img_bgr, (0, 0), fx=0.25, fy=0.25)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Detect region of interest (where brown cells might should be present)
    region_of_interest = cv2.erode((img_gray <= 128).astype('uint8')*255, np.ones((3, 3)))
    # Randomly select a seed within the region of interest
    positive_points = np.where(region_of_interest != 0)
    seed_idx = np.random.choice(len(positive_points[0]))
    seed_point = positive_points[0][seed_idx], positive_points[1][seed_idx]

    results = {
        'Region of interest': region_of_interest,
        'Region-growing segmentation': region_growing_segmentation(img_gray, seed_point),
    }
    results = {k: v for k, v in results.items() if v is not None}

    # Visualize images
    fig, axs = plt.subplots(1, 2)
    [ax.axis('off') for ax in axs.flatten()]
    # Show one image per subplot
    for ax, (title, binary_image) in zip(axs.flatten(), results.items()):
        subimage = np.copy(img_bgr)
        subimage[binary_image != cv2.erode(binary_image.astype('uint8'), np.ones((3, 3))), ...] = (0, 255, 0)
        cv2.line(subimage, pt1=(seed_point[1] - 5, seed_point[0] - 5), pt2=(seed_point[1] + 5, seed_point[0] + 5),
                 color=(0, 0, 255), thickness=2)
        cv2.line(subimage, pt1=(seed_point[1] - 5, seed_point[0] + 5), pt2=(seed_point[1] + 5, seed_point[0] - 5),
                 color=(0, 0, 255), thickness=2)
        ax.imshow(cv2.cvtColor(subimage, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
    plt.show()
