import numpy as np
from PIL import Image, ImageDraw
from skimage.transform import radon
import matplotlib.pyplot as plt
import pandas as pd
import cv2


def crop_to_circle(image_path):
    # Open the image
    img = Image.open(image_path).convert("RGBA")

    # Calculate the size and radius of the largest circle
    width, height = img.size
    radius = min(width, height) // 2

    # Create a mask with a white circle on a black background
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    center = (width // 2, height // 2)
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        fill=255,
    )

    # Apply the mask to the image
    circular_img = Image.new("RGBA", (width, height))
    circular_img.paste(img, (0, 0), mask=mask)

    # Crop the image to the bounding box of the circle
    bbox = (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )
    circular_img = circular_img.crop(bbox)

    return circular_img


def perform_radon_transform(circular_img):
    # Convert to grayscale
    gray_img = circular_img.convert("L")

    # Convert to numpy array
    img_array = np.array(gray_img)

    # Perform Radon transform
    theta = np.linspace(0.0, 180.0, max(img_array.shape), endpoint=False)
    sinogram = radon(img_array, theta=theta, circle=True)

    # Display the original and the Radon transform
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.set_title("Original Circular Image")
    ax1.imshow(img_array, cmap=plt.cm.Greys_r)
    ax1.axis("off")

    ax2.set_title("Radon Transform")
    ax2.imshow(
        sinogram,
        cmap=plt.cm.Greys_r,
        extent=(0, 180, 0, sinogram.shape[0]),
        aspect="auto",
    )
    ax2.set_xlabel("Projection angle (degrees)")
    ax2.set_ylabel("Projection position (pixels)")

    plt.tight_layout()
    plt.show()


def draw_line_on_circle_with2theta(canvas_size, circle_center, radius, theta1, theta2):
    # Create a blank canvas
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    # Convert angles from degrees to radians
    theta1 = np.deg2rad(theta1)
    theta2 = np.deg2rad(theta2)

    # Calculate Cartesian coordinates for the points
    x1 = int(circle_center[0] + radius * np.cos(theta1))
    y1 = int(circle_center[1] + radius * np.sin(theta1))
    x2 = int(circle_center[0] + radius * np.cos(theta2))
    y2 = int(circle_center[1] + radius * np.sin(theta2))

    # Draw the circle
    cv2.circle(canvas, circle_center, radius, (255, 255, 255), 1)

    # Draw the line
    cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return canvas


def get_strings(circular_img):
    img = Image.open(circular_img).convert("L")

    # Convert to numpy array
    img_array = np.array(img)

    # Perform Radon transform
    theta = np.linspace(0.0, 180.0, max(img_array.shape), endpoint=False)
    sinogram = radon(img_array, theta=theta, circle=True)

    print(pd.DataFrame(sinogram, index=theta))

    pass


def radon_lines():
    n = 500
    circle_theta1 = np.linspace(0.0, 360.0, n, endpoint=False)
    circle_theta2 = np.linspace(0.0, 360.0, n, endpoint=False)

    line_theta_lookup = np.array([])
    line_radon_lookup = np.zeros([])
    for i, theta1 in enumerate(circle_theta1):
        for j, theta2 in enumerate(circle_theta2):
            if theta1 >= theta2:
                continue
            else:
                index = 512 * i + j
                radius = 512
                s_string = radius * np.cos(np.deg2rad(theta2 - theta1) / 2)
                alpha_string = (theta2 + theta1) / 2
                line_radon_lookup[index] = radon_string(s_string, alpha_string, radius)
                line_theta_lookup[index] = np.array([theta1, theta2])

    return line_radon_lookup


def radon_string(s_string, alpha_string, radius):
    proj_angle = np.linspace(0.0, 180.0, 512, endpoint=False)
    proj_pos = np.linspace(0.0, 512.0, 512, endpoint=False)
    radon_transform = np.zeros((512, 512))

    for alpha_idx, alpha in enumerate(proj_angle):
        for s_idx, s in enumerate(proj_pos):
            evaluator = (
                s**2
                + s_string**2
                - 2 * s * s_string * np.cos(np.deg2rad(alpha - alpha_string))
            ) / (np.sin(np.deg2rad(alpha - alpha_string)) ** 2)
            if evaluator > radius**2:
                radon_transform[alpha_idx, s_idx] = 0
            else:
                radon_transform[alpha_idx, s_idx] = 1 / np.abs(
                    np.sin(np.deg2rad(alpha - alpha_string))
                )

    return radon_transform


# Example usage
# circular_img = crop_to_circle("img.png")
# perform_radon_transform(circular_img)
# get_strings("circle.png")

sinogram = radon_string(256, 90, 512)
theta = np.linspace(0.0, 180.0, 512, endpoint=False)


# Display the original and the Radon transform
fig, ax2 = plt.subplots(1, 1, figsize=(10, 4.5))

ax2.set_title("Radon Transform")
ax2.imshow(
    sinogram,
    cmap=plt.cm.Greys_r,
    extent=(0, 180, 0, sinogram.shape[0]),
    aspect="auto",
)
ax2.set_xlabel("Projection angle (degrees)")
ax2.set_ylabel("Projection position (pixels)")

plt.tight_layout()
plt.show()
