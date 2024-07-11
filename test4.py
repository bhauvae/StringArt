import numpy as np
from skimage.transform import radon
import cv2
import matplotlib.pyplot as plt


def crop_image_to_circle(image_path, output_path, max_length=1920):
    # Read the input image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Resize the image while preserving aspect ratio
    height, width, _ = image.shape
    if max(width, height) > max_length:
        if width > height:
            new_width = max_length
            new_height = int(height * max_length / width)
        else:
            new_height = max_length
            new_width = int(width * max_length / height)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a circular mask
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    radius = min(center[0], center[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Apply the mask to the image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Crop a square around the circle
    square_size = radius * 2
    crop_x = (width - square_size) // 2
    crop_y = (height - square_size) // 2
    cropped_square = result[
        crop_y : crop_y + square_size, crop_x : crop_x + square_size
    ]

    # Create output image with black background
    output_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)
    output_image[:, :] = (0, 0, 0)
    output_image[: cropped_square.shape[0], : cropped_square.shape[1]] = cropped_square

    # Save the output image
    cv2.imwrite(output_path, output_image)

    return radius


def radon_string(s_string, alpha_string, radius):
    proj_angle = np.linspace(0.0, 180.0, radius * 2, endpoint=False)
    proj_pos = np.linspace(-radius, radius, radius * 2, endpoint=False)
    radon_transform = np.zeros((radius * 2, radius * 2))
    max_index = [0, 0]
    for alpha_idx, alpha in enumerate(proj_angle):
        sin_angle = np.sin(np.deg2rad(alpha - alpha_string))
        if np.abs(sin_angle) == 0:
            max_index[1] = alpha_idx
            continue

        for s_idx, s in enumerate(proj_pos):
            if s == s_string:
                max_index[0] = s_idx

            evaluator = (
                s**2
                + s_string**2
                - 2 * s * s_string * np.cos(np.deg2rad(alpha - alpha_string))
            ) / (sin_angle**2)
            if evaluator > radius**2:
                radon_transform[s_idx, alpha_idx] = 0
            else:
                radon_transform[s_idx, alpha_idx] = 1 / np.abs(sin_angle)

    radon_transform[max_index[0], max_index[1]] = radon_transform.max() * 1.1

    radon_transform = radon_transform / radon_transform.max()

    return radon_transform


def plot_sinogram(sinogram):
    plt.imshow(
        sinogram,
        cmap=plt.cm.hot,
        extent=(0, 180, 0, sinogram.shape[0]),
        aspect="auto",
    )
    plt.colorbar(label="Intensity")
    plt.xlabel("Projection angle (degrees)")
    plt.ylabel("Projection position (pixels)")
    plt.title("Sinogram")
    plt.show()


def get_strings(img_path, num_of_anchors):  # using max
    # Load and convert the image to grayscale
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    height, width = image.shape
    center = (width // 2, height // 2)
    radius = min(center[0], center[1])

    theta = np.linspace(0.0, 180.0, radius * 2, endpoint=False)

    # Perform Radon transform
    sinogram_img = radon(image, theta=theta, circle=True)
    sinogram_img = sinogram_img / sinogram_img.max()

    strings = []
    len_adj_sinogram = np.zeros_like(sinogram_img)
    for _ in range(num_of_anchors):  # Extracting top strings
        for s in range(radius * 2):
            if s == 0:
                len_adj_sinogram[s, :] = 0
            else:
                len_of_proj = 2 * ((radius**2 - (s - radius) ** 2) ** 0.5)
                len_adj_sinogram[s, :] = sinogram_img[s, :] / len_of_proj

        max_index = np.unravel_index(
            np.argmax(len_adj_sinogram), len_adj_sinogram.shape
        )

        s_idx, theta_idx = max_index[0], max_index[1]

        alpha = theta[theta_idx]
        s = s_idx - radius

        theta1 = alpha - np.rad2deg(np.arccos(s / radius))
        theta2 = alpha + np.rad2deg(np.arccos(s / radius))

        strings.append((theta1, theta2))

        # Subtracting the identified string's radon transform

        sinogram_string = radon_string(s, alpha, radius)

        sinogram_img = np.subtract(sinogram_img, sinogram_string)

        sinogram_img[sinogram_img < 0] = 0

        print(f"String {len(strings)}: {theta1} to {theta2}")

    return strings, radius


def create_string_art(strings, radius):
    canvas = np.zeros((radius * 2, radius * 2), dtype=np.uint8)
    circle_center = [radius * 2 // 2, radius * 2 // 2]

    for string in strings:

        theta1 = np.deg2rad(string[0])
        theta2 = np.deg2rad(string[1])

        x1 = int(circle_center[0] + radius * np.cos(theta1))
        y1 = int(circle_center[1] + radius * np.sin(theta1))
        x2 = int(circle_center[0] + radius * np.cos(theta2))
        y2 = int(circle_center[1] + radius * np.sin(theta2))

        cv2.line(canvas, (x1, y1), (x2, y2), 255, 1)

    return canvas


# crop_image_to_circle("test.jpg", "test.png")
# art = create_string_art(
#     [tuple(np.random.uniform(0, 360, size=2)) for _ in range(500)], 1278 // 2
# )
art = create_string_art(get_strings("test.png", 250))
# # plot_sinogram(radon_string(128, 90, 256))
# # Save and display the result
cv2.imwrite("test_250.png", art)
# cv2.imshow("Canvas", art)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
