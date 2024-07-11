import cupy as cp
from skimage.transform import radon
import cv2
import matplotlib.pyplot as plt

def crop_image_to_circle(image_path, output_path, max_length=1920):
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")

    print("Image loaded successfully.")
    height, width, _ = image.shape
    print(f"Original image size: {width}x{height}")

    if max(width, height) > max_length:
        if width > height:
            new_width = max_length
            new_height = int(height * max_length / width)
        else:
            new_height = max_length
            new_width = int(width * max_length / height)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Resized image to: {new_width}x{new_height}")

    # Create a circular mask
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    radius = min(center[0], center[1])
    print(f"Creating circular mask with radius: {radius}")

    mask = cp.zeros((height, width), dtype=cp.uint8)
    cp.circle(mask, center, radius, 255, -1)
    result = cp.bitwise_and(image, image, mask=mask)

    # Crop a square around the circle
    square_size = radius * 2
    crop_x = (width - square_size) // 2
    crop_y = (height - square_size) // 2
    cropped_square = result[
        crop_y: crop_y + square_size, crop_x: crop_x + square_size
    ]
    print(f"Cropped image to square size: {square_size}x{square_size}")

    # Create output image with black background
    output_image = cp.zeros((square_size, square_size, 3), dtype=cp.uint8)
    output_image[:, :] = (0, 0, 0)
    output_image[: cropped_square.shape[0], : cropped_square.shape[1]] = cropped_square

    # Save the output image
    cv2.imwrite(output_path, cp.asnumpy(output_image))
    print(f"Output image saved to: {output_path}")

    return radius


def radon_string(s_string, alpha_string, radius):
    print(f"Generating Radon transform for s_string={s_string}, alpha_string={alpha_string}, radius={radius}")
    proj_angle = cp.linspace(0.0, 180.0, radius * 2, endpoint=False)
    proj_pos = cp.linspace(-radius, radius, radius * 2, endpoint=False)
    radon_transform = cp.zeros((radius * 2, radius * 2))
    max_index = cp.array([0, 0])
    
    for alpha_idx, alpha in enumerate(proj_angle):
        sin_angle = cp.sin(cp.deg2rad(alpha - alpha_string))
        if cp.abs(sin_angle) == 0:
            max_index[1] = alpha_idx
            continue

        for s_idx, s in enumerate(proj_pos):
            if s == s_string:
                max_index[0] = s_idx

            evaluator = (
                s**2
                + s_string**2
                - 2 * s * s_string * cp.cos(cp.deg2rad(alpha - alpha_string))
            ) / (sin_angle**2)
            radon_transform[s_idx, alpha_idx] = cp.where(evaluator > radius**2, 0, 1 / cp.abs(sin_angle))

    radon_transform[max_index[0], max_index[1]] = radon_transform.max() * 1.1

    radon_transform = radon_transform / radon_transform.max()

    print("Radon transform generated.")
    return radon_transform


def plot_sinogram(sinogram):
    print("Plotting sinogram.")
    plt.imshow(
        cp.asnumpy(sinogram),
        cmap=plt.cm.hot,
        extent=(0, 180, 0, sinogram.shape[0]),
        aspect="auto",
    )
    plt.colorbar(label="Intensity")
    plt.xlabel("Projection angle (degrees)")
    plt.ylabel("Projection position (pixels)")
    plt.title("Sinogram")
    plt.show()
    print("Sinogram plotted.")


def get_strings(img_path, num_of_anchors):  # using max
    print(f"Loading image for Radon transform from: {img_path}")
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image file not found at {img_path}")

    height, width = image.shape
    center = (width // 2, height // 2)
    radius = min(center[0], center[1])
    print(f"Image loaded. Radius for Radon transform: {radius}")

    theta = cp.linspace(0.0, 180.0, radius * 2, endpoint=False)

    # Perform Radon transform
    print("Performing Radon transform.")
    sinogram_img = radon(image, theta=cp.asnumpy(theta), circle=True)
    sinogram_img = cp.array(sinogram_img) / cp.max(sinogram_img)

    strings = []
    len_adj_sinogram = cp.zeros_like(sinogram_img)
    for _ in range(num_of_anchors):  # Extracting top strings
        for s in range(radius * 2):
            if s == 0:
                len_adj_sinogram[s, :] = 0
            else:
                len_of_proj = 2 * ((radius**2 - (s - radius) ** 2) ** 0.5)
                len_adj_sinogram[s, :] = sinogram_img[s, :] / len_of_proj

        max_index = cp.unravel_index(
            cp.argmax(len_adj_sinogram), len_adj_sinogram.shape
        )

        s_idx, theta_idx = max_index[0], max_index[1]

        alpha = theta[theta_idx]
        s = s_idx - radius

        theta1 = alpha - cp.rad2deg(cp.arccos(s / radius))
        theta2 = alpha + cp.rad2deg(cp.arccos(s / radius))

        strings.append((theta1, theta2))

        # Subtracting the identified string's Radon transform
        print(f"Processing string: {theta1} to {theta2}")
        sinogram_string = radon_string(s, alpha, radius)
        sinogram_img = cp.subtract(sinogram_img, sinogram_string)
        sinogram_img = cp.maximum(sinogram_img, 0)

        print(f"String {len(strings)} of {num_of_anchors}: {theta1} to {theta2}")

    print("Strings extraction completed.")
    return strings, radius


def create_string_art(strings, radius):
    print(f"Creating string art with {len(strings)} strings and radius {radius}.")
    canvas = cp.zeros((radius * 2, radius * 2), dtype=cp.uint8)
    circle_center = [radius * 2 // 2, radius * 2 // 2]

    for string in strings:
        theta1 = cp.deg2rad(string[0])
        theta2 = cp.deg2rad(string[1])

        x1 = int(circle_center[0] + radius * cp.cos(theta1))
        y1 = int(circle_center[1] + radius * cp.sin(theta1))
        x2 = int(circle_center[0] + radius * cp.cos(theta2))
        y2 = int(circle_center[1] + radius * cp.sin(theta2))

        cv2.line(cp.asnumpy(canvas), (x1, y1), (x2, y2), 255, 1)

    print("String art created.")
    return canvas


# Example usage:
# Uncomment these lines to run the example usage
# crop_image_to_circle("test.jpg", "test.png")
# art = create_string_art(
#     [tuple(cp.random.uniform(0, 360, size=2)) for _ in range(500)], 1278 // 2
# )
# cv2.imwrite("test_500.png", cp.asnumpy(art))

strings, radius = get_strings("test.png", 50)
art = create_string_art(strings, radius)
cv2.imwrite("test_500.png", cp.asnumpy(art))
