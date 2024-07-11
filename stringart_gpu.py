import cupy as cp
import numpy as np
from PIL import Image
from skimage.transform import radon as sk_radon
import cv2

def crop_image_to_circle(image_path, output_path):
    # Read the input image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Ensure the image has four channels (with alpha)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Create a circular mask
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    radius = min(center[0], center[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Apply the mask to the image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Save the output image
    cv2.imwrite(output_path, result)


def radon_transform_gpu(img_array, theta):
    sinogram_img = sk_radon(img_array, theta=theta, circle=True, gpu=True)
    return cp.asarray(sinogram_img)


def radon_string_gpu(s_string, alpha_string, radius):
    proj_angle = cp.linspace(0.0, 180.0, 512, endpoint=False)
    proj_pos = cp.linspace(0.0, 512.0, 512, endpoint=False)
    radon_transform = cp.zeros((512, 512))

    for alpha_idx, alpha in enumerate(proj_angle):
        for s_idx, s in enumerate(proj_pos):
            sin_angle = cp.sin(cp.deg2rad(alpha - alpha_string))
            if sin_angle == 0:
                radon_transform[alpha_idx, s_idx] = 0
                break

            evaluator = (
                s**2
                + s_string**2
                - 2 * s * s_string * cp.cos(cp.deg2rad(alpha - alpha_string))
            ) / (sin_angle**2)
            if evaluator > radius**2:
                radon_transform[alpha_idx, s_idx] = 0
            else:
                radon_transform[alpha_idx, s_idx] = 1 / cp.abs(sin_angle)

    return radon_transform


def radon_lines():
    n = 300
    circle_theta1 = np.linspace(0.0, 360.0, n, endpoint=False)
    circle_theta2 = np.linspace(0.0, 360.0, n, endpoint=False)

    total_elements = (
        n * (n - 1)
    ) // 2  # Number of unique pairs (theta1, theta2) with theta1 < theta2
    line_theta_lookup = np.zeros((total_elements, 2))
    line_radon_lookup = np.zeros(total_elements)

    index = 0
    for i, theta1 in enumerate(circle_theta1):
        for j, theta2 in enumerate(circle_theta2):
            if theta1 >= theta2:
                continue
            else:
                radius = 256
                s_string = radius * np.cos(np.deg2rad(theta2 - theta1) / 2)
                alpha_string = (theta2 + theta1) / 2
                line_radon_lookup[index] = radon_string_gpu(s_string, alpha_string, radius)
                line_theta_lookup[index] = np.array([theta1, theta2])
                index += 1

    return line_radon_lookup, line_theta_lookup


def get_strings(img_path):
    img = Image.open(img_path).convert("L")
    img_array = cp.array(img)
    radius = 256
    theta = cp.linspace(0.0, 180.0, max(img_array.shape), endpoint=False)

    # Perform Radon transform on GPU
    sinogram_img = radon_transform_gpu(img_array, theta=theta)
    len_adj_sinogram = cp.zeros_like(sinogram_img)
    strings = []

    for _ in range(5):  # Extracting top strings
        for s in range(radius):
            len_of_proj = 2 * ((radius**2 - s**2) ** 0.5)
            len_adj_sinogram[s] = sinogram_img[s, :] / len_of_proj

        max_index = cp.unravel_index(
            cp.argmax(len_adj_sinogram), len_adj_sinogram.shape
        )
        s_idx, theta_idx = max_index

        alpha = theta[theta_idx]
        s = s_idx

        theta1 = alpha - cp.rad2deg(cp.arccos(s / radius))
        theta2 = alpha + cp.rad2deg(cp.arccos(s / radius))
        print((theta1, theta2))
        strings.append((theta1, theta2))

        # Subtracting the identified string's radon transform
        sinogram_img = cp.subtract(sinogram_img, radon_string_gpu(s, alpha, radius))

    return strings


def greedy_solve_gpu(img_path):
    radon_string = cp.load("radon_lookup.npy")
    img = Image.open(img_path).convert("L")
    img_array = cp.array(img)

    theta = cp.linspace(0.0, 180.0, max(img_array.shape), endpoint=False)

    # Perform Radon transform on GPU
    radon_img = radon_transform_gpu(img_array, theta=theta)

    m, n, p = radon_string.shape  # A is 3D: m x n x p
    x = cp.zeros(p)  # x is 1D of length p

    for i in range(p):
        x[i] = 1
        current_residual = cp.sum(
            cp.linalg.norm(cp.dot(radon_string[:, :, k], x) - radon_img[:, k])
            for k in range(n)
        )

        x[i] = 0
        if current_residual < cp.sum(
            cp.linalg.norm(cp.dot(radon_string[:, :, k], x) - radon_img[:, k])
            for k in range(n)
        ):
            x[i] = 1

    return cp.asnumpy(x)


def create_string_art(strings):
    canvas = np.zeros((512, 512), dtype=np.uint8)
    circle_center = [512 // 2, 512 // 2]
    radius = 256

    for string in strings:
        theta1 = np.deg2rad(string[0])
        theta2 = np.deg2rad(string[1])

        x1 = int(circle_center[0] + radius * np.cos(theta1))
        y1 = int(circle_center[1] + radius * np.sin(theta1))
        x2 = int(circle_center[0] + radius * np.cos(theta2))
        y2 = int(circle_center[1] + radius * np.sin(theta2))

        cv2.line(canvas, (x1, y1), (x2, y2), 255, 1)

    return canvas


# Example usage:
# np.save('strings', greedy_solve_gpu('circle.png'))
# art = create_string_art(get_strings("circle.png"))
# cv2.imwrite("art.png", art)
# cv2.imshow("Canvas", art)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
