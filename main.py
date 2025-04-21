import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon
from concurrent.futures import ThreadPoolExecutor
from numba import njit


def image_preprocessing(image_path, output_path, max_length=512):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")

    height, width = image.shape[:2]
    if max(width, height) > max_length:
        if width > height:
            new_width = max_length
            new_height = int(height * max_length / width)
        else:
            new_height = max_length
            new_width = int(width * max_length / height)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        image = image[:, :, 0]  # fallback if already grayscale

    image = cv2.bitwise_not(image)

    h, w = image.shape
    center = (w // 2, h // 2)
    radius = min(center)
    circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle, center, radius, 255, -1)
    final_gray = cv2.bitwise_and(image, circle)

    output = cv2.merge([final_gray] * 3)
    cv2.imwrite(output_path, output)

def plot_sinogram(sinogram):
    plt.imshow(
        sinogram,
        cmap=plt.cm.hot,
        extent=(0, 180, -sinogram.shape[0] / 2, sinogram.shape[0] / 2),
        aspect="auto",
    )
    plt.colorbar(label="Intensity")
    plt.xlabel("Projection angle (degrees)")
    plt.ylabel("Projection position (pixels)")
    plt.title("Sinogram")
    plt.show()


@njit
def compute_radon_string_fast(positions, alpha_string, distance_string, alpha, radius):
    delta_radians = np.deg2rad(alpha - alpha_string)
    output = np.zeros((len(positions), len(delta_radians)), dtype=np.float32)

    for i in range(len(positions)):
        s = positions[i]
        for j in range(len(delta_radians)):
            delta = delta_radians[j]
            sin_val = np.sin(delta)
            cos_val = np.cos(delta)
            if sin_val == 0:
                continue
            eval_ = (s**2 + distance_string**2 - 2 * s * distance_string * cos_val) / (
                sin_val**2
            )
            if eval_ <= radius**2:
                output[i, j] = 1.0 / abs(sin_val)
    return output


@njit
def smooth_profile_transform(inverted, core_half, trans_width):
    h, w = inverted.shape
    dist_map = np.zeros_like(inverted, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            if inverted[y, x] == 0:
                continue
            min_dist = float("inf")
            for j in range(h):
                for i in range(w):
                    if inverted[j, i] == 0:
                        dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                        if dist < min_dist:
                            min_dist = dist
            dist_map[y, x] = min_dist

    profile = np.clip((core_half + trans_width - dist_map) / trans_width, 0.0, 1.0)
    for y in range(h):
        for x in range(w):
            if dist_map[y, x] <= core_half:
                profile[y, x] = 1.0
    return profile


def generate_image(output_path, iterations=15):
    image = cv2.imread("temp.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("temp.png not found")

    height, width = image.shape
    center = (width // 2, height // 2)
    radius = min(center[0], center[1])
    resolution = radius * 2
    alpha = np.linspace(0.0, 180.0, resolution, endpoint=False)
    sinogram_img = radon(image, theta=alpha, circle=True)
    positions = np.linspace(-radius, radius, sinogram_img.shape[0], endpoint=False)

    strings = []

    for _ in range(iterations):
        max_idx = np.unravel_index(np.argmax(sinogram_img), sinogram_img.shape)
        s_idx, alpha_idx = max_idx
        s_val = positions[s_idx]
        a_val = alpha[alpha_idx]

        sinogram_string = compute_radon_string_fast(
            positions, a_val, s_val, alpha, radius
        )
        sinogram_img -= sinogram_string
        np.clip(sinogram_img, 0, None, out=sinogram_img)
        sinogram_img[s_idx, alpha_idx] = 0

        theta = np.arccos(s_val / radius)
        theta1 = np.deg2rad(a_val) - theta
        theta2 = np.deg2rad(a_val) + theta
        strings.append((theta1, theta2))

    def draw_string_lines():
        scale = 4
        canvas = np.ones((height * scale, width * scale), dtype=np.uint8) * 255

        for t1, t2 in strings:
            x1 = int(center[0] + radius * np.cos(t1)) * scale
            y1 = int(center[1] + radius * np.sin(t1)) * scale
            x2 = int(center[0] + radius * np.cos(t2)) * scale
            y2 = int(center[1] + radius * np.sin(t2)) * scale
            cv2.line(canvas, (x1, y1), (x2, y2), 0, 1)

        return canvas

    def smooth_and_save(canvas):
        inverted = np.where(canvas > 0, 255, 0).astype(np.uint8)
        smoothed = smooth_profile_transform(inverted, 0.5, 2.0)
        output = (smoothed * 255).astype(np.uint8)
        cv2.imwrite(output_path, output)

    with ThreadPoolExecutor() as executor:
        future = executor.submit(draw_string_lines)
        canvas = future.result()
        executor.submit(smooth_and_save, canvas).result()


def main(input_image_path="img.jpg", output_image_path="string-art.png"):
    image_preprocessing(input_image_path, "temp.png")
    generate_image(output_image_path)


if __name__ == "__main__":
    main()
