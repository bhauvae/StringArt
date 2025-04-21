# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon
from numba import njit


def image_preprocessing(image_path, output_path, max_length=1440):
    """
    1. Reads an image (with alpha channel if present).
    2. Optionally resizes preserving aspect ratio (max side = max_length).
    3. Removes background using a uniform threshold (default 0.29).
    4. Converts to grayscale, negates, and auto-adjusts intensity.
    5. Applies a circular crop (outside circle -> transparent).

    """
    # --- Load image ---
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")

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

    # --- Subject isolation ---
    def subject_isolation():
        pass

    # --- Greyscale and negative ---
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(image)

    # --- Circular crop ---
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center)
    circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle, center, radius, 255, -1)

    final_gray = cv2.bitwise_and(image, circle)

    # --- Merge to BGRA ---
    output = cv2.merge([final_gray, final_gray, final_gray])

    # --- Save ---
    cv2.imwrite(output_path, output)


def generate_image(output_path, iterations=5000, angular_resolution=4):
    # Load and convert the image to grayscale
    image = cv2.imread("temp.png", cv2.IMREAD_GRAYSCALE)

    height, width = image.shape
    center = (width // 2, height // 2)
    radius = min(center[0], center[1])
    projection_resolution = angular_resolution * (
        radius * 2
    )  # Controls resolution of the projection, number of nails essentially
    strings = []
    alpha = np.linspace(0.0, 180.0, projection_resolution, endpoint=False)

    # Perform Radon transform
    print("Performing Radon transform...")
    sinogram_img = radon(image, theta=alpha, circle=True)

    # # Normalize the radon transform
    # sinogram_img = sinogram_img / np.max(sinogram_img + 1e-8)  # avoid /0
    # sinogram_img = np.clip(sinogram_img, 0, 1)

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

    @njit(cache=True)
    def _compute_raw_transform(
        positions, sin_delta, cos_delta, denom, distance, radius
    ):
        M = positions.shape[0]
        N = sin_delta.shape[0]
        raw = np.zeros((M, N), dtype=np.float32)
        r2 = radius * radius
        dist2 = distance * distance
        for i in range(M):
            s = positions[i]
            for j in range(N):
                dnm = denom[j]
                if dnm == 0.0:
                    continue
                ev = (s * s + dist2 - 2 * s * distance * cos_delta[j]) / dnm
                if ev <= r2:
                    raw[i, j] = 1.0 / abs(sin_delta[j])
        return raw

    def string_radon_transform(
        alpha_string,
        distance_string,
        projection_angles=alpha,
        image_radius=radius,
    ):
        """
        Accelerated string Radon transform with Numba JIT and no meshgrids.
        """
        # 1) positions from -R to +R
        positions = np.linspace(
            -image_radius, image_radius, sinogram_img.shape[0], endpoint=False
        ).astype(np.float32)

        # 2) angle diffs
        delta = np.deg2rad(projection_angles - alpha_string).astype(np.float32)
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)

        # 3) denom = sin²(delta)
        denom = sin_delta * sin_delta

        # 4) raw transform via fast JIT loop
        raw_transform = _compute_raw_transform(
            positions,
            sin_delta,
            cos_delta,
            denom,
            float(distance_string),
            float(image_radius),
        )

        # 5) compute start/end angles of the string
        θ = np.arccos(distance_string / image_radius)

        theta1_rad = np.deg2rad(alpha_string) - θ
        theta2_rad = np.deg2rad(alpha_string) + θ

        return raw_transform, theta1_rad, theta2_rad

    # compute row positions from –radius to +radius
    positions = np.linspace(-radius, radius, sinogram_img.shape[0], endpoint=False, dtype=np.float32)

    # compute length L = 2 * sqrt(r^2 – s^2)
    lengths = 2.0 * np.sqrt(np.clip(radius*radius - positions*positions, 0.0, None))

    # avoid division by zero
    lengths[lengths == 0] = np.inf
    
    for _ in range(iterations):
        # === FIND MAX VALUE INDEX IN SINOGRAM ===  LEN ADJUSTED RHO
              

        # apply weighting to each row of the sinogram
        weighted_sinogram = sinogram_img / lengths[:, np.newaxis]

        # find the max in the length-adjusted sinogram

        max_index = np.unravel_index(
            np.nanargmax(weighted_sinogram), weighted_sinogram.shape
        )
        max_s_position, max_alpha_index = (
            max_index  # row is s (projection position), col is angle index
        )

        sinogram_string, theta1, theta2 = string_radon_transform(
            alpha[max_alpha_index], max_s_position - radius
        )  # s is the projection position

        #  subtract *scaled* string projection
        sinogram_img = sinogram_img - sinogram_string
        sinogram_img = np.clip(sinogram_img, 0, None)  # Avoid negative values
        sinogram_img[max_s_position, max_alpha_index] = 0  # Set the peak to zero

        print(f"String {len(strings)}: {np.rad2deg(theta1)} to {np.rad2deg(theta2)}")
        strings.append((theta1, theta2))

    def plot_strings(strings, core_half=0.5, transition=2.0, upscale_factor=4):
        def smooth_string_canvas(
            canvas, core_half_width: float, transition_width: float
        ) -> np.ndarray:
            # 1) invert mask so that line pixels become 0, background becomes 255
            inverted = np.where(canvas > 0, 0, 255).astype(np.uint8)

            # 2) distance transform: for each pixel, distance to nearest zero (i.e. to the line)
            dist_map = cv2.distanceTransform(
                inverted, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )

            # 3) build the piecewise‑linear profile
            #    - everywhere ≤ core_half_width        → 1.0
            #    - everywhere > core_half_width+trans → 0.0
            #    - in between: 1.0 − (d − core_half_width)/transition_width
            profile = np.clip(
                (core_half_width + transition_width - dist_map) / transition_width,
                a_min=0.0,
                a_max=1.0,
            )
            profile[dist_map <= core_half_width] = 1.0

            return profile.astype(np.float32)

        canvas_size = np.array(image.shape) * upscale_factor
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255

        for theta1, theta2 in strings:
            x1 = int(center[0] + radius * np.cos(theta1)) * upscale_factor
            y1 = int(center[1] + radius * np.sin(theta1)) * upscale_factor
            x2 = int(center[0] + radius * np.cos(theta2)) * upscale_factor
            y2 = int(center[1] + radius * np.sin(theta2)) * upscale_factor
            cv2.line(canvas, (x1, y1), (x2, y2), color=0, thickness=1)

        # 2) Smooth it
        canvas = smooth_string_canvas(
            canvas, core_half_width=core_half, transition_width=transition
        )

        # 3) Save or display

        #    To save as an 8‑bit PNG you can rescale back to 0–255:
        canvas = (canvas * 255).astype(np.uint8)
        cv2.imwrite(output_path, canvas)

    plot_strings(strings)


def main(input_image_path="img.jpg", output_image_path="string-art.png"):
    image_preprocessing(input_image_path, "temp.png")
    generate_image(output_image_path)


if __name__ == "__main__":
    main()
# %%
