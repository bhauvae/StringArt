import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from skimage.transform import radon
from numba import njit, prange


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging format and level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class Config:
    """
    Configuration container for all key parameters.
    """

    # Input/Output
    input_image = Path("img.jpg")  # input image path
    output_image = Path("output.png")  # output image path

    # Image preprocessing
    max_canvas_size: int = 512  # max width/height after crop & resize

    # Radon transform
    cache_file: Path = Path("cache_radon.npz")
    angular_resolution: int = 2  # subdivisions per degree (affects theta sampling)

    # Greedy seeding
    attenuation_factor: float = 0.1  # reduce projection strength each step
    max_string_count: int = 15000  # maximum number of strings to seed
    diag_interval: int = 1000  # diagnostic print interval

    # Simulated annealing refinement
    perform_sim_anneal: bool = False
    sa_iterations: int = 30000
    sa_lambda: float = 1e-5
    sa_initial_temp: float = 5.0
    sa_cooling_rate: float = 0.995
    sa_remove_probability: float = 0.3

    # Rendering
    line_thickness: int = 1
    smooth_core_radius: float = 0.5  # pixels
    smooth_taper_width: float = 1.0  # pixels
    line_strength: float = 0.036  # contribution per string
    upscale_factor: int = 4


class StringArtGenerator:
    """
    Generates string-art from a grayscale image:
    1. Preprocess image: crop, resize, invert.
    2. Compute Radon transform (sinogram).
    3. Greedy seeding of string segments.
    4. Optional simulated annealing refinement.
    5. Composite strings into final image.


    """

    def __init__(
        self,
        config: Config,
        input_path: Path,
        output_path: Path,
    ) -> None:
        self.config = config
        self.input_path = input_path
        self.output_path = output_path
        self.gray: Optional[np.ndarray] = None
        self.sinogram: Optional[np.ndarray] = None
        self.positions: Optional[np.ndarray] = None
        self.thetas: Optional[np.ndarray] = None
        self.radius: Optional[float] = None
        self.center: Optional[Tuple[int, int]] = None
        logging.getLogger(__name__)

    def run(self) -> None:
        """
        High-level entry: generate string-art and save to disk.
        """
        self._preprocess_image()
        logging.info("Computing image radon transform...")
        self._compute_radon()
        seeds = self._greedy_seed()
        if self.config.perform_sim_anneal:
            seeds = self._simulated_anneal(seeds)
        art = self._render_strings(seeds)
        self._save_image(art)
        logging.info(f"Saved string art to {self.output_path}")

    def _preprocess_image(self) -> None:
        """
        Load image, center-crop to square, resize to max_canvas_size,
        convert to grayscale, invert, and apply circular mask.
        """
        # Assume new_width == new_height == your desired canvas size
        new_width, new_height = self.config.max_canvas_size, self.config.max_canvas_size

        # 1) Read and get original dimensions
        img = cv2.imread(str(self.input_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {self.input_path}")
        h, w = img.shape[:2]

        # 2) Compute largest square side & its top/left in the original image
        side  = min(h, w)
        top   = (h - side) // 2
        left  = (w - side) // 2

        # 3) Crop the centered square
        crop  = img[top : top + side, left : left + side]

        # 4) Downscale that square to your canvas
        resized = cv2.resize(
            crop,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        inverted = cv2.bitwise_not(gray.astype(np.uint8)).astype(np.float32)
        # inverted = gray

        # Apply circular mask
        size = inverted.shape[0]
        self.center = (size // 2, size // 2)
        self.radius = size / 2
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2)
        mask = dist <= self.radius

        self.gray = (inverted * mask).astype(np.float32)
        cv2.imwrite("temp.png", self.gray.astype(np.uint8))
        logging.info("Image preprocessed: size=%d", size)

    def _compute_radon(self) -> None:
        """
        Compute or load cached Radon transform (sinogram).
        """
        cfg = self.config
        self.thetas = np.linspace(
            0,
            180,
            int(cfg.angular_resolution * 2 * self.radius),
            endpoint=False,
            dtype=np.float32,
        )
        if cfg.cache_file.exists():
            data = np.load(cfg.cache_file)
            self.sinogram = data["sinogram"]
            logging.info(f"Loaded sinogram from cache: {cfg.cache_file}")
        else:
            self.sinogram = radon(
                self.gray,
                theta=self.thetas,
                circle=True,  # Analytical Radon
            )
            np.savez_compressed(cfg.cache_file, sinogram=self.sinogram)
            logging.info(f"Computed and cached sinogram: {cfg.cache_file}")

        # Precompute sample positions along detector
        n_r, n_theta = self.sinogram.shape
        self.positions = np.linspace(
            -self.radius,
            self.radius,
            n_r,
            endpoint=False,
            dtype=np.float32,
        )
        logging.info("Radon transform ready: shape=%s", self.sinogram.shape)

    @staticmethod
    @njit(cache=True, parallel=True)
    def _compute_projection_kernel(
        positions: np.ndarray,
        sin_vals: np.ndarray,
        cos_vals: np.ndarray,
        denom: np.ndarray,
        distance: float,
        radius: float,
    ) -> np.ndarray:
        """
        Compute binary projection influence kernel:
        K(i,j) = 1/|sin(delta_theta)| if point at position i and angle j is on string path,
        else 0. Used for greedy subtraction.
        """
        M, N = positions.shape[0], sin_vals.shape[0]

        kernel = np.zeros((M, N), dtype=np.float32)
        r2 = radius**2
        dist2 = distance**2
        for i in prange(M):  # <— parallel loop over detector positions
            s = positions[i]
            for j in range(N):
                dnm = denom[j]
                if dnm == 0.0:
                    continue
                val = (s * s + dist2 - 2 * s * distance * cos_vals[j]) / dnm
                if val <= r2:
                    kernel[i, j] = 1.0 / abs(sin_vals[j])
        return kernel

    # TODO: IMPROVE
    def _greedy_seed(self) -> List[Tuple[float, float]]:
        """
        Greedy selection of string endpoints by subtracting strongest projections.
        """
        cfg = self.config
        sg = self.sinogram.copy().astype(np.float32)

        # Precompute geometric weights
        lengths = 2 * np.sqrt(np.clip(self.radius**2 - self.positions**2, 0, None))
        lengths[lengths == 0] = np.inf

        seeds: List[Tuple[float, float]] = []
        sin_t = None
        cos_t = None
        denom = None

        for count in range(cfg.max_string_count):
            weights = sg / lengths[:, None]
            idx = np.nanargmax(weights)
            i, j = divmod(idx, weights.shape[1])

            if sg[i, j] <= 0:
                break

            # Initialize precomputed arrays once
            if sin_t is None:
                delta = np.deg2rad(self.thetas - self.thetas[j]).astype(np.float32)
                sin_t = np.sin(delta)
                cos_t = np.cos(delta)
                denom = sin_t * sin_t

            kernel = self._compute_projection_kernel(
                self.positions,
                sin_t,
                cos_t,
                denom,
                float(self.positions[i]),
                float(self.radius),
            )

            norm = kernel / (kernel.max() + 1e-8)
            sg -= norm * sg[i, j] * cfg.attenuation_factor
            sg = np.clip(sg, 0, None)
            sg[i, j] = 0

            #TODO: the values are negative, and also greater than 180, restriction required?
            theta1 = np.deg2rad(self.thetas[j]) - np.arccos(
                self.positions[i] / self.radius
            )
            theta2 = np.deg2rad(self.thetas[j]) + np.arccos(
                self.positions[i] / self.radius
            )
            seeds.append((theta1, theta2))

            if (count + 1) % cfg.diag_interval == 0:
                logging.info(
                    "[Greedy max lookup] %d strings added, latest theta1 = %.2f°, theta2 = %.2f°",
                    count + 1,
                    np.rad2deg(theta1),
                    np.rad2deg(theta2),
                )

        logging.info("Completed greedy seeding with %d strings", len(seeds))
        return seeds

    def _simulated_anneal(
        self, seeds: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Perform simulated annealing to refine string set by minimizing energy.
        E = MSE(target,rendered) + lambda * N_strings
        """
        cfg = self.config
        best = seeds.copy()
        best_canvas = self._render_strings(best)
        best_energy = self._compute_energy(best_canvas, len(best))
        T = cfg.sa_initial_temp
        logging.info("Starting simulated annealing refinement...")
        for it in range(1, cfg.sa_iterations + 1):
            candidate = best.copy()
            # Randomly remove or add a string
            if candidate and np.random.rand() < cfg.sa_remove_probability:
                candidate.pop(np.random.randint(len(candidate)))
            else:
                theta1, theta2 = np.random.uniform(0, 2 * np.pi, 2)
                candidate.append((theta1, theta2))

            cand_canvas = self._render_strings(candidate)
            cand_energy = self._compute_energy(cand_canvas, len(candidate))
            dE = cand_energy - best_energy
            if dE < 0 or np.exp(-dE / T) > np.random.rand():
                best, best_canvas, best_energy = candidate, cand_canvas, cand_energy

            T *= cfg.sa_cooling_rate
            if it % (cfg.sa_iterations // 10) == 0:
                logging.info(
                    "[SA] %d/%d — strings=%d — energy=%.4f",
                    it,
                    cfg.sa_iterations,
                    len(best),
                    best_energy,
                )
        return best

    def _compute_energy(self, canvas: np.ndarray, n_strings: int) -> float:
        """
        Energy = MSE(target, canvas) + lambda * n_strings
        """
        mse = np.mean((self.gray - canvas.astype(np.float32)) ** 2)
        return mse + self.config.sa_lambda * n_strings

    def _render_strings(self, seed_angles: List[Tuple[float, float]]) -> np.ndarray:
        """
        Draw each string with trapezoidal brightness profile and composite.

        Returns:
            2D uint8 image: 0=white, 255=black
        """
        size = self.gray.shape[0]
        canvas = np.zeros((size, size), dtype=np.float32)

        logging.info("Rendering final image")

        for theta1, theta2 in seed_angles:
            # Draw infinite-resolution line mask
            mask = np.zeros((size, size), dtype=np.uint8)
            x1 = int(self.center[0] + self.radius * np.cos(theta1))
            y1 = int(self.center[1] + self.radius * np.sin(theta1))
            x2 = int(self.center[0] + self.radius * np.cos(theta2))
            y2 = int(self.center[1] + self.radius * np.sin(theta2))
            cv2.line(
                mask, (x1, y1), (x2, y2), 255, thickness=self.config.line_thickness
            )

            dist_map = cv2.distanceTransform(
                255 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            # Brightness profile: flat core + linear taper
            prof = np.clip(
                (
                    self.config.smooth_core_radius
                    + self.config.smooth_taper_width
                    - dist_map
                )
                / self.config.smooth_taper_width,
                0.0,
                1.0,
            )
            prof[dist_map <= self.config.smooth_core_radius] = 1.0

            canvas += prof * self.config.line_strength

        # Clamp and invert: canvas=1 -> black (255), 0-> white (0)
        canvas = np.clip(canvas, 0.0, 1.0)
        rendered = (255 * (1 - canvas)).astype(np.uint8)
        # Upscale with nearest neighbor for crisp edges
        return cv2.resize(
            rendered,
            (size * self.config.upscale_factor, size * self.config.upscale_factor),
            interpolation=cv2.INTER_NEAREST,
        )

    def _save_image(self, img: np.ndarray) -> None:
        """
        Save final image to output path, creating directories if needed.
        """
        os.makedirs(self.output_path.parent, exist_ok=True)
        cv2.imwrite(str(self.output_path), img)


if __name__ == "__main__":
    setup_logging()

    conf = Config()

    generator = StringArtGenerator(conf, conf.input_image, conf.output_image)
    generator.run()
    logging.info("String art generation completed.")
