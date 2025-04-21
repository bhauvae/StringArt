# %%
import cv2
import numpy as np
import random
from skimage.transform import radon
import os
# from main import plot_sinogram

from numba import njit


def image_preprocessing(image_path, output_path="temp.png", max_length=1440):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load '{image_path}'")
    h, w = img.shape[:2]
    # 1) center‐crop to square ≤ max_length
    side = min(h, w, max_length)
    top, left = (h - side) // 2, (w - side) // 2
    img = img[top : top + side, left : left + side]

    # 2) if the cropped square is smaller than max_length, upscale it;
    #    if it's exactly max_length, this is a no‑op
    img = cv2.resize(img, (max_length, max_length), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)

    # circular mask
    h, w = inv.shape
    center, radius = (w // 2, h // 2), min(w, h) // 2
    mask = np.zeros_like(inv)
    cv2.circle(mask, center, radius, 255, -1)
    circ = cv2.bitwise_and(inv, mask)

    out = cv2.merge([circ, circ, circ])
    cv2.imwrite(output_path, out)
    return circ


@njit(cache=True)
def _compute_raw_transform(positions, sin_d, cos_d, denom, dist, radius):
    M, N = positions.shape[0], sin_d.shape[0]
    raw = np.zeros((M, N), dtype=np.float32)
    r2, d2 = radius * radius, dist * dist
    for i in range(M):
        s = positions[i]
        for j in range(N):
            dnm = denom[j]
            if dnm == 0.0:
                continue
            ev = (s * s + d2 - 2 * s * dist * cos_d[j]) / dnm
            if ev <= r2:
                raw[i, j] = 1.0 / abs(sin_d[j])
    return raw


def string_radon_transform(alpha_str, dist_str, alphas, sinogram_shape, radius):
    positions = np.linspace(
        -radius, radius, sinogram_shape[0], endpoint=False, dtype=np.float32
    )
    delta = np.deg2rad(alphas - alpha_str).astype(np.float32)
    sin_d, cos_d = np.sin(delta), np.cos(delta)
    denom = sin_d * sin_d
    raw = _compute_raw_transform(
        positions, sin_d, cos_d, denom, float(dist_str), float(radius)
    )
    θ = np.arccos(dist_str / radius)

    return raw, np.deg2rad(alpha_str) - θ, np.deg2rad(alpha_str) + θ


def energy(target, rendered, lam, string_count):
    mse = np.mean((target.astype(np.float32) - rendered.astype(np.float32)) ** 2)
    return mse + lam * string_count


def compute_canvas(
    strings, canvas_size, center, radius, smooth_core=0.5, smooth_trans=1.0
):
    """
    Draw each string with a flat‐top + linear‐taper profile:
      ‑ for  d <= smooth_core:      brightness = 1.0
      ‑ for  smooth_core<d<smooth_core+smooth_trans:
                                    brightness = (smooth_core+smooth_trans–d)/smooth_trans
      ‑ for  d >= smooth_core+smooth_trans: brightness = 0

    strings     : list of (θ1, θ2) pairs
    canvas_size : (h, w)
    center      : (cx, cy)
    radius      : circle radius
    smooth_core : radius of flat (max brightness) core, in pixels
    smooth_trans: width of linear taper region, in pixels
    """
    h, w = canvas_size

    # we'll build a floating canvas in [0…1], where 1 is “string‐center brightness”
    canvas = np.zeros((h, w), dtype=np.float32)

    for θ1, θ2 in strings:
        # 1) draw the one‐pixel‐thick line into a binary mask
        mask = np.zeros((h, w), dtype=np.uint8)
        x1 = int(center[0] + radius * np.cos(θ1))
        y1 = int(center[1] + radius * np.sin(θ1))
        x2 = int(center[0] + radius * np.cos(θ2))
        y2 = int(center[1] + radius * np.sin(θ2))
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=1)

        # 2) distance‐transform: for every pixel, how far from that line?
        #    invert mask so line pixels are zero (distance=0)
        dist = cv2.distanceTransform(
            255 - mask, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE
        )

        # 3) build the trapezoid profile: flat + taper
        prof = np.clip((smooth_core + smooth_trans - dist) / smooth_trans, 0.0, 1.0)
        prof[dist <= smooth_core] = 1.0

        # 4) composite into canvas.
        line_strength = 0.1
        canvas += prof * line_strength

    canvas = np.clip(canvas, 0.0, 1.0)

    # 5) convert back to 8‑bit image (0=white background, 255=black lines)
    #    note: canvas=1→ black (255), canvas=0→ white (0)
    out = (255 * (1.0 - canvas)).astype(np.uint8)
    return out





# @njit(cache=True)
def generate_strings(
    sinogram,  # 2D float32
    pos,  # 1D float32 positions
    alphas,  # 1D float32 angles (deg)
    radius,  # float64
    attenuation,  # float64
    string_count,  # int64
    diag_interval=10,  # int64
):
   
    sg = sinogram.copy()

    # precompute length weights
    lengths = 2.0 * np.sqrt(np.clip(radius * radius - pos * pos, 0.0, np.inf))
    lengths[lengths == 0] = np.inf

    
    count = 0
    seeds = []
    for it in range(string_count):
        # find highest remaining peak
        ws = sg / lengths[:, None]
        idx = np.nanargmax(ws)
        i, j = np.unravel_index(idx, ws.shape)
        if sg[i, j] <= 0:
            break  # no more non-zero peaks
        proj, t1, t2 = string_radon_transform(
            alphas[j], pos[i], alphas, sinogram.shape, radius
        )

        # full subtraction
        proj_norm = proj / np.max(proj + 1e-8)  # avoid /0
        sg -= proj_norm * sg[i, j] * attenuation
        sg = np.clip(sg, 0, None)
        # explicitly zero that bin
        sg[i, j] = 0.0

        seeds.append((t1, t2))
        count += 1
        if string_count == count:
            break
        
        
        # plot_sinogram(sg)
        # plot_sinogram(proj)


        # 5) diagnostic output every diag_interval iters
        if (it + 1) % diag_interval == 0:
            print(
                "[Seed] iter", it + 1, ">> strings added:", count, ">> angle:", np.rad2deg(t1), "to", np.rad2deg(t2)
            )

    print("[Seed] Completed", count, "strings in", it + 1, "iterations")
    return seeds



def generate_image(
    input_preprocessed="temp.png",
    output_path="string-art.png",
    string_count=None,
    angular_resolution=16,
    perform_sar=False,
    sa_iterations=30000,
    sa_lam=1e-5,
    sa_temp=5.0,
    sa_cooling=0.995,
    remove_prob=0.3,
    upscale_factor=4,
):
    # --- Load and setup ---
    gray = cv2.imread(input_preprocessed, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if gray is None:
        raise FileNotFoundError(f"Cannot load '{input_preprocessed}'")
    h, w = gray.shape
    center = (w // 2, h // 2)
    radius = min(center)

    
    alphas = np.linspace(
        0, 180, angular_resolution * (2 * radius), endpoint=False, dtype=np.float32
    )


    # --- Radon with caching ---
    # Construct a unique cache key based on image path, resolution, and radius
    
    cache_file = os.path.join( "cache_radon.npz")
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        sinogram = data["sinogram"]
        print(f"Loaded sinogram from cache: {cache_file}")
    else:
        print("Computing Radon transform and caching...")
        
        sinogram = radon(gray, theta=alphas, circle=True)
        np.savez_compressed(cache_file, sinogram=sinogram)
        print(f"Sinogram saved to cache: {cache_file}")

    # --- Precompute length weights ---
    pos = np.linspace(
        -radius, radius, sinogram.shape[0], endpoint=False, dtype=np.float32
    )
    lengths = 2 * np.sqrt(np.clip(radius * radius - pos * pos, 0, None))
    lengths[lengths == 0] = np.inf

    # === 1) GREEDY SEEDING UNTIL EXHAUSTION ===
    # === 1) GREEDY SEEDING UNTIL EXHAUSTION (now in Numba) ===

    # placeholder; will be recalculated inside Numba

    # run Numba seeding
    attenuation = 0.7
    # run Numba seeding
    seeds = generate_strings(
        sinogram.astype(np.float32),
        pos.astype(np.float32),
        alphas.astype(np.float32),
        float(radius),
        float(attenuation),
        int(string_count or 1e6),
    )

    
    print(f"Completed seeding: {len(seeds)} total strings")

    # === 2) SIMULATED ANNEALING REFINEMENT ===
    best = seeds.copy()
    best_canvas = compute_canvas(best, gray.shape, center, radius)
    best_E = energy(gray, best_canvas, sa_lam, len(best))
    T = sa_temp

    if perform_sar:
        for it in range(1, sa_iterations + 1):
            cand = best.copy()
            if random.random() < remove_prob and cand:
                cand.pop(random.randrange(len(cand)))
            else:
                θ1, θ2 = np.random.uniform(0, 2 * np.pi, 2)
                cand.append((θ1, θ2))

            canvas = compute_canvas(cand, gray.shape, center, radius)
            E = energy(gray, canvas, sa_lam, len(cand))
            Δ = E - best_E

            if Δ < 0 or np.exp(-Δ / T) > random.random():
                best, best_canvas, best_E = cand, canvas, E

            T *= sa_cooling
            if it % 2000 == 0:
                print(
                    f"[SA] {it}/{sa_iterations} — strings={len(best)} — energy={best_E:.4f}"
                )

    # === 3) SMOOTHING & UPSCALE ===

    final = compute_canvas(best, gray.shape, center, radius)

    big = cv2.resize(
        final, (w * upscale_factor, h * upscale_factor), interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite(output_path, big)

    print(f"Saved → {output_path} (total strings after SA: {len(best)})")


def main(input_image="img2.jpeg", output_art="string-art2.png"):
    image_preprocessing(input_image)
    generate_image(
        input_preprocessed="temp.png", output_path=output_art, string_count=15000
    )


if __name__ == "__main__":
    main()

# %%
