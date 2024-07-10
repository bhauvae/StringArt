import cv2
import numpy as np


def radon_string(s_string, alpha_string, radius):
    proj_angle = np.linspace(0.0, 180.0, 512, endpoint=False)
    proj_pos = np.linspace(0.0, 512.0, 512, endpoint=False)
    radon_transform = np.zeros((512, 512))

    for alpha_idx, alpha in enumerate(proj_angle):
        for s_idx, s in enumerate(proj_pos):
            sin_angle = np.sin(np.deg2rad(alpha - alpha_string))
            if sin_angle == 0:
                radon_transform[alpha_idx, s_idx] = 0
                break

            evaluator = (
                s**2
                + s_string**2
                - 2 * s * s_string * np.cos(np.deg2rad(alpha - alpha_string))
            ) / (sin_angle**2)
            if evaluator > radius**2:
                radon_transform[alpha_idx, s_idx] = 0
            else:
                radon_transform[alpha_idx, s_idx] = 1 / np.abs(sin_angle)

    return radon_transform


def radon_lines_lookup(num_anchors):
    n = num_anchors
    circle_theta1 = np.linspace(0.0, 360.0, n, endpoint=False)
    circle_theta2 = np.linspace(0.0, 360.0, n, endpoint=False)

    total_elements = (
        n * (n - 1)
    ) // 2  # Number of unique pairs (theta1, theta2) with theta1 < theta2
    line_theta_lookup = []
    line_radon_lookup = []

    for i, theta1 in enumerate(circle_theta1):
        for j, theta2 in enumerate(circle_theta2):
            if theta1 >= theta2:
                continue
            else:
                print(f"element {i* n + j} out of {total_elements}")
                radius = 256
                s_string = radius * np.cos(np.deg2rad(theta2 - theta1) / 2)
                alpha_string = (theta2 + theta1) / 2
                line_radon_lookup.append(radon_string(s_string, alpha_string, radius))
                line_theta_lookup.append(np.array([theta1, theta2]))

    return line_radon_lookup, line_theta_lookup


radon_lookup, theta_lookup = radon_lines_lookup(num_anchors=50)
# np.save("radon_lookup", radon_lookup)
# np.save("theta_lookup", theta_lookup)
