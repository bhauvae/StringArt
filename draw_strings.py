import cv2
import numpy as np


def draw_line_on_circle(canvas_size, circle_center, radius, theta1, theta2):
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


# Parameters
canvas_size = 1080
circle_center = (canvas_size // 2, canvas_size // 2)
radius = 256
theta1 = 30  # Angle in degrees
theta2 = 150  # Angle in degrees

# Draw the line on the circle
canvas = draw_line_on_circle(canvas_size, circle_center, radius, theta1, theta2)

# Display the result
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
