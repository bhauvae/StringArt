import numpy as np
from PIL import Image
from skimage.transform import radon

# Load the radon lookup table
radon_string = np.load("radon_lookup.npy")
print("radon_string shape:", radon_string.shape)

# Load and preprocess the image
img = Image.open('circle.png').convert("L")
img_array = np.array(img)

# Define theta for the Radon transform
theta = np.linspace(0.0, 180.0, max(img_array.shape), endpoint=False)

# Perform Radon transform
radon_img = radon(img_array, theta=theta, circle=True)
print("radon_img shape:", radon_img.shape)
np.save("radon_img.npy", radon_img)

# Dimensions of radon_string
m, n, p = radon_string.shape

# Initialize x
x = np.zeros(m)

# Calculate the residuals and update x
for i in range(m):
    # Test with x[i] set to 1
    print(f"{i} out of {m}")
    x[i] = 1
    current_residual = sum(
        np.linalg.norm(np.dot(radon_string[i, :, :], x[i]) - radon_img[:, k])
        for k in range(n)
    )

    # Reset x[i] and calculate residual with x[i] set to 0
    x[i] = 0
    alternative_residual = sum(
        np.linalg.norm(np.dot(radon_string[i, :, :], x[i]) - radon_img[:, k])
        for k in range(n)
    )

    # Update x[i] based on the residuals
    if current_residual < alternative_residual:
        x[i] = 1

print("x:", x)
np.save("strings", x)