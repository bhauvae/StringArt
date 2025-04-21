from skimage.transform import radon, resize
from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
import numpy as np

image = shepp_logan_phantom()
image = resize(image, (128, 128))

theta = np.linspace(0.0, 180.0, 60, endpoint=False)
sinogram = radon(image, theta=theta, circle=True)

plt.imshow(sinogram, cmap="gray", aspect="auto")
plt.title("Sinogram")
plt.xlabel("Angle")
plt.ylabel("Projection position")
plt.show()
