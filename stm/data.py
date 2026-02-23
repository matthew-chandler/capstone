# import necessary libraries
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set the filename containing the 64x64 matrix
filename = 'fft_output.txt'  # Change this to your actual data file

# Read the data from the file
data = np.loadtxt(filename)
data = np.fft.fftshift(data)  # Center the FFT data

# Check if the data is 64x64
if data.shape != (64, 64):
	raise ValueError(f"Expected 64x64 matrix, got {data.shape}")

# Create X, Y meshgrid
x = np.arange(64)
y = np.arange(64)
X, Y = np.meshgrid(x, y)

# Plot using plot_surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, data, cmap='viridis')
fig.colorbar(surf)
ax.set_title('2D FFT Output Surface Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Amplitude')
plt.show()

plt.savefig("fft_output.png") # Saves the plot as a PNG image