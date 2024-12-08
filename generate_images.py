import numpy as np
import matplotlib.pyplot as plt
import os

# Function to generate standing wave
def standing_wave(x, t, n, A, L, v):
    """
    Parameters:
        x: Position along the string (array-like)
        t: Time (scalar)
        n: Harmonic number (integer)
        A: Amplitude (scalar)
        L: Length of the string (scalar)
        v: Wave speed (scalar)
    Returns:
        y: Displacement of the string (array-like)
    """
    omega = n * np.pi * v / L  # Angular frequency
    k = n * np.pi / L          # Wave number
    return A * np.sin(k * x) * np.sin(omega * t)

# Create folders for each harmonic
output_base = 'standing_wave_images'
for n in range(1, 5):
    folder_path = os.path.join(output_base, f'harmonic_{n}')
    os.makedirs(folder_path, exist_ok=True)

# Parameters
num_examples = 1

# Generate and save plots
for i in range(num_examples):
    for n in range(1, 5):
        t = np.random.uniform(0, 2)
        L = np.random.uniform(0, 2)
        A = np.random.uniform(0, 2)
        v = np.random.uniform(0, 2)
        x = np.linspace(0, L, 500)  # Discretize the string
        y = standing_wave(x, t, n, A, L, v)

        # Plot the standing wave
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, label=f'Harmonic {n}, t={t:.2f}s', color='blue')

        # Apply random horizontal and vertical stretches after plotting
        horizontal_stretch = np.random.uniform(0.8, 1.2)  # Horizontal stretch factor
        vertical_stretch = np.random.uniform(0.8, 1.2)    # Vertical stretch factor
        ax.set_aspect(horizontal_stretch / vertical_stretch)

        # Save the plot to the appropriate folder
        folder_path = os.path.join(output_base, f'harmonic_{n}')
        file_path = os.path.join(folder_path, f'standing_wave_{i+1:04d}.png')
        plt.savefig(file_path, dpi=150)
        plt.close()

print(f"Plots saved in the '{output_base}' directory.")
