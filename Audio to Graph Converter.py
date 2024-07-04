import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa

# Define parameters
N_FFT = 2048 # Number of FFT points
FPS = 30 # Frames per second
WIN_SIZE = int(N_FFT * FPS / librosa.get_default_sr()) # Window size in seconds

# Create figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize audio stream
stream = librosa.stream('input.wav', block_length=WIN_SIZE, hop_length=WIN_SIZE)

# Initialize plot data
x = np.arange(0, N_FFT)
y = np.arange(0, WIN_SIZE)
X, Y = np.meshgrid(x, y)
Z = np.zeros((WIN_SIZE, N_FFT))

# Create 3D surface plot
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', vmin=-1, vmax=1)

# Define update function
def update(frame):
    # Read audio data
    data = next(stream)[0]
    
    # Compute STFT
    stft = np.abs(librosa.stft(data, n_fft=N_FFT))
    
    # Normalize and transpose data
    stft /= np.max(stft)
    stft = np.transpose(stft)
    
    # Update surface plot data
    surf.set_array(stft.ravel())
    surf.changed()
    
    return surf,

# Create animation
ani = animation.FuncAnimation(fig, update, interval=1000/FPS, blit=True)

# Show plot
plt.show()