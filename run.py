from signals import (
    generate_sine_wave, generate_unit_step,
    time_shift, time_scale,
    fourier_series_basic, fs_reconstruct_basic, simple_fft
)
import matplotlib.pyplot as plt

# Create signals
t, y = generate_sine_wave(5, 2, 100)
t_step, u = generate_unit_step(2, 100, step_time=1)

# Transformations
t_shifted, y_shifted = time_shift(t, y, 0.5)
t_scaled, y_scaled = time_scale(t, y, 0.5)

# Plot basic transformations
plt.plot(t, y, label="Original sine (5 Hz)")
plt.plot(t_shifted, y_shifted, label="Time shift +0.5s")
plt.plot(t_scaled, y_scaled, label="Time scale 0.5x")
plt.plot(t_step, u, label="Unit step (1s)")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.title("Signal Transformations")
plt.legend(); plt.grid(True); plt.show()

# Fourier Series demo
T = 1 / 5
mask = t < T
a0, an, bn = fourier_series_basic(y[mask], t[mask], period=T, M=5)
y_fs = fs_reconstruct_basic(t, T, a0, an, bn)

plt.plot(t, y, label="Original")
plt.plot(t, y_fs, "--", label="FS reconstruction (M=5)")
plt.legend(); plt.grid(True); plt.show()

# Fourier Transform demo
f, A, P = simple_fft(y, sample_rate=100)
plt.plot(f, A, label="Amplitude spectrum")
plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude")
plt.title("Fourier Transform")
plt.grid(True); plt.legend(); plt.show()
