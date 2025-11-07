from signals import generate_sine_wave, generate_unit_step, time_shift, time_scale
import matplotlib.pyplot as plt

# Create signals

t, y = generate_sine_wave(3, 2, 150)
t_step, u = generate_unit_step(5, 100, step_time=1.1)

# Transformations

t_shifted, y_shifted = time_shift(t, y, 0.6)
t_scaled, y_scaled = time_scale(t, y, 0.8)

# Plot all

plt.plot(t, y, label="sine wave 3 Hz")
plt.plot(t_shifted, y_shifted, label="time shifted by 0.6 s")
plt.plot(t_scaled, y_scaled, label="time scaled by 0.8")
plt.plot(t_step, u, label="unit step at 1.1 s")

plt.xlabel("time (s)")
plt.ylabel("amplltude")
plt.title("signals")
plt.legend()
plt.grid(True)
plt.show()