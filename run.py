from signals import generate_sine_wave, generate_unit_step
import matplotlib.pyplot as plt

t,y = generate_sine_wave(5,2,100) # Generate original sine wave
plt.plot(t,y,label="Original sine wave')

t_shifted = t + 0.5
plt.plot(t_shifted, y, label="Time shifted (+0.5s)") # sine wave delayed by 0.5s

t_scaled = t * 0.5
plt.plot(t_scaled, y, label="Time scaled （0.5x)") # sine wave compressed by factor 2

t2,u= generate_unit_step(2, 100, step_time=1) # Generate original step unit function
plt.plot(t2, u, label="Unit step (at t=1s)")

y_sum = y + u
plt.plot(t, y_sum, label="sine + unit step") # Add the 2 signals

plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title ("Signal transformations")
plt.legend()
plt.grid(True)
plt.show()