from signals import generate_sine_wave
t,y = generate_sine_wave(5,2,100)
print("First 10 samples of sine waves:")
print(y[:10])