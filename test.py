from signals import generate_sine_wave, generate_unit_step
import numpy as np

# Test 1: Normal + edge case sine wave
def test_generate_sine_wave():
    frequency = 5
    duration = 2
    sample_rate = 100
    amplitude = 1.0
    phase = 0

    t, y = generate_sine_wave(frequency, duration, sample_rate, amplitude, phase)

    # Normal cases
    assert len(t) == int(sample_rate * duration)      
    assert np.isclose(t[0], 0.0)                      

    # Edge case: amplitude = 0 gives all zeros
    assert np.allclose(generate_sine_wave(5, 2, 100, 0, 0)[1], 0)

# Test 2: Unit step 
def test_generate_unit_step():
    duration = 2
    sample_rate = 100
    step_time = 1

    t_step, u = generate_unit_step(duration, sample_rate, step_time)

    # Normal cases
    assert len(t_step) == len(u)                     
    assert set(np.unique(u)).issubset({0.0, 1.0})    
