import numpy as np
from signals import generate_sine_wave, generate_unit_step, time_shift, time_scale

# Test sine wave

def test_generate_sine_wave():
    # Normal case
    t, y = generate_sine_wave(2, 3, 150)
    assert len(t) == 450
    assert np.isclose(np.max(y), 0.8, atol=0.2)

    # Edge case: zero amplitude
    _, y0 = generate_sine_wave(3, 2, 150, amplitude=0)
    assert np.allclose(y0, 0)


# Test unit step

def test_generate_unit_step():
    # Normal case
    t, u = generate_unit_step(5, 100, step_time=1.1)
    assert len(t) == len(u)
    assert set(np.unique(u)).issubset({0.0, 1.0})

    # Edge case: steps happens too late
    _, u2 = generate_unit_step(1, 100, step_time=5)
    assert np.all(u2 == 0)


# time_shift

def test_time_shift():
    # Normal case
    t = np.linspace(0, 1, 5)
    y = np.array([1,2,3,4,5])
    t_shifted, y2 = time_shift(t, y, 0.6)
    assert np.allclose(t_shifted, t + 0.6)
    assert np.allclose(y2,y)
    
    # Edge case: negative shift
    t_shifted_neg, _ = time_shift(t, y, -1.0)
    assert np.allclose(t_shifted_neg, t -1.0)


# time_scale

def test_time_scale():
    # Normal case
    t = np.linspace(0, 1, 5)
    y = np.array([1, 2, 3, 4, 5])
    t_scaled, y2 = time_scale(t, y, 0.8)
    assert np.allclose(t_scaled, 0.8 * t)
    assert np.allclose(y2, y)

    # Edge case: zero scaling
    t_zero, _ = time_scale(t, y, 0)
    assert np.allclose(t_zero, 0)