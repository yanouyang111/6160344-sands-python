import numpy as np
from signals import (
    generate_sine_wave, generate_unit_step,
    time_shift, time_scale,
    fourier_series_basic, fs_reconstruct_basic, simple_fft
)

# generate_sine_wave

def test_generate_sine_wave():
    # Normal case
    t, y = generate_sine_wave(5, 2, 100)
    assert len(t) == 200        # 2 s × 100 Hz
    assert np.isclose(np.max(y), 1.0, atol=0.1)

    # Edge case: zero amplitude
    _, y0 = generate_sine_wave(5, 2, 100, amplitude=0)
    assert np.allclose(y0, 0)


# generate_unit_step

def test_generate_unit_step():
    # Normal case
    t, u = generate_unit_step(2, 100, step_time=1)
    assert len(t) == len(u)
    assert set(np.unique(u)).issubset({0.0, 1.0})

    # Edge case: step_time beyond duration
    _, u2 = generate_unit_step(2, 100, step_time=3)
    assert np.all(u2 == 0)


# time_shift

def test_time_shift():
    # Normal case
    t = np.linspace(0, 1, 5)
    y = np.arange(5)
    t_shifted, _ = time_shift(t, y, 1)
    assert np.allclose(t_shifted, t + 1)

    # Edge case: negative shift
    t_shifted_neg, _ = time_shift(t, y, -0.5)
    assert np.allclose(t_shifted_neg, t - 0.5)


# time_scale

def test_time_scale():
    # Normal case
    t = np.linspace(0, 1, 5)
    y = np.arange(5)
    t_scaled, _ = time_scale(t, y, 2)
    assert np.allclose(t_scaled, 2 * t)

    # Edge case: zero scaling (all times zero)
    t_zero, _ = time_scale(t, y, 0)
    assert np.allclose(t_zero, 0)


# fourier_series_basic

def test_fourier_series_basic():
    # Normal case: 5 Hz sine wave
    f = 5
    T = 1 / f
    sr = 200
    t = np.linspace(0, T, int(sr * T), endpoint=False)
    y = np.sin(2 * np.pi * f * t)

    a0, an, bn = fourier_series_basic(y, t, T, M=5)
    y_rec = fs_reconstruct_basic(t, T, a0, an, bn)
    assert np.allclose(y, y_rec, atol=0.05)

    # Edge case: zero signal
    y_zero = np.zeros_like(y)
    a0, an, bn = fourier_series_basic(y_zero, t, T, M=5)
    assert np.allclose(a0, 0)
    assert np.allclose(an, 0)
    assert np.allclose(bn, 0)


# simple_fft

def test_simple_fft():
    # Normal case: 5 Hz sine wave
    f = 5
    sr = 200
    t = np.linspace(0, 1, sr, endpoint=False)
    y = np.sin(2 * np.pi * f * t)
    freq, amp, phase = simple_fft(y, sr)
    main_freq = freq[np.argmax(amp)]
    assert np.isclose(main_freq, f, atol=0.5)

    # Edge case: zero signal
    y_zero = np.zeros_like(y)
    freq, amp, phase = simple_fft(y_zero, sr)
    assert np.allclose(amp, 0)

