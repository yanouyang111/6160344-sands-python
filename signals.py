import numpy as np

# Signal Generators 

def generate_sine_wave(frequency, duration, sample_rate, amplitude=1.0, phase=0.0):
    """
    Create a sine wave signal.

    Args:
        frequency (float): Frequency in Hz.
        duration (float): Duration in seconds.
        sample_rate (int): Sampling rate in Hz.
        amplitude (float): Amplitude (default 1.0).
        phase (float): Phase in radians (default 0.0).

    Returns:
        t (ndarray): Time values.
        y (ndarray): Sine wave values.
    """
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    y = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return t, y


def generate_unit_step(duration, sample_rate, step_time=0.0):
    """
    Create a unit step signal u(t - step_time).

    Args:
        duration (float): Duration in seconds.
        sample_rate (int): Sampling rate in Hz.
        step_time (float): Step position in seconds.

    Returns:
        t (ndarray): Time values.
        y (ndarray): Step signal (0 or 1).
    """
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    y = np.where(t >= step_time, 1.0, 0.0)
    return t, y


# Signal Operations

def time_shift(t, y, shift):
    """
    Shift the signal in time.

    Args:
        t (ndarray): Original time.
        y (ndarray): Original signal.
        shift (float): Time shift in seconds (positive = delay).

    Returns:
        t_shifted (ndarray): Shifted time.
        y (ndarray): Same signal values.
    """
    return t + shift, y


def time_scale(t, y, alpha):
    """
    Scale the signal in time.

    Args:
        t (ndarray): Original time.
        y (ndarray): Original signal.
        alpha (float): Scale factor (t_new = alpha * t).
                       alpha < 1 → stretch, alpha > 1 → compress.

    Returns:
        t_scaled (ndarray): Scaled time.
        y (ndarray): Same signal values.
    """
    return alpha * t, y


# Fourier Series

def fourier_series_basic(y, t, period, M):
    """
    Compute basic Fourier series coefficients numerically.

    Args:
        y (ndarray): Samples over one period [0, T).
        t (ndarray): Time samples.
        period (float): Fundamental period T.
        M (int): Number of harmonics.

    Returns:
        a0 (float): Constant term.
        an (ndarray): Cosine coefficients.
        bn (ndarray): Sine coefficients.
    """
    N = len(t)
    w0 = 2 * np.pi / period
    a0 = (2.0 / N) * np.sum(y)
    an = np.zeros(M)
    bn = np.zeros(M)
    for n in range(1, M + 1):
        an[n-1] = (2.0 / N) * np.sum(y * np.cos(n * w0 * t))
        bn[n-1] = (2.0 / N) * np.sum(y * np.sin(n * w0 * t))
    return a0, an, bn


def fs_reconstruct_basic(t, period, a0, an, bn):
    """
    Reconstruct signal from Fourier series coefficients.

    Args:
        t (ndarray): Time array.
        period (float): Fundamental period T.
        a0, an, bn: Coefficients from fourier_series_basic().

    Returns:
        yM (ndarray): Reconstructed signal.
    """
    w0 = 2 * np.pi / period
    yM = 0.5 * a0 * np.ones_like(t)
    for n in range(1, len(an) + 1):
        yM += an[n-1] * np.cos(n * w0 * t) + bn[n-1] * np.sin(n * w0 * t)
    return yM


# Fourier Transform 

def simple_fft(y, sample_rate):
    """
    Compute one-sided Fourier transform using FFT.

    Args:
        y (ndarray): Signal samples.
        sample_rate (float): Sampling rate in Hz.

    Returns:
        f (ndarray): Frequencies (Hz).
        amp (ndarray): Amplitude spectrum.
        phase (ndarray): Phase in radians.
    """
    N = len(y)
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(N, d=1.0 / sample_rate)
    amp = np.abs(Y) / N
    if N > 1:
        amp[1:-1] *= 2.0
    phase = np.angle(Y)
    return f, amp, phase 