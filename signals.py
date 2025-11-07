import numpy as np

# Signal generators 

def generate_sine_wave(frequency, duration, sample_rate, amplitude=0.8, phase=0.5):
    """
    Create a sine wave signal.

    Parameters:
        frequency: freqeuncy in Hz.
        duration: duration in seconds.
        sample_rate: sampling rate in Hz.
        amplitude: amplitude (default 0.8).
        phase: phase in radians (default 0.5).

    Returns:
        t: time values.
        y: sine wave values.
    """
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    y = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return t, y

def generate_unit_step(duration, sample_rate, step_time=0.7):
    """
    Create a unit step signal from step_time=0.7.

    Parameters:
        duration: duration in seconds.
        sample_rate: sampling rate in Hz.
        step_time: step positon in seconds.

    Returns:
        t: time values.
        y: step signal (0 or 1).
    """
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    y = []
    for i in range(len(t)):
        if t[i]>=step_time:
            y.append(1.0)
        else:
            y.append(0.0)
    return t, np.array(y)

# Signal operations

def time_shift(t, y, shift):
    """
    Shift the signal in time.

    parameters:
        t: original time.
        y: original signal.
        shift: time shift in secods.

    Returns:
        t_shifted: shifted time.
        y: same signal values.
    """
    t2 = []
    for i in range(len(t)):
        t2.append(t[i]+shift)
    return np.array(t2),y

def time_scale(t, y, alpha=1.0):
    """
    scale the signal in time.

    parameters:
        t: original time.
        y: original signal.
        alpha: scale facter (default 1.0)
        
    Returns:
        t_scaled: scaled time.
        y: same signal values.
    """
    t3 = []
    for i in range(len(t)):
        t3.append(alpha*t[i])
    return np.array(t3), y