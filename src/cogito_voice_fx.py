"""
cogito_voice_fx.py — "Distant transmission" voice post-processing for Cogito.

Takes a clean TTS-rendered WAV (mono, any engine) and runs it through a chain
built for one specific feel: something transmitting from far away, on a signal
that isn't fully locked on. Not a cheerful radio-host sound. Not a cartoon
robot. Cold, textured, slightly unstable — the kind of "weird" that makes you
lean in rather than laugh.

Design notes (why each stage exists, not just what it does):

  1. Wow & flutter            -> a slow, small, irregular pitch wobble. This is
                                 the single biggest lever for "weird" over
                                 "robotic" — it says the signal is unstable,
                                 not that the voice is synthetic.
  2. Bandpass (300-3200 Hz)   -> real transmission bandwidth. Removes the warm
                                 low end and airy high end that make a voice
                                 feel "present in the room."
  3. Metallic resonance       -> a short feedback delay (comb filter) that
                                 carves resonant peaks/notches into the
                                 spectrum, instead of the flat, even rolloff a
                                 plain bandpass leaves behind. This is what
                                 makes it sound like it's resonating through
                                 something hollow/metallic rather than just
                                 "filtered" — and it's most of the fix for
                                 flatness.
  4. Saturation                -> soft (tanh) harmonic saturation, mixed in
                                 partially. Adds grit and edge back into the
                                 voice after the bandpass has narrowed it.
                                 The other main fix for flatness.
  5. Ring modulation (subtle) -> a very low mix of amplitude modulation at a
                                 slowly drifting low frequency. A fixed-
                                 frequency ring mod reads as a flat, static
                                 buzz; letting it drift keeps it feeling
                                 unstable instead of mechanical. Too much
                                 mix = cheap sci-fi robot; 3-6% just makes the
                                 voice sound slightly *wrong* in a way that's
                                 hard to name.
  6. Compression               -> broadcast audio is flattened dynamically,
                                 which reads as detachment. Kept gentle here
                                 (threshold 0.4, ratio 3.0) so it detaches the
                                 voice without crushing all the texture the
                                 previous two stages just added.
  7. Static bed                -> noise that breathes with the signal envelope,
                                 like real interference riding under speech,
                                 not a flat hiss layered on top.
  8. Dropouts                  -> short random attenuation events + occasional
                                 click. This is what makes it sound like it's
                                 "not really heard good" rather than just
                                 filtered — the signal is actively failing to
                                 arrive sometimes.

Deliberately avoided: pitch-up (reads as playful/chipmunk), heavy vocoder
robot buzz (reads as camp), reverb/room tone (reads as "present," the
opposite of distant).

Usage:
    python cogito_voice_fx.py input.wav output.wav

Dependencies: numpy, scipy
"""

import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, lfilter


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def load_wav(path: str):
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)  # fold to mono
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float64) / max_val
    return rate, data.astype(np.float64)


def save_wav(path: str, rate: int, signal: np.ndarray):
    signal = np.clip(signal, -1.0, 1.0)
    out = (signal * 32767).astype(np.int16)
    wavfile.write(path, rate, out)


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def bandpass(signal: np.ndarray, rate: int, low=300, high=3200, order=6):
    sos = butter(order, [low, high], btype="bandpass", fs=rate, output="sos")
    return sosfilt(sos, signal)


def compress(signal: np.ndarray, threshold=0.4, ratio=3.0, makeup=1.25):
    """Simple broadcast-style compressor: flatten dynamics, then boost."""
    sign = np.sign(signal)
    mag = np.abs(signal)
    over = mag > threshold
    compressed_mag = np.where(
        over,
        threshold + (mag - threshold) / ratio,
        mag,
    )
    return sign * compressed_mag * makeup


def wow_flutter(signal: np.ndarray, rate: int, depth=0.0035, speed_hz=0.35, seed=7):
    """
    Variable-rate resample: build a slowly, irregularly wobbling read-head
    position and interpolate the signal at that position. depth is the max
    fractional time-warp; speed_hz is roughly how fast the wobble drifts.
    """
    rng = np.random.default_rng(seed)
    n = len(signal)
    t = np.arange(n) / rate

    # Slow wobble = low-frequency sine + a slower random walk so it never
    # feels perfectly periodic (perfectly periodic reads as "effect", not
    # "unstable signal").
    wobble = np.sin(2 * np.pi * speed_hz * t)
    drift = np.cumsum(rng.normal(0, 1, n))
    drift = drift / (np.max(np.abs(drift)) + 1e-9)
    warp = depth * (0.7 * wobble + 0.3 * drift)

    read_pos = np.arange(n) + warp * rate
    read_pos = np.clip(read_pos, 0, n - 1)
    return np.interp(read_pos, np.arange(n), signal)


def ring_mod(signal: np.ndarray, rate: int, base_freq=45, wobble_hz=0.15,
             wobble_depth=6, mix=0.045):
    """
    A ring-mod carrier at a fixed frequency reads as a flat, static buzz —
    which is its own kind of "flat" even if the rest of the chain isn't.
    Letting the carrier frequency drift slowly makes the artifact sound
    unstable rather than mechanical.
    """
    n = len(signal)
    t = np.arange(n) / rate
    freq_t = base_freq + wobble_depth * np.sin(2 * np.pi * wobble_hz * t)
    phase = 2 * np.pi * np.cumsum(freq_t) / rate
    carrier = np.sin(phase)
    modulated = signal * carrier
    return (1 - mix) * signal + mix * modulated


def saturate(signal: np.ndarray, drive=2.4, mix=0.4):
    """
    Soft (tanh) saturation adds odd harmonics — this is what actually puts
    texture and edge back into the voice after the bandpass has narrowed it.
    Mixed at partial wet so it adds grit without turning into distortion.
    """
    driven = np.tanh(signal * drive) / np.tanh(drive)
    return (1 - mix) * signal + mix * driven


def metallic_resonance(signal: np.ndarray, rate: int, delay_ms=9.0,
                        feedback=0.35, mix=0.3):
    """
    A short feedback delay (comb filter) carves resonant peaks/notches into
    the spectrum instead of the flat, even rolloff a plain bandpass gives you.
    This is most of what makes a voice sound like it's resonating through
    something metallic/hollow rather than just "filtered."
    """
    delay_samples = max(1, int(rate * delay_ms / 1000.0))
    a = np.zeros(delay_samples + 1)
    a[0] = 1.0
    a[delay_samples] = -feedback
    wet = lfilter([1.0], a, signal)
    return (1 - mix) * signal + mix * wet


def static_bed(signal: np.ndarray, rate: int, level=0.02, seed=13):
    """
    Noise that rides the signal's own envelope (louder under speech, doesn't
    vanish in silence) rather than a constant hiss laid on top — that's what
    makes it read as interference on the line, not a de-noiser gone wrong.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, len(signal))

    # crude envelope: rolling abs-mean
    window = max(1, rate // 50)
    kernel = np.ones(window) / window
    envelope = np.convolve(np.abs(signal), kernel, mode="same")
    envelope = envelope / (np.max(envelope) + 1e-9)

    noise_floor = 0.3  # noise never fully disappears, even in gaps
    shaped_noise = noise * level * (noise_floor + (1 - noise_floor) * envelope)
    return signal + shaped_noise


def dropouts(signal: np.ndarray, rate: int, num_events=6, seed=21):
    """Short attenuation dips + the occasional hard click, placed randomly."""
    rng = np.random.default_rng(seed)
    out = signal.copy()
    n = len(signal)

    for _ in range(num_events):
        start = rng.integers(0, max(1, n - rate // 4))
        dur = rng.integers(int(rate * 0.03), int(rate * 0.18))
        end = min(n, start + dur)
        depth = rng.uniform(0.15, 0.75)  # how much signal survives the dip
        out[start:end] *= depth

        if rng.random() < 0.4:
            click_pos = min(n - 1, end)
            click_len = max(2, rate // 4000)
            out[click_pos:click_pos + click_len] += rng.uniform(-0.6, 0.6, click_len)

    return out


def normalize(signal: np.ndarray, peak=0.9):
    m = np.max(np.abs(signal)) + 1e-9
    return signal * (peak / m)


# --------------------------------------------------------------------------
# Chain
# --------------------------------------------------------------------------

def process_cogito_voice(signal: np.ndarray, rate: int) -> np.ndarray:
    x = signal.astype(np.float64)
    x = wow_flutter(x, rate, depth=0.005, speed_hz=0.5) # more instability
    x = bandpass(x, rate, low=150, high=3500)         # let more bass through for heaviness
    x = metallic_resonance(x, rate, feedback=0.5)     # harsher resonance
    x = saturate(x, drive=6.0, mix=0.8)               # heavy distortion/grit
    x = ring_mod(x, rate)             
    x = compress(x, threshold=0.15, ratio=8.0, makeup=2.0)  # squash hard for that detached, heavy broadcast feel
    x = static_bed(x, rate, level=0.04)               # more static
    x = dropouts(x, rate, num_events=14)              # aggressive failing
    x = normalize(x)
    return x


def main():
    if len(sys.argv) != 3:
        print("Usage: python cogito_voice_fx.py input.wav output.wav")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    rate, signal = load_wav(in_path)
    processed = process_cogito_voice(signal, rate)
    save_wav(out_path, rate, processed)
    print(f"Wrote {out_path} ({rate} Hz, {len(processed)/rate:.1f}s)")


if __name__ == "__main__":
    main()
