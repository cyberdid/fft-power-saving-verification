"""Deterministic synthetic IQ and explicitly normalized comparison metrics."""
import numpy as np


CASE_DESCRIPTIONS = {
    'impulse': 'Unit impulse at the midpoint; survives Hann and has flat magnitude DFT.',
    'bin_tone': 'Complex unit tone at integer FFT bin 7.',
    'off_bin_tone': 'Complex tone at fractional bin 7.37; leakage.',
    'weak_near_strong': 'Unit tone at bin 7 plus amplitude 1e-5 tone at bin 10.23.',
    'white_noise': 'Complex Gaussian noise, expected mean power 0.01.',
    'chirp': 'Complex chirp from -0.35 to +0.35 cycles/sample.',
    'qpsk_bursts': 'Synthetic rectangular QPSK symbols with periodic silent gaps; not LTE/Wi-Fi.',
    'dc_and_spurs': 'DC plus tone, conjugate image and small spur.',
    'clipped_multitone': 'I/Q components independently clipped to [-1,1].',
}


def make_cases(n, seed=20261010):
    if not isinstance(n, int) or n < 16:
        raise ValueError('synthetic suite requires integer n >= 16')
    rng = np.random.default_rng(np.random.SeedSequence([seed, n]))
    t = np.arange(n)
    tone = lambda k: np.exp(2j*np.pi*k*t/n)
    impulse = np.zeros(n, dtype=np.complex128)
    impulse[n//2] = 1
    chirp_phase = -.35*t + .35*t*t/max(n-1, 1)
    symbols = np.exp(1j*(np.pi/4+np.pi/2*rng.integers(0, 4, (n+7)//8)))
    burst = np.repeat(symbols, 8)[:n]
    burst[(t % 64) >= 48] = 0
    multitone = 1.4*tone(7)+.8*tone(-5.3)
    return {
        'impulse': impulse,
        'bin_tone': tone(7),
        'off_bin_tone': .7*tone(7.37),
        'weak_near_strong': tone(7)+1e-5*tone(10.23),
        'white_noise': .1/np.sqrt(2)*(rng.standard_normal(n)+1j*rng.standard_normal(n)),
        'chirp': .8*np.exp(2j*np.pi*chirp_phase),
        'qpsk_bursts': .6*burst*tone(3.25),
        'dc_and_spurs': .3+.7*tone(7)+.03*tone(-7)+.002*tone(11.2),
        'clipped_multitone': np.clip(multitone.real, -1, 1)+1j*np.clip(multitone.imag, -1, 1),
    }


def hann(n):
    return .5-.5*np.cos(2*np.pi*np.arange(n)/n)


def periodogram(transform, sample_rate, window):
    if sample_rate <= 0 or len(transform) != len(window) or np.sum(window**2) <= 0:
        raise ValueError('positive Fs and matching nonzero window required')
    # Two-sided periodogram, no extra factor two for complex IQ.
    # Units: sample-amplitude squared / Hz, not calibrated physical dBm.
    return np.abs(transform)**2/(sample_rate*np.sum(window**2))


def compare(actual, reference, sample_rate, window):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or not np.all(np.isfinite(reference)):
        raise ValueError('same-shape finite reference required')
    if not np.all(np.isfinite(actual)):
        return {'all_finite': False, 'relative_l2_error': None, 'max_abs_error': None,
                'psd_max_delta_db': None, 'peak_bin_matches': False}
    delta = actual-reference
    scale = float(np.linalg.norm(reference))
    error = float(np.linalg.norm(delta))
    ref_psd = periodogram(reference, sample_rate, window)
    psd = periodogram(actual, sample_rate, window)
    mask = ref_psd > max(float(np.max(ref_psd))*1e-10, np.finfo(float).tiny)
    db_error = (float(np.max(np.abs(10*np.log10(np.maximum(psd[mask], np.finfo(float).tiny))
                                    -10*np.log10(ref_psd[mask])))) if np.any(mask) else 0.0)
    return {'all_finite': True, 'relative_l2_error': error/scale if scale else error,
            'max_abs_error': float(np.max(np.abs(delta))), 'psd_max_delta_db': db_error,
            'peak_bin_matches': bool(np.argmax(psd) == np.argmax(ref_psd))}
