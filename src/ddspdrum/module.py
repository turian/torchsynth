"""
Synth modules.

    TODO    :   - VCA is slow. Blocks? Not sure what's best for tensors.
                - Convert operations to tensors, obvs.
"""

import numpy as np
from scipy.signal import resample

from ddspdrum.defaults import CONTROL_RATE, SAMPLE_RATE, EPSILON


class SynthModule:
    """
    Base class for synthesis modules. Mostly helper functions for the moment.
    """

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.control_rate = CONTROL_RATE

        # TODO: this doesn't need to be here.
        self.eps = EPSILON

    def control_to_sample_rate(self, control: np.array) -> np.array:
        """
        Resample a control signal to the sample rate
        TODO: One thing I worry about with all these casts to
        ints back and forth between sample and control rate is
        off-by-one errors.
        I'm beginning to believe we should standardize as much
        as possible on nsamples in most places instead of
        float duration in seconds, and just convert to ncontrol
        when needed..
        """
        # Right now it appears that all signals are 1d, but later
        # we'll probably convert things to 2d: instance x signal

        if self.control_rate == self.sample_rate:
            return control
        else:
            assert control.ndim == 1
            num_samples = int(round(len(control) * self.sample_rate / self.control_rate))
            return resample(control, num_samples)

    # TODO: move to "utils."
    def hz_to_midi(self, hz):
        return 12 * np.log2((hz + self.eps) / 440) + 69

    # TODO: move to "utils."
    @staticmethod
    def midi_to_hz(midi):
        return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

    # TODO: move to "utils."
    @staticmethod
    def fix_length(signal: np.array, length: int) -> np.array:
        # Right now it appears that all signals are 1d, but later
        # we'll probably convert things to 2d: instance x signal
        assert signal.ndim == 1
        if len(signal) < length:
            signal = np.pad(signal, [0, length - len(signal)])
        elif len(signal) > length:
            signal = signal[:length]
        assert signal.shape == (length,)
        return signal


class ADSR(SynthModule):
    """
    Envelope class for building a control rate ADSR signal
    """

    def __init__(
        self,
        a: float = 0.25,
        d: float = 0.25,
        s: float = 0.5,
        r: float = 0.5,
        alpha: float = 3.0,
    ):
        """
        Parameters
        ----------
        a                   :   attack time (sec), >= 0
        d                   :   decay time (sec), >= 0
        s                   :   sustain amplitude between 0-1. The only part of
                                ADSR that (confusingly, by convention) is not in
                                the seconds domain.
        r                   :   release time (sec), >= 0
        alpha               :   envelope curve, >= 0. 1 is linear, >1 is exponential.
        """

        super().__init__()
        assert alpha >= 0
        self.alpha = alpha

        assert s >= 0 and s <= 1
        self.s = s

        assert a >= 0
        self.a = a

        assert d >= 0
        self.d = d

        assert r >= 0
        self.r = r

    def __call__(self, sustain_duration=0):
        """ Generate an envelope that sustains for a given duration in seconds.

        Generates a control-rate envelope signal with given attack, decay and release times, sustained for
        `sustain_duration` in seconds. E.g., an envelope with no attack or decay, a sustain duration of 1 and a 0.5
        release will last for 1.5 seconds.

        """

        assert sustain_duration >= 0
        return np.concatenate(
            (
                self.note_on,
                self.sustain(sustain_duration),
                self.note_off,
            )
        )

    def _ramp(self, duration: float):
        """ Makes a ramp of a given duration in seconds, returned at control rate.

        This function is used for the piece-wise construction of the envelope signal. Its output monotonically
        increases from 0 to 1. As a result, each component of the envelope is a scaled and possibly reversed version of
        this ramp:

        attack      -->     returns an `a`-length ramp, as is.
        decay       -->     `d`-length reverse ramp, scaled and shifted to descend from 1 to `s`.
        release     -->     `r`-length reverse ramp, scaled and shifted to descend from `s` to 0.

        Its curve is determined by alpha:

        alpha = 1 --> linear,
        alpha > 1 --> exponential,
        alpha < 1 --> logarithmic.

        """

        t = np.linspace(
            0, duration, int(round(duration * self.control_rate)), endpoint=False
        )
        return (t / duration) ** self.alpha

    @property
    def attack(self):
        return self._ramp(self.a)

    @property
    def decay(self):
        # `d`-length reverse ramp, scaled and shifted to descend from 1 to `s`.
        return self._ramp(self.d)[::-1] * (1 - self.s) + self.s

    def sustain(self, duration):
        return np.full(round(int(duration * CONTROL_RATE)), fill_value=self.s)

    @property
    def release(self):
        # `r`-length reverse ramp, scaled and shifted to descend from `s` to 0.
        return self._ramp(self.r)[::-1] * self.s

    @property
    def note_on(self):
        return np.append(self.attack, self.decay)

    @property
    def note_off(self):
        return self.release

    def __str__(self):
        return (
            f"ADSR(a={self.a}, d={self.d}, s={self.s}, r={self.r}, alpha={self.alpha})"
        )


class VCO(SynthModule):
    """
    Voltage controlled oscillator. Is called with control-rate pitch modulation, outputs audio.

    Think of this as a VCO on a modular synthesizer. It has a base pitch (specified here as a midi value), and a pitch
    modulation depth. Its call accepts a control-rate modulation signal between 0 - 1. An array of 0's returns a
    stationary audio signal at its base pitch.


    Parameters
    ----------

    midi_f0 (flt)       :       pitch value in 'midi' (69 = 440Hz).
    mod_depth (flt)     :       depth of the pitch modulation; 0 means no modulation.

    TODO:   - more than just cosine.

    Examples
    --------

    >>> myVCO = VCO(midi_f0=69, mod_depth=24)
    >>> two_8ve_chirp = myVCO(np.linspace(0, 1, 1000, endpoint=False))
    """

    def __init__(
        self, midi_f0: float = 69, mod_depth: float = 1, phase: float = 0
    ):
        super().__init__()

        assert 0 <= midi_f0 <= 127
        self.midi_f0 = midi_f0

        assert mod_depth >= 0
        self.mod_depth = mod_depth

        self.phase = phase

    def __call__(self, mod_signal: np.array, phase: float) -> np.array:
        """ Generates audio signal from control-rate mod.

        There are three representations of the 'pitch' at play here: (1) midi, (2) instantaneous frequency, and
        (3) phase, a.k.a. 'argument'.

        (1) midi    This is an abuse of the standard midi convention, where semitone pitches are mapped from 0 - 127.
                    Here it's a convenient way to represent pitch linearly. An A above middle C is midi 69.

        (2) freq    Pitch scales logarithmically in frequency. An A above middle C is 440Hz.

        (3) phase   This is the argument of the cosine function that generates sound. Frequency is the first derivative
                    of phase; phase is integrated frequency (~ish).

        First we generate the 'pitch contour' of the signal in midi values (mod contour + base pitch). Then we convert
        to a phase argument (via frequency), then output sound.

        """

        assert (mod_signal >= 0).all() and (mod_signal <= 1).all()

        control_as_midi = self.mod_depth * mod_signal + self.midi_f0
        control_as_frequency = self.midi_to_hz(control_as_midi)
        cosine_argument = self.make_argument(control_as_frequency) + phase

        self.phase = cosine_argument[-1]
        return np.cos(cosine_argument)

    def make_argument(self, control_as_frequency: np.array):
        """ Generates the phase argument to feed a cosine function in order to generate audio.
        """

        up_sampled = self.control_to_sample_rate(control_as_frequency)
        return np.cumsum(2 * np.pi * up_sampled / SAMPLE_RATE)


class VCA(SynthModule):
    """
    Voltage controlled amplifier. Shapes amplitude of audio rate signal with control rate level.
    """

    def __init__(self):
        super().__init__()
        self.__envelope = np.array([])
        self.__audio = np.array([])

    def __call__(self, envelopecontrol: np.array, audiosample: np.array):
        envelopecontrol = np.clip(envelopecontrol, 0, 1)
        audiosample = np.clip(audiosample, -1, 1)
        amp = self.control_to_sample_rate(envelopecontrol)
        signal = self.fix_length(audiosample, len(amp))
        return amp * signal


class SVF(SynthModule):
    """
    A State Variable Filter that can do lowpass, highpass, bandpass, and bandreject
    filtering. Allows modulation of the cutoff frequency and an adjustable resonance
    parameter. Can self-oscillate to make a sine / cosine wave.
    """

    def __init__(
        self,
        mode: str = "LPF",
        cutoff: float = 1000,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.mode = mode
        self.cutoff = cutoff
        self.resonance = resonance
        self.self_oscillate = self_oscillate

    def __call__(
        self,
        audio: np.ndarray,
        cutoff_mod: np.ndarray = None,
        cutoff_mod_amount: float = 0.0,
    ) -> np.ndarray:
        """
        Process audio samples
        """

        h = np.zeros(2)
        y = np.zeros_like(audio)

        # Calculate initial coefficients
        coeff0, coeff1, rho = self.calculate_coefficients(self.cutoff)

        # Check if there is a filter cutoff envelope to apply
        apply_modulation = False
        if cutoff_mod is not None and cutoff_mod_amount != 0.0:
            # Cutoff modulation must be same length as audio input
            assert len(cutoff_mod) == len(audio)
            apply_modulation = True

        # Processing loop
        for i in range(len(audio)):

            # If there is a cutoff modulation envelope, update coefficients
            if apply_modulation:
                cutoff = self.cutoff + cutoff_mod[i] * cutoff_mod_amount
                coeff0, coeff1, rho = self.calculate_coefficients(
                    self.cutoff + cutoff_mod[i] * cutoff_mod_amount
                )

            # Calculate each of the filter components
            hpf = coeff0 * (audio[i] - rho * h[0] - h[1])
            bpf = coeff1 * hpf + h[0]
            lpf = coeff1 * bpf + h[1]

            # Feedback samples
            h[0] = coeff1 * hpf + bpf
            h[1] = coeff1 * bpf + lpf

            if self.mode == "LPF":
                y[i] = lpf
            elif self.mode == "BPF":
                y[i] = bpf
            elif self.mode == "BSF":
                y[i] = hpf + lpf
            else:
                y[i] = hpf

        return y

    def calculate_coefficients(self, cutoff: float) -> (float, float, float):
        """
        Calculates the filter coefficients for SVF given a cutoff frequency
        """

        g = np.tan(np.pi * cutoff / self.sample_rate)
        R = 0.0 if self.self_oscillate else 1.0 / (2.0 * self.resonance)
        coeff0 = 1.0 / (1.0 + 2.0 * R * g + g * g)
        coeff1 = g
        rho = 2.0 * R + g

        return coeff0, coeff1, rho


class FIR(SynthModule):
    """
    A Finite Impulse Response filter
    """

    def __init__(
        self,
        cutoff: float = 1000,
        filter_length: int = 512,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.filter_length = filter_length
        self.cutoff = cutoff

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Filter audio samples
        TODO: Cutoff frequency modulation, if there is an efficient way to do it
        """

        impulse = self.windowed_sinc(self.cutoff, self.filter_length)
        y = np.convolve(audio, impulse)
        return y

    def windowed_sinc(self, cutoff: float, filter_length: float) -> np.ndarray:
        """
        Calculates the impulse response for FIR lowpass filter using the
        windowed sinc function method
        """

        ir = np.zeros(filter_length + 1)
        omega = 2 * np.pi * cutoff / self.sample_rate

        for i in range(filter_length + 1):
            n = i - filter_length / 2
            if n != 0:
                ir[i] = np.sin(n * omega) / n
            else:
                ir[i] = omega

            window = (
                0.42
                - 0.5 * np.cos(2 * np.pi * i / filter_length)
                + 0.08 * np.cos(2 * np.pi * i / filter_length)
            )
            ir[i] *= window

        ir /= omega

        return ir


class MovingAverage(SynthModule):
    """
    A finite impulse response moving average filter
    """

    def __init__(self, filter_length: int = 32, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.filter_length = filter_length

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Filter audio samples
        """

        impulse = np.ones(self.filter_length) / self.filter_length
        y = np.convolve(audio, impulse)
        return y