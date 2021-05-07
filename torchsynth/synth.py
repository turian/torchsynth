"""
:class:`~torchsynth.module.SynthModule` wired together form a modular synthesizer.
:class:`~torchsynth.synth.Voice` is our default synthesizer, and is used to
generate `synth1B1 <../reproducibility/synth1B1.html>`_.

We base off pytorch-lightning :class:`~pytorch_lightning.core.lightning.LightningModule`
because it makes `multi-GPU
<https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html>`_
inference easy. Nonetheless, you can treat each synth as a native
torch :class:`~torch.nn.Module`.
"""

import json
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import pkg_resources

if sys.version_info.major == 3 and sys.version_info.minor >= 8:
    from typing import OrderedDict as OrderedDictType
else:
    # pypi version typing_extensions doesn't yet supports OrderedDict (only master)
    # from typing_extensions import OrderedDict as OrderedDictType
    from typing import Dict as OrderedDictType

import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor as T

from torchsynth.config import N_BATCHSIZE_FOR_TRAIN_TEST_REPRODUCIBILITY, SynthConfig
from torchsynth.module import (
    ADSR,
    LFO,
    VCA,
    AudioMixer,
    ControlRateUpsample,
    ControlRateVCA,
    ModulationMixer,
    MonophonicKeyboard,
    Noise,
    SineVCO,
    SquareSawVCO,
    SynthModule,
)
from torchsynth.parameter import ModuleParameter
from torchsynth.signal import Signal


class AbstractSynth(LightningModule):
    """
    Base class for synthesizers that combine one or more
    :class:`~torchsynth.module.SynthModule` to create a full synth
    architecture.

    Args:
        synthconfig: Global configuration for this synth and all its
            component :class:`~torchsynth.module.SynthModule`. If none
            is provided, we use our defaults.
    """

    def __init__(self, synthconfig: Optional[SynthConfig] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if synthconfig is not None:
            self.synthconfig = synthconfig
        else:
            # Use the default
            self.synthconfig = SynthConfig()

    @property
    def batch_size(self) -> T:
        assert self.synthconfig.batch_size.ndim == 0
        return self.synthconfig.batch_size

    @property
    def sample_rate(self) -> T:
        assert self.synthconfig.sample_rate.ndim == 0
        return self.synthconfig.sample_rate

    @property
    def buffer_size(self) -> T:
        assert self.synthconfig.buffer_size.ndim == 0
        return self.synthconfig.buffer_size

    @property
    def buffer_size_seconds(self) -> T:
        assert self.synthconfig.buffer_size_seconds.ndim == 0
        return self.synthconfig.buffer_size_seconds

    def add_synth_modules(
        self, modules: List[Tuple[str, SynthModule, Optional[Dict[str, Any]]]]
    ):
        """
        Add a set of named children :class:`~torchsynth.module.SynthModule`
        to this synth.
        Registers them with the torch :class:`~torch.nn.Module` so that
        all parameters are recognized.

        Args:
            modules: A list of :class:`~torchsynth.module.SynthModule`
                classes with their names and any parameters to pass to
                their constructor.
        """

        for module_tuple in modules:
            if len(module_tuple) == 3:
                name, module, params = module_tuple
            else:
                name, module = module_tuple
                params = {}

            if not issubclass(module, SynthModule):
                raise TypeError(f"{module} is not a SynthModule")

            self.add_module(
                name, module(self.synthconfig, device=self.device, **params)
            )

    def get_parameters(
        self, include_frozen: bool = False
    ) -> OrderedDictType[Tuple[str, str], ModuleParameter]:
        """
        Returns a dictionary of
        :class:`~torchsynth.parameter.ModuleParameterRange` for this synth,
        keyed on a tuple of the :class:`~torchsynth.module.SynthModule` name
        and the parameter name.

        Args:
            include_frozen: If some parameter is frozen, return it anyway.
        """
        parameters = []

        # Each parameter in this synth will have a unique combination of module name
        # and parameter name -- create a dictionary keyed on that.
        for module_name, module in sorted(self.named_modules()):
            # Make sure this is a SynthModule, b/c we are using ParameterDict
            # and ParameterDict is a module, we get those returned as well
            # TODO: see https://github.com/torchsynth/torchsynth/issues/213
            if isinstance(module, SynthModule):
                for parameter in module.parameters():
                    if include_frozen or not ModuleParameter.is_parameter_frozen(
                        parameter
                    ):
                        parameters.append(
                            ((module_name, parameter.parameter_name), parameter)
                        )

        return OrderedDict(parameters)

    def set_parameters(
        self,
        params: Dict[Tuple[str, str], T],
        freeze: Optional[bool] = False,
        range0to1=False,
    ):
        """
        Set various :class:`~torchsynth.parameter.ModuleParameter` for this synth.

        Args:
            params: Module and parameter strings, with the corresponding value.
            freeze: Optionally, freeze these parameters to prevent further updates.
        """
        for (module_name, param_name), value in params.items():
            module = getattr(self, module_name)
            if range0to1:
                module.set_parameter_0to1(param_name, value.to(self.device))
            else:
                module.set_parameter(param_name, value.to(self.device))
            # Freeze this parameter at this value now if freeze is True
            if freeze:
                module.get_parameter(param_name).frozen = True

    def freeze_parameters(self, params: List[Tuple[str, str]]):
        """
        Freeze a set of parameters by passing in a tuple of the module and param name.
        """
        for module_name, param_name in params:
            module = getattr(self, module_name)
            module.get_parameter(param_name).frozen = True

    def unfreeze_all_parameters(self):
        """
        Unfreeze all parameters in this synth.
        """
        for param in self.parameters():
            if isinstance(param, ModuleParameter):
                param.frozen = False

    def output(self, *args: Any, **kwargs: Any) -> Signal:  # pragma: no cover
        """
        Each `AbstractSynth` should override this.
        """
        raise NotImplementedError("Derived classes must override this method")

    def forward(
        self, batch_idx: Optional[int] = None, *args: Any, **kwargs: Any
    ) -> Tuple[Signal, torch.Tensor, Union[torch.Tensor, None]]:  # pragma: no cover
        """
        Wrapper around `output`, which optionally randomizes the
        synth :class:`~torchsynth.parameter.ModuleParameter` values
        in a deterministic way, and optionally disables gradient
        computations. This all depends on
        :attr:`~torchsynth.synth.AbstractSynth.synthconfig`.

        Args:
            batch_idx: If provided, we generate this batch, in a
                deterministic random way, according to
                :attr:`~torchsynth.config.SynthConfig.batch_size`.
                If None (default), we just use the current
                module parameter settings.

        Returns:
            audio, parameters, is_train as a Tuple.

            (batch_size x buffer_size audio tensor,

            batch_size x n_parameters [0, 1] parameters tensor,

            batch_size Boolean tensor of is this example train [or test],
            None if batch_idx is None)
        """
        if self.synthconfig.reproducible and batch_idx is None:
            raise ValueError(
                "Reproducible mode is on, you must "
                "pass a batch index when calling this synth"
            )
        is_train = self._batch_idx_to_is_train(batch_idx)
        if self.synthconfig.no_grad:
            with torch.no_grad():
                if batch_idx is not None:
                    self.randomize(seed=batch_idx)
                params = torch.stack([p.data for p in self.parameters()], dim=1)
                return self.output(*args, **kwargs), params, is_train
        else:
            if batch_idx is not None:
                self.randomize(seed=batch_idx)
            params = torch.stack([p.data for p in self.parameters()], dim=1)
            return self.output(*args, **kwargs), params, is_train

    def _batch_idx_to_is_train(
        self, batch_idx: Union[None, int]
    ) -> Union[None, torch.tensor]:
        """
        Determine which samples are training examples if batch_idx is provided
        """
        if batch_idx is not None:
            idxs = torch.range(
                self.batch_size * batch_idx,
                self.batch_size * (batch_idx + 1) - 1,
                device=self.device,
            )
            assert len(idxs) == self.batch_size
            # As specified in our paper, the first 9x1024 samples
            # are train, and the next 1024 are test.
            is_train = (idxs // N_BATCHSIZE_FOR_TRAIN_TEST_REPRODUCIBILITY) % 10 != 9
        else:
            is_train = None
        return is_train

    def test_step(self, batch, batch_idx):
        """
        This is boilerplate required by pytorch-lightning
        :class:`~pytorch_lightning.LightningTrainer`
        when calling test.
        """
        return 0.0

    @property
    def hyperparameters(self) -> OrderedDictType[Tuple[str, str, str], Any]:
        """
        Returns a dictionary of curve and symmetry hyperparameter values keyed
        on a tuple of the module name, parameter name, and hyperparameter name
        """
        hparams = []
        for (module_name, parameter_name), parameter in self.get_parameters().items():
            hparams.append(
                (
                    (module_name, parameter_name, "curve"),
                    parameter.parameter_range.curve,
                )
            )
            hparams.append(
                (
                    (module_name, parameter_name, "symmetric"),
                    parameter.parameter_range.symmetric,
                )
            )

        return OrderedDict(hparams)

    def set_hyperparameter(self, hyperparameter: Tuple[str, str, str], value: Any):
        """
        Set a hyperparameter. Pass in the module name, parameter name, and
        hyperparameter to set, and the value to set it to.
        """
        module = getattr(self, hyperparameter[0])
        parameter = module.get_parameter(hyperparameter[1])
        assert not ModuleParameter.is_parameter_frozen(parameter)
        setattr(parameter.parameter_range, hyperparameter[2], value)

    def save_hyperparameters(self, filename: str, indent=True) -> None:
        """
        Save hyperparameters to a JSON file
        """
        # Render all hyperparameters as JSON
        hp = [{"name": key, "value": val} for key, val in self.hyperparameters.items()]
        with open(os.path.abspath(filename), "w") as fp:
            json.dump(hp, fp, indent=indent)

    def load_hyperparameters(self, nebula: str) -> None:
        """
        Load hyperparameters from a JSON file

        Args:
            nebula: nebula to load. This can either be the name of a nebula that is
                included in torchsynth, or the filename of a nebula json file to load.

        TODO add nebula list in docs
        See https://github.com/torchsynth/torchsynth/issues/324
        """

        # Try to load nebulae from package resources, otherwise, try
        # to load from a filename
        try:
            synth = type(self).__name__.lower()
            nebulae_str = f"nebulae/{synth}/{nebula}.json"
            data = pkg_resources.resource_string(__name__, nebulae_str)
            hyperparameters = json.loads(data)
        except FileNotFoundError:
            with open(os.path.abspath(nebula), "r") as fp:
                hyperparameters = json.load(fp)

        # Update all hyperparameters in this synth
        for hp in hyperparameters:
            self.set_hyperparameter(hp["name"], hp["value"])

    def randomize(self, seed: Optional[int] = None):
        """
        Randomize all parameters
        """
        parameters = [param for _, param in sorted(self.named_parameters())]
        if seed is not None:
            # Generate batch_size x parameter number of random values
            # Reseed the random number generator for every item in the batch
            cpu_rng = torch.Generator(device="cpu")
            new_values = []
            for i in range(self.batch_size):
                cpu_rng.manual_seed(seed * self.batch_size.numpy().item() + i)
                new_values.append(
                    torch.rand((len(parameters),), device="cpu", generator=cpu_rng)
                )

            # Move to device if necessary
            new_values = torch.stack(new_values, dim=1)
            if self.device.type != "cpu":
                new_values = new_values.pin_memory().to(self.device, non_blocking=True)

            # Set parameter data
            for i, parameter in enumerate(parameters):
                if not ModuleParameter.is_parameter_frozen(parameter):
                    parameter.data = new_values[i]
        else:
            assert not self.synthconfig.reproducible
            for parameter in parameters:
                if not ModuleParameter.is_parameter_frozen(parameter):
                    parameter.data.uniform_(0, 1)

        # Add seed to all modules
        for module in self._modules:
            self._modules[module].seed = seed

    def on_post_move_to_device(self) -> None:
        """
        LightningModule trigger after this Synth has been moved to a different device.
        Use this to update children SynthModules device settings
        """
        self.synthconfig.to(self.device)
        for module in self.modules():
            if isinstance(module, SynthModule):
                # TODO look into performance of calling to instead
                module._update_device(self.device)


class Voice(AbstractSynth):
    """
    The default configuration in torchsynth is the Voice, which is
    the architecture used in synth1B1. The Voice architecture
    comprises the following modules: a
    :class:`~torchsynth.module.MonophonicKeyboard`, two
    :class:`~torchsynth.module.LFO`, six :class:`~torchsynth.module.ADSR`
    envelopes (each :class:`~torchsynth.module.LFO` module includes
    two dedicated :class:`~torchsynth.module.ADSR`: one for rate
    modulation and another for amplitude modulation), one
    :class:`~torchsynth.module.SineVCO`, one
    :class:`~torchsynth.module.SquareSawVCO`, one
    :class:`~torchsynth.module.Noise` generator,
    :class:`~torchsynth.module.VCA`, a
    :class:`~torchsynth.module.ModulationMixer` and an
    :class:`~torchsynth.module.AudioMixer`. Modulation signals
    generated from control modules (:class:`~torchsynth.module.ADSR`
    and :class:`~torchsynth.module.LFO`) are upsampled to the audio
    sample rate before being passed to audio rate modules.

    You can find a diagram of Voice in `Synth Architectures documentation
    <../modular-design/modular-principles.html#synth-architectures>`_.
    """

    def __init__(
        self,
        synthconfig: Optional[SynthConfig] = None,
        nebula: Optional[str] = "default",
        *args,
        **kwargs,
    ):
        AbstractSynth.__init__(self, synthconfig=synthconfig, *args, **kwargs)

        # Register all modules as children
        self.add_synth_modules(
            [
                ("keyboard", MonophonicKeyboard),
                ("adsr_1", ADSR),
                ("adsr_2", ADSR),
                ("lfo_1", LFO),
                ("lfo_2", LFO),
                ("lfo_1_amp_adsr", ADSR),
                ("lfo_2_amp_adsr", ADSR),
                ("lfo_1_rate_adsr", ADSR),
                ("lfo_2_rate_adsr", ADSR),
                ("control_vca", ControlRateVCA),
                ("control_upsample", ControlRateUpsample),
                (
                    "mod_matrix",
                    ModulationMixer,
                    {
                        "n_input": 4,
                        "n_output": 5,
                        "input_names": ["adsr_1", "adsr_2", "lfo_1", "lfo_2"],
                        "output_names": [
                            "vco_1_pitch",
                            "vco_1_amp",
                            "vco_2_pitch",
                            "vco_2_amp",
                            "noise_amp",
                        ],
                    },
                ),
                ("vco_1", SineVCO),
                ("vco_2", SquareSawVCO),
                ("noise", Noise, {"seed": 13}),
                ("vca", VCA),
                (
                    "mixer",
                    AudioMixer,
                    {
                        "n_input": 3,
                        "curves": [1.0, 1.0, 0.025],
                        "names": ["vco_1", "vco_2", "noise"],
                    },
                ),
            ]
        )

        # Load the nebula
        self.load_hyperparameters(nebula)

    def output(self) -> T:
        # The convention for triggering a note event is that it has
        # the same note_on_duration for both ADSRs.
        midi_f0, note_on_duration = self.keyboard()

        # ADSRs for modulating LFOs
        lfo_1_rate = self.lfo_1_rate_adsr(note_on_duration)
        lfo_2_rate = self.lfo_2_rate_adsr(note_on_duration)
        lfo_1_amp = self.lfo_1_amp_adsr(note_on_duration)
        lfo_2_amp = self.lfo_2_amp_adsr(note_on_duration)

        # Compute LFOs with envelopes
        lfo_1 = self.control_vca(self.lfo_1(lfo_1_rate), lfo_1_amp)
        lfo_2 = self.control_vca(self.lfo_2(lfo_2_rate), lfo_2_amp)

        # ADSRs for Oscillators and noise
        adsr_1 = self.adsr_1(note_on_duration)
        adsr_2 = self.adsr_2(note_on_duration)

        # Mix all modulation signals
        (vco_1_pitch, vco_1_amp, vco_2_pitch, vco_2_amp, noise_amp) = self.mod_matrix(
            adsr_1, adsr_2, lfo_1, lfo_2
        )

        # Create signal and with modulations and mix together
        vco_1_out = self.vca(
            self.vco_1(midi_f0, self.control_upsample(vco_1_pitch)),
            self.control_upsample(vco_1_amp),
        )
        vco_2_out = self.vca(
            self.vco_2(midi_f0, self.control_upsample(vco_2_pitch)),
            self.control_upsample(vco_2_amp),
        )
        noise_out = self.vca(self.noise(), self.control_upsample(noise_amp))

        return self.mixer(vco_1_out, vco_2_out, noise_out)
