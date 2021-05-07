# -*- coding: utf-8 -*-
# # lightningsynthmatch
#
# Attempt to match a 909 Snare sound to minimize multi-scale spectral loss, using Optuna.
#
# Unfortunately, I believe Optuna restricts us to [batch size=1](https://github.com/optuna/optuna/issues/2626).
#
# This currently only has been tested with one GPU, but Optuna should support multiprocessing. We just haven't implemented it.
#
# TODO: Actually move stuff to CUDA if it exists.

# +
import json
import multiprocessing
from typing import Any

import auraloss
import IPython.display as ipd
import optuna
import pytorch_lightning as pl
import soundfile as sf
import torch
from IPython.core.display import display
from optuna.visualization import plot_optimization_history
from torchsynth.config import SynthConfig
from torchsynth.synth import Voice

# -

synthconfig = SynthConfig(batch_size=1, reproducible=False)
voice = Voice(synthconfig)

# +
target_sound, sr = sf.read("909-snare-4sec.ogg")
# This won't work with multi-GPU :(
device = "cuda" if torch.cuda.is_available() else "cpu"
target_sound = torch.tensor(target_sound, device=device)
# Normalize, but not entirely sure this is necessary
target_sound_max = torch.max(torch.abs(target_sound))
print("Normalizing target sound by", target_sound_max)
target_sound /= target_sound_max
assert sr == synthconfig.sample_rate
assert target_sound.ndim == 1
assert target_sound.shape[0] == synthconfig.buffer_size

display(ipd.Audio(target_sound, rate=int(voice.sample_rate.numpy().item())))


# +


def print_best_jsonl(study):
    for name, value in study.best_params.items():
        print(json.dumps([name, value]))


class RunningBestCallback:
    def __init__(self):
        pass

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if study.best_trial.number == trial.number:
            fig = plot_optimization_history(study)
            fig.show()
            print("New best value", study.best_value)
            print("New best trial", study.best_trial.number)
            display(
                ipd.Audio(
                    voice()[0].numpy(), rate=int(voice.sample_rate.numpy().item())
                )
            )
            print("New best params")
            print_best_jsonl(study)
            print("\n")


# +
mrstft = auraloss.freq.MultiResolutionSTFTLoss()


def objective(trial):
    params = voice.get_parameters(include_frozen=True)
    for param_name in list(params.keys()):
        value = trial.suggest_float(json.dumps(param_name), 0, 1)
        params[param_name] = torch.tensor([value])

    voice.set_parameters(params, freeze=True, range0to1=True)
    # print(dict(voice.get_parameters()))
    # https://github.com/optuna/optuna/issues/2626
    assert voice.batch_size == 1
    batchsounds = voice()
    assert len(batchsounds) == 1
    sound = batchsounds[0]
    loss = mrstft(sound, target_sound)
    return loss


# +
# Make optuna less chatty
optuna.logging.set_verbosity(optuna.logging.WARNING)

sampler = optuna.samplers.CmaEsSampler(n_startup_trials=2000)
study = optuna.create_study(sampler=sampler)  # , study_name=study_name)
# -

study.optimize(
    objective, n_trials=10000, callbacks=[RunningBestCallback()], show_progress_bar=True
)
fig = plot_optimization_history(study)
fig.show()
