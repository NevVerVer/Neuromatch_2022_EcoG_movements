import pandas as pd
import pytest
from mne.io.meas_info import create_info
import numpy as np
import mne

from pipeline.utils import prepare_tfr_data


def test_prepare_tfr_data():
    n_channels = 2
    t_min = -5
    t_max = 5
    sfreq = 500
    n_trials = 10
    decim = 50
    pad_val = 0.5
    info = create_info(n_channels, ch_types='ecog', sfreq=sfreq)
    epochs = mne.EpochsArray(
        np.random.random((n_trials, n_channels, (t_max - t_min) * sfreq + 1)),
        info, tmin=t_min)

    epochs.metadata = pd.DataFrame(
        {'false_pos': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})

    freqs = np.arange(1, 124, 10)
    epochs_tfr = prepare_tfr_data(epochs, freqs=freqs, decim=decim,
                                  pad_val=pad_val)
    assert len(epochs_tfr.data.shape) == 4
    assert epochs_tfr.data.shape[0] == n_trials
    assert epochs_tfr.data.shape[1] == n_channels
    assert epochs_tfr.data.shape[2] == len(freqs)
    # NOTE: add 1
    assert epochs_tfr.data.shape[3] == (t_max - t_min - pad_val * 4) * \
           (sfreq / decim) + 1
