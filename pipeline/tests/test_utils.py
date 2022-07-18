import pytest
from mne.io.meas_info import create_info
import numpy as np
import mne

from pipeline.utils import make_mne_epochs, make_epochs_tfr


def test_make_mne_epochs():
    n_channels = 1
    t_min = -0.2
    t_max = 0.5
    sfreq = 500
    info = create_info(n_channels, ch_types='ecog', sfreq=sfreq)
    data = np.random.random((n_channels, 3000))
    raw = mne.io.RawArray(data, info)
    begin_times = [500, 1000, 1500]
    epochs = make_mne_epochs(raw, begin_times)
    assert epochs.events.shape[0] == len(begin_times)
    assert epochs.get_data().shape[0] == len(begin_times)
    assert epochs.get_data().shape[1] == n_channels
    # NOTE: add 1
    assert epochs.get_data().shape[2] == (t_max - t_min) * sfreq + 1


def test_make_epochs_tfr():
    n_channels = 2
    t_min = -0.5
    t_max = 1.0
    sfreq = 500
    info = create_info(n_channels, ch_types='ecog', sfreq=sfreq)
    data = np.random.random((n_channels, 3000))
    raw = mne.io.RawArray(data, info)
    begin_times = [500, 1000, 1500]
    epochs = make_mne_epochs(raw, begin_times, t_min, t_max)
    freqs = np.logspace(*np.log10([10, 150]), num=8)
    epochs_tfr = make_epochs_tfr(epochs, freqs)
    assert len(epochs_tfr.data.shape) == 4
    assert epochs_tfr.data.shape[0] == len(begin_times)
    assert epochs_tfr.data.shape[1] == n_channels
    assert epochs_tfr.data.shape[2] == len(freqs)
    # NOTE: add 1
    assert epochs_tfr.data.shape[3] == (t_max - t_min) * sfreq + 1
