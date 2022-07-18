import pytest
from mne.io.meas_info import create_info
import numpy as np
import mne

from pipeline.utils import make_mne_epochs


def test_make_mne_epochs():
    n_channels = 1
    t_min = -0.2
    t_max = 0.5
    sfreq = 500
    info = create_info(n_channels, sfreq=sfreq)
    data = np.random.random((n_channels, 3000))
    raw = mne.io.RawArray(data, info)
    begin_times = [500, 1000, 1500]
    epochs = make_mne_epochs(raw, begin_times)
    assert epochs.events.shape[0] == len(begin_times)
    assert epochs.get_data().shape[0] == len(begin_times)
    assert epochs.get_data().shape[1] == n_channels
    # NOTE: add 1
    assert epochs.get_data().shape[2] == (t_max - t_min) * sfreq + 1

