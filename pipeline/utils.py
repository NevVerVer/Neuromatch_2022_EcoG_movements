from typing import List, Union

import mne
import numpy as np


def make_mne_epochs(data: mne.io.RawArray,
                    begin_times: Union[List[int], np.ndarray], tmin: float=-0.2,
                    tmax: float=0.5) -> mne.Epochs:
    """
    Make epochs from raw data.

    Parameters:
        data: mne.io.RawArray
            Raw data.
        begin_times: Any(List[int], np.ndarray)
            List of begin times in milliseconds.
        tmin: float
            Start time of epochs.
        tmax: float
            End time of epochs.

    Returns:
        epochs: mne.Epochs
    """
    # check if begin_times are smaller than max duration of data
    assert max(begin_times) < data.times.shape[0]

    # create events
    events = np.empty((len(begin_times), 3))
    events[:, 0] = np.array(begin_times)
    events[:, 1] = 0
    events[:, 2] = 1
    events = events.astype(int)

    # create epochs
    epochs = mne.Epochs(data, events=events, tmin=tmin, tmax=tmax,
                        preload=True)
    return epochs


def make_epochs_psd(epochs: mne.Epochs):
    psds, freqs = mne.time_frequency.psd_multitaper(epochs, fmin=0, fmax=150,)
    return psds, freqs
