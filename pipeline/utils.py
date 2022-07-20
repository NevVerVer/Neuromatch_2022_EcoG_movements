from typing import List, Union
from pathlib import Path

import mne
from mne.time_frequency import tfr_morlet
import numpy as np


def prepare_tfr_data(epochs: Union[mne.Epochs, mne.EpochsArray],
                     pad_val: float = 0.5,
                     freqs: np.ndarray = np.arange(1, 124, 5), decim: int = 50,
                     baseline_vals: List[float] = (-1.5, -1),
                     ) -> mne.time_frequency.EpochsTFR:
    """
    Function to prepare TFR data for statistical testing.
    Copied from: https://github.com/stepeter/naturalistic_arm_movements_ecog/blob/master/compute_power_gen_figs_python/tfr_utils.py

    Parameters:
        epochs: mne.Epochs
            Epochs.
        pad_val: float
            Value to pad the start and end of the epoch.
        freqs: np.ndarray
            Frequencies to compute TFRs.
        decim: int
            Decimation factor.
        baseline_vals: List[float]
            Baseline values to subtract.

    Returns:
        power: mne.time_frequency.EpochsTFR
            Baseline-subtracted TFRs
            shape: trials x channels x freqs x
            (t_max - t_min - pad_val * 4) * (sfreq / decim)
            NOTE: pad_val is subtracted twice

    """
    # remove padding from times
    epoch_times = [epochs.times.min() + pad_val, epochs.times.max() - pad_val]

    bad_chans = epochs.info['bads'].copy()  # store for later
    epochs.info['bads'] = []

    # Remove false positive events
    if 'false_pos' in epochs.metadata.columns:
        bad_ev_inds = np.nonzero(epochs.metadata['false_pos'].values)[0]
        epochs.drop(bad_ev_inds)

    # Compute TFR
    power = compute_tfr(epochs, epoch_times, freqs=freqs, crop_val=pad_val,
                        decim=decim)

    # add in metadata
    power._metadata = epochs.metadata.copy()

    # Calculate and subtract baseline (for each channel)
    power_ave_masked = power.copy()
    baseidx = np.nonzero(np.logical_and(power.times >= baseline_vals[0],
                                        power.times <= baseline_vals[1]))[0]
    for chan in range(power.data.shape[1]):
        curr_power = tfr_subtract_baseline(power, chan, baseidx,
                                           compute_mean=True)
        curr_masked_power_ave = np.copy(curr_power)

        # Return masked data to original variable
        curr_masked_power_ave = np.moveaxis(curr_masked_power_ave, -1, 0)
        power_ave_masked.data[:, chan, ...] = curr_masked_power_ave
        del curr_masked_power_ave, curr_power

    power_ave_masked.info['bads'] = bad_chans.copy()

    return power_ave_masked


def compute_tfr(epochs: mne.Epochs, epoch_times: List[float],
                freqs: np.ndarray = np.arange(6, 123, 3),
                crop_val: float = 0.5, decim: float = 30):
    """
    Function to compute TFRs from epochs.
    Copied from: https://github.com/stepeter/naturalistic_arm_movements_ecog/blob/master/compute_power_gen_figs_python/tfr_utils.py

    Parameters:
        epochs: mne.Epochs
            Epochs.
        epoch_times: np.ndarray
            Times to compute TFRs.
        freqs: np.ndarray
            Frequencies to compute TFRs.
        crop_val: float
            Value to crop from the start and end of the epoch.
        decim: float
            Decimation factor.

    Returns:
        power: mne.time_frequency.EpochsTFR
    """
    n_cycles = freqs / 4.  # different number of cycle per frequency

    # Compute power for move trials
    power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                       return_itc=False, decim=decim, n_jobs=1, average=False)

    # trim epoch to avoid edge effects
    power.crop(epoch_times[0] + crop_val, epoch_times[1] - crop_val)

    # convert to log scale
    power.data = 10 * np.log10(power.data)

    # set infinite values to 0
    power.data[np.isinf(power.data)] = 0
    return power


def tfr_subtract_baseline(power, chan_ind, baseidx, compute_mean=False):
    """
    From MNE format to time x freq x trials with baseline subtraction
    Copied from: https://github.com/stepeter/naturalistic_arm_movements_ecog/blob/master/compute_power_gen_figs_python/tfr_utils.py
    """
    input_power = np.squeeze(
        power.data[:, chan_ind, :, :])  # trials x freq x time
    input_power = np.moveaxis(input_power, 0, -1)  # freq x time x trials
    if compute_mean:
        baseline = np.mean(input_power[:, baseidx, :], axis=1)
    else:
        baseline = np.median(input_power[:, baseidx, :], axis=1)
    curr_power = input_power - np.tile(np.expand_dims(baseline, 1), (
        1, input_power.shape[1], 1))  # subtract baseline
    return curr_power
