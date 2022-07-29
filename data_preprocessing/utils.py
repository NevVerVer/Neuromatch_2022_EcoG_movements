from typing import List, Union
import natsort
import glob

import mne
from mne.time_frequency import tfr_morlet
import numpy as np
import pandas as pd


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
    print('Computing power...')
    power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                       return_itc=False, decim=decim, n_jobs=1, average=False)
    print('Done computing power.')

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


def project_spectral_power(tfr_lp, roi_proj_loadpath, good_rois, n_subjs,
                           atlas='aal', rem_bad_chans=False):
    '''
    Loads in electrode-level spectral power and projects it to ROI's.
    Copied from: https://github.com/stepeter/naturalistic_arm_movements_ecog/blob/04c6faf2f219586fa834d10c614ef48a694cec55/compute_power_gen_figs_python/tfr_utils.py#L135
    '''
    first_pass = 0
    metadata_list = []
    metadata_all = []
    for j in range(n_subjs):
        fname_tfr = natsort.natsorted(glob.glob(
            tfr_lp + '/subj_' + str(j + 1).zfill(2) + '*_epo-tfr.h5'))
        for i, fname_curr in enumerate(fname_tfr):
            if i == 0:
                power_load = mne.time_frequency.read_tfrs(fname_curr)[0]
                bad_chans = power_load.info['bads']
                ch_list = np.asarray(power_load.info['ch_names'])
                metadata_all.append(
                    power_load.metadata[power_load.metadata.false_pos == 0])
            else:
                pow_temp = mne.time_frequency.read_tfrs(fname_curr)[0]
                power_load.data = np.concatenate(
                    (power_load.data, pow_temp.data), axis=0)
                metadata_all.append(
                    pow_temp.metadata[pow_temp.metadata.false_pos == 0])
        metadata_list += [str(j)] * int(power_load.data.shape[0])

        # Project to ROI's
        df = pd.read_csv(
            roi_proj_loadpath + '/' + atlas + '_' + str(j + 1).zfill(
                2) + '_elecs2ROI.csv')
        chan_ind_vals = np.nonzero(df.transpose().mean().values != 0)[0][
                        1:]  # + 1

        if rem_bad_chans:
            inds2drop = []
            for i, bad_ch in enumerate(bad_chans):
                inds2drop.append(np.nonzero(ch_list == bad_ch)[0])
            inds2drop = np.asarray(inds2drop)
            if len(inds2drop) == 1:
                inds2drop = inds2drop[0]

            df.iloc[inds2drop] = 0
            sum_vals = df.sum(axis=0).values
            for s in range(len(sum_vals)):
                df.iloc[:, s] = df.iloc[:, s] / sum_vals[s]

        if first_pass == 0:
            power_ROI = power_load.copy()
            # Remove channels so have some number as good ROIs
            n_ch = len(power_load.info['ch_names'])
            chs_rem = power_load.info['ch_names'][len(good_rois):]
            power_ROI.drop_channels(chs_rem)
            first_pass = 1

            for s, roi_ind in enumerate(good_rois):
                power_tmp = power_load.copy()
                normalized_weights = np.asarray(df.iloc[chan_ind_vals, roi_ind])
                pow_dat_tmp = np.moveaxis(power_tmp.data, 0,
                                          -1)  # move epochs to last dimension
                orig_pow_shape = pow_dat_tmp.shape
                reshaped_pow_dat = np.reshape(pow_dat_tmp, (
                    orig_pow_shape[0], np.prod(orig_pow_shape[1:])))
                del pow_dat_tmp
                power_norm = np.dot(normalized_weights, reshaped_pow_dat)
                power_ROI.data[:, s, :, :] = np.moveaxis(
                    np.reshape(power_norm, orig_pow_shape[1:]), -1, 0)
        else:
            pow_dat_all_roi_tmp = np.zeros(
                [power_load.data.shape[0], len(good_rois),
                 power_load.data.shape[2], power_load.data.shape[3]])
            for s, roi_ind in enumerate(good_rois):
                power_tmp = power_load.copy()
                normalized_weights = np.asarray(df.iloc[chan_ind_vals, roi_ind])
                pow_dat_tmp = np.moveaxis(power_tmp.data, 0,
                                          -1)  # move epochs to last dimension
                orig_pow_shape = pow_dat_tmp.shape
                reshaped_pow_dat = np.reshape(pow_dat_tmp, (
                    orig_pow_shape[0], np.prod(orig_pow_shape[1:])))
                del pow_dat_tmp
                power_norm = np.dot(normalized_weights, reshaped_pow_dat)
                pow_dat_all_roi_tmp[:, s, :, :] = np.moveaxis(
                    np.reshape(power_norm, orig_pow_shape[1:]), -1, 0)

            # Concatenate along epoch dimension
            power_ROI.data = np.concatenate(
                (power_ROI.data, pow_dat_all_roi_tmp), axis=0)

    power_ROI._metadata = pd.DataFrame(metadata_list, columns=['patient_id'])
    power_ROI.metadata_all = pd.concat(metadata_all)
    # add patient_id to metadata
    power_ROI.metadata_all[
        'patient_index'] = power_ROI._metadata.patient_id.to_numpy(dtype=int)
    power_ROI.metadata_all.reset_index(drop=True, inplace=True)
    return power_ROI
