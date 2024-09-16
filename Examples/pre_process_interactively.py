import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
import matplotlib.pyplot as plt
from pathlib import Path
from imports.message_user import show_message_plot
import numpy as np

# Let's define some basics right here
visualize_difference = True
apply_projection = True

# Define EEG channels
eeg_channels = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
# Frequencies of interest for EEG analysis
freqs = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40)
}

# Create array of frequencies
freqs_array = []
for band in freqs.values():
    freqs_array += list(range(int(band[0]), int(band[1]) + 1))

# Read your EDF file
raw = mne.io.read_raw_edf(r'EDF+\Zacker\Zacker.edf', preload=True, infer_types=True)

# Select only EEG channels
raw.pick(eeg_channels)  # Explicitly pick EEG channels to avoid picking other channels
print(raw.info)

# Set montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, on_missing='ignore')


# Define your message and instructions
message = "Pick bads"
instructions = """
1. Please review the data.
2. Make note of any anomalies.
3. Use the toolbar to zoom and pan.
4. Press A on your keyboard to annotate.
5. Close this window to proceed.
NOTE: The home, end, page up, page down,
and arrow keeys all do special functions too
"""
# Display the message plot
show_message_plot(message, instructions)

# Apply preprocessing steps
raw.set_eeg_reference('average', ch_type = 'eeg', projection=apply_projection)  # Set EEG average reference, NOTE: must be applied later!
if apply_projection:
    raw.apply_proj() # MUST be applied or it doesn't work!
raw.filter(l_freq=1, h_freq=40)  # Apply band-pass filter
raw.plot(picks=eeg_channels,block=True)


# Define your message and instructions
message = "Now Interpolating!"
instructions = """
1. Please wait while I interpolate!
2. You may wish to view a before and after
NOTE: The home, end, page up, page down,
and arrow keeys all do special functions too
"""
# Display the message plot
show_message_plot(message, instructions)

# Interpolate bads
print(f'bads: {raw.info["bads"]}')
# raw_interpolated = raw.copy().interpolate_bads(reset_bads=False)# NOTE: Reset bads = false keeps the bads in the data, just different color!!!
raw_interpolated = raw.copy().interpolate_bads(method='spline')# NOTE: Reset bads = false keeps the bads in the data, just different color!!!

raw_interpolated.plot(picks = eeg_channels ,block=True, title='Interpolated bads')

# Compute and plot PSD
raw_interpolated.compute_psd().plot()
plt.show(block=True)
print('Showed interpolated plot')
# Extract data for plotting
times = raw.times * 1e3  # Convert to milliseconds
data_orig = raw.get_data() * 1e6  # Convert to microvolts
data_interp = raw_interpolated.get_data() * 1e6  # Convert to microvolts


if visualize_difference:
    # Plot original and interpolated data
    fig, ax = plt.subplots(figsize=(15, 10))
    for ch_idx, ch_name in enumerate(eeg_channels):
        ax.plot(times, data_orig[ch_idx] + ch_idx * 100, color='blue', label='Original' if ch_idx == 0 else "")
        ax.plot(times, data_interp[ch_idx] + ch_idx * 100, color='red', linestyle='--', label='Interpolated' if ch_idx == 0 else "")

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend(loc='upper right')
    ax.set_title('Original vs Interpolated EEG Data')
    ax.set_yticks([i * 100 for i in range(len(eeg_channels))])
    ax.set_yticklabels(eeg_channels)
    plt.tight_layout()
    plt.show(block=True)
print('Visualized difference')

# # Create virtual EOG channel
# eog_data = raw.copy().pick_channels(['Fp1', 'Fp2']).get_data()
# virtual_eog = eog_data[0] - eog_data[1]  # Difference between Fp1 and Fp2
# info = mne.create_info(ch_names=['EOG'], sfreq=raw.info['sfreq'], ch_types=['eog'])
# virtual_eog_raw = mne.io.RawArray(virtual_eog[np.newaxis, :], info)

# plt.tight_layout()
# plt.show(block=True)
eog_events = mne.preprocessing.find_eog_events(raw,ch_name=['Fp1','Fp2'])
n_blinks = len(eog_events)
# Center to cover the whole blink with full duration of 0.5s:
onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_blinks)
annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                                  orig_time=raw.info['meas_date'])
raw.set_annotations(annotations)
raw.plot(events=eog_events,block=True)  # To see the annotated segments.
print('Plotted events difference')

#method : ``'morlet'`` | ``'multitaper'`` | None
#     freqs : array-like | None
        # The frequencies at which to compute the power estimates.
        # Must be an array of shape (n_freqs,). ``None`` (the
        # default) only works when using ``__setstate__`` and will raise an error otherwise.
        #tmin, tmax : float | None
        #
# raw_interpolated.compute_tfr(method='multitaper',freqs=) #  
# method: Any,
#     freqs: Any,
#     *,
#     tmin: Any | None = None,
#     tmax: Any | None = None,
#     picks: Any | None = None,
#     proj: bool = False,
#     output: str = "power",
#     reject_by_annotation: bool = True,
#     decim: int = 1,
#     n_jobs: Any | None = None,
#     verbose: Any | None = None,
#     **method_kw: Any

# Compute TFR #NOTE: This takes forever if you do the entire dataset!  It will crash if you don't have enough ram!
# tfr = mne.time_frequency.tfr_multitaper(raw_interpolated, freqs=freqs_array, n_cycles=freqs_array, time_bandwidth=2.0, return_itc=False)
# tfr.plot(picks=eeg_channels, baseline=(-0.5, 0), mode='logratio', title='TFR')
# plt.show(block=True)