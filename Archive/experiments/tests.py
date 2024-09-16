import mne
import numpy as np
import matplotlib.pyplot as plt

# Read your EDF file
raw = mne.io.read_raw_edf(r'EDF+\254362\254362.edf', preload=True)
print(f"Raw data loaded. Number of channels: {len(raw.ch_names)}, Sampling frequency: {raw.info['sfreq']} Hz")

# Select EEG channels
eeg_channels = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
raw.pick_channels(eeg_channels)
print(f"Selected EEG channels: {eeg_channels}")

# Set montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, on_missing='ignore')
print("Standard 10-20 montage applied.")

# Apply preprocessing steps
raw.set_eeg_reference('average', projection=True)
raw.filter(l_freq=1, h_freq=40)
print("Applied average reference and band-pass filter (1 - 40 Hz).")

# Perform ICA to identify and remove EOG artifacts
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
print("ICA fitting done.")

# Find EOG artifacts using Fp1 and Fp2 channels
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])
print(f"Identified EOG artifacts: {eog_indices}")

# Create epochs around EOG events
eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=['Fp1', 'Fp2'])

# Annotate entire epochs around EOG events
eog_epochs_annot = []
for idx, event in enumerate(eog_epochs.events):
    onset = event[0]  # onset sample of the event
    duration = 1.0 / raw.info['sfreq']  # duration of each epoch (assuming 1 sample duration)
    description = f'EOG Artifact {idx + 1}'  # unique description for each epoch
    eog_epochs_annot.append([onset, duration, description])

# Convert to numpy array
eog_epochs_annot = np.array(eog_epochs_annot)

# Create Annotations object
annotations = mne.Annotations(onset=eog_epochs_annot[:, 0],
                              duration=eog_epochs_annot[:, 1],
                              description=eog_epochs_annot[:, 2],
                              orig_time=raw.info['meas_date'])

# Set annotations to include entire epochs around EOG events
raw.set_annotations(annotations)

# Plot raw data with annotations
events_from_annot = mne.events_from_annotations(raw, event_id=None, regexp=None, use_rounding=False)[0]
order = np.arange(0, len(raw.ch_names))  # Replace with your valid channel indices
raw.plot(start=5, duration=20, order=order, title='Raw EEG Data with EOG Artifacts', scalings='auto', events=events_from_annot)

plt.show()

print("Plotted raw EEG data with EOG artifacts highlighted.")
