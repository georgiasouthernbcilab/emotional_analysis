import os # Handy OS functions, explore file directory, etc. 
import glob # Useful to grab the EDF files easily
import mne # The main eeg package / library
import matplotlib.pyplot as plt # Use as backend when needed
from datetime import datetime # To time & date stamp output files as needed
import re # To sanitize filename
from mne.preprocessing import ICA # Import it explicitly to minimize required code and refer to it more easily

description = 'PLOT_PSD' # Put a nice description here as it gets saved in the output directory name and code output file
eeg_channels = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']

def sanitize_filename(filename):
    ### Sanitize the filename to remove or replace invalid characters
    return re.sub(r'[\\/*?:"<>|,]', '_', filename)

def save_script_copy(script_path, output_directory):
    ## Save a copy of the script in the output directory
    sanitized_description = sanitize_filename(description)
    script_name = os.path.basename(script_path).replace('.py', f'_{sanitized_description}.py')
    output_path = os.path.join(output_directory, script_name)
    with open(script_path, 'r') as original_script:
        with open(output_path, 'w') as copy_script:
            copy_script.write(original_script.read())
    print(f"Saved a copy of the script to {output_path}")

def plot_psd(edf_file, output_directory):
    try:
        # Load the EDF/BDF file
        raw = mne.io.read_raw_edf(edf_file, preload=True, infer_types=True, verbose=True)

        # Pick only EEG channels
        raw.pick_types(eeg=True)

        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        
        # Set EEG average reference and apply band-pass filter
        raw.set_eeg_reference('average', projection=True)
        raw.filter(l_freq=1, h_freq=40)

        # Set up ICA
        ica = ICA(n_components=32, random_state=97, max_iter="auto")
        ica.fit(raw)

        # Find EOG and muscle artifacts
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='Fp2')
        muscle_noise_indices, muscle_noise_scores = ica.find_bads_muscle(raw)
        
        # Exclude the identified artifact components
        ica.exclude = list(set(eog_indices + muscle_noise_indices))
        
        # Apply ICA to the raw data
        raw_clean = ica.apply(raw.copy())
        
        # Compute PSD for the original raw data
        psd_orig = raw.compute_psd(picks=eeg_channels, fmin=0.5, fmax=40)
        fig_orig = psd_orig.plot(dB=False, show=False)  # Set dB=False here
        ylim_orig = fig_orig.gca().get_ylim()  # Get y-axis limits of the original plot
        
        # Save PSD plot before ICA cleaning
        output_filename_orig = f"{os.path.basename(edf_file).replace('.edf', '').replace('.bdf', '')}_before_ICA_psd.png"
        output_path_orig = os.path.join(output_directory, output_filename_orig)
        fig_orig.savefig(output_path_orig)
        plt.close(fig_orig)
        print(f"Saved PSD plot before ICA cleaning of {edf_file} to {output_path_orig}")
        
        # Plot PSD after ICA cleaning
        psd_clean = raw_clean.compute_psd(picks=eeg_channels, fmin=0.5, fmax=40)
        fig_clean = psd_clean.plot(dB=False, show=False)  # Set dB=False here
        fig_clean.gca().set_ylim(ylim_orig)  # Apply y-axis limits of original plot to maintain scale
        output_filename_clean = f"{os.path.basename(edf_file).replace('.edf', '').replace('.bdf', '')}_after_ICA_psd.png"
        output_path_clean = os.path.join(output_directory, output_filename_clean)
        fig_clean.savefig(output_path_clean)
        plt.close(fig_clean)
        print(f"Saved PSD plot after ICA cleaning of {edf_file} to {output_path_clean}")

        # Extract events and create epochs
        events, event_dict = mne.events_from_annotations(raw, regexp='^(?=.*videos)(?!.*neutralVideo)')
        
        # Loop through each event and plot PSD
        for i, event in enumerate(events):
            event_id = event[-1]
            event_name = list(event_dict.keys())[list(event_dict.values()).index(event_id)]
            
            # Define the time span for the event
            start, stop = event[0] / raw.info['sfreq'], min((event[0] + raw.n_times) / raw.info['sfreq'], raw.times[-1])
            start = start + 15
            stop = start + 45
            
            # Crop the raw data to the event span
            cropped_raw = raw_clean.copy().crop(tmin=start, tmax=stop)
            
            # Plot PSD for the cropped raw data
            psd_epoch = cropped_raw.compute_psd(picks=eeg_channels, fmin=0.5, fmax=40)
            fig_epoch = psd_epoch.plot(dB=False, show=False)  # Set dB=False here
            fig_epoch.gca().set_ylim(ylim_orig)  # Apply y-axis limits of original plot to maintain scale
            
            # Sanitize event name for the filename
            sanitized_event_name = sanitize_filename(event_name)
            
            # Determine subfolder based on the first 6 characters of the filename
            subfolder_name = os.path.basename(edf_file)[:6]
            subfolder_path = os.path.join(output_directory, subfolder_name)
            os.makedirs(subfolder_path, exist_ok=True)
            
            # Save the plot to a PNG file
            output_filename_epoch = f"{os.path.basename(edf_file).replace('.edf', '').replace('.bdf', '')}_epoch_{i + 1}_{sanitized_event_name}_psd.png"
            output_path_epoch = os.path.join(subfolder_path, output_filename_epoch)
            fig_epoch.savefig(output_path_epoch)
            plt.close(fig_epoch)  # Close the figure to free up memory
            print(f"Saved PSD plot for epoch {i + 1} ({sanitized_event_name}) of {edf_file} to {output_path_epoch}")
        
    except Exception as e:
        print(f"Error processing {edf_file}: {e}")

# def plot_psd(edf_file, output_directory):
#     try:
#         # Load the EDF/BDF file
#         raw = mne.io.read_raw_edf(edf_file, preload=True, infer_types=True, verbose=True)

#         # Pick only EEG channels
#         raw.pick_types(eeg=True)

#         montage = mne.channels.make_standard_montage('standard_1020')
#         raw.set_montage(montage, on_missing='ignore')
        
#         # Set EEG average reference and apply band-pass filter
#         raw.set_eeg_reference('average', projection=True)
#         raw.filter(l_freq=1, h_freq=40)

#         # Set up ICA
#         ica = ICA(n_components=32, random_state=97, max_iter="auto")
#         ica.fit(raw)

#         # Find EOG and muscle artifacts
#         eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='Fp2')
#         muscle_noise_indices, muscle_noise_scores = ica.find_bads_muscle(raw)
        
#         # Exclude the identified artifact components
#         ica.exclude = list(set(eog_indices + muscle_noise_indices))
        
#         # Apply ICA to the raw data
#         raw_clean = ica.apply(raw.copy())
        
#         # Plot the PSD before ICA cleaning
#         psd_before = raw.compute_psd(picks=eeg_channels, fmin=0.5, fmax=40)
#         fig_before = psd_before.plot(dB=False, show=False)  # Set dB=False here
#         ylim_before = fig_before.gca().get_ylim()  # Get y-axis limits of the original plot
#         output_filename_before = f"{os.path.basename(edf_file).replace('.edf', '').replace('.bdf', '')}_before_ICA_psd.png"
#         output_path_before = os.path.join(output_directory, output_filename_before)
#         fig_before.savefig(output_path_before)
#         plt.close(fig_before)
#         print(f"Saved PSD plot before ICA cleaning of {edf_file} to {output_path_before}")
        
#         # Plot the PSD after ICA cleaning
#         psd_after = raw_clean.compute_psd(picks=eeg_channels, fmin=0.5, fmax=40)
#         fig_after = psd_after.plot(dB=False, show=False)  # Set dB=False here
#         fig_after.gca().set_ylim(ylim_before)  # Apply y-axis limits of original plot to maintain scale
#         output_filename_after = f"{os.path.basename(edf_file).replace('.edf', '').replace('.bdf', '')}_after_ICA_psd.png"
#         output_path_after = os.path.join(output_directory, output_filename_after)
#         fig_after.savefig(output_path_after)
#         plt.close(fig_after)
#         print(f"Saved PSD plot after ICA cleaning of {edf_file} to {output_path_after}")

#         # Extract events and create epochs
#         events, event_dict = mne.events_from_annotations(raw, regexp='^(?=.*videos)(?!.*neutralVideo)')
        
#         # Loop through each event and plot PSD
#         for i, event in enumerate(events):
#             event_id = event[-1]
#             event_name = list(event_dict.keys())[list(event_dict.values()).index(event_id)]
            
#             # Define the time span for the event
#             start, stop = event[0] / raw.info['sfreq'], min((event[0] + raw.n_times) / raw.info['sfreq'], raw.times[-1])
#             start = start + 15
#             stop = start + 45
            
#             # Crop the raw data to the event span
#             cropped_raw = raw_clean.copy().crop(tmin=start, tmax=stop)
            
#             # Plot the PSD for the cropped raw data
#             psd_epoch = cropped_raw.compute_psd(picks=eeg_channels, fmin=0.5, fmax=40)
#             fig = psd_epoch.plot(dB=False, show=False)  # Set dB=False here
#             fig.gca().set_ylim(ylim_before)  # Apply y-axis limits of original plot to maintain scale
            
#             # Sanitize event name for the filename
#             sanitized_event_name = sanitize_filename(event_name)
            
#             # Determine subfolder based on the first 6 characters of the filename
#             subfolder_name = os.path.basename(edf_file)[:6]
#             subfolder_path = os.path.join(output_directory, subfolder_name)
#             os.makedirs(subfolder_path, exist_ok=True)
            
#             # Save the plot to a PNG file
#             output_filename = f"{os.path.basename(edf_file).replace('.edf', '').replace('.bdf', '')}_epoch_{i + 1}_{sanitized_event_name}_psd.png"
#             output_path = os.path.join(subfolder_path, output_filename)
#             fig.savefig(output_path)
#             plt.close(fig)  # Close the figure to free up memory
#             print(f"Saved PSD plot for epoch {i + 1} ({sanitized_event_name}) of {edf_file} to {output_path}")
        
#     except Exception as e:
#         print(f"Error processing {edf_file}: {e}")


# def plot_psd(edf_file, output_directory):
    # try:
    #     # Load the EDF/BDF file
    #     raw = mne.io.read_raw_edf(edf_file, preload=True, infer_types=True, verbose=True)

    #     # Pick only EEG channels
    #     raw.pick_types(eeg=True)

    #     montage = mne.channels.make_standard_montage('standard_1020')
    #     raw.set_montage(montage, on_missing='ignore')
        
    #     # Set EEG average reference and apply band-pass filter
    #     raw.set_eeg_reference('average', projection=True)
    #     raw.filter(l_freq=1, h_freq=40)

    #     # Set up ICA
    #     ica = ICA(n_components=32, random_state=97, max_iter="auto")
    #     ica.fit(raw)

    #     # Find EOG and muscle artifacts
    #     eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='Fp2')
    #     muscle_noise_indices, muscle_noise_scores = ica.find_bads_muscle(raw)
        
    #     # Exclude the identified artifact components
    #     ica.exclude = list(set(eog_indices + muscle_noise_indices))
        
    #     # Apply ICA to the raw data
    #     raw_clean = ica.apply(raw.copy())
        
    #     # Plot the PSD before ICA cleaning
    #     fig_before = raw.compute_psd(picks=eeg_channels, fmin=1, fmax=40).plot(dB=False, show=False)  # Set dB=False here
    #     output_filename_before = f"{os.path.basename(edf_file).replace('.edf', '').replace('.bdf', '')}_before_ICA_psd.png"
    #     output_path_before = os.path.join(output_directory, output_filename_before)
    #     fig_before.savefig(output_path_before)
    #     plt.close(fig_before)
    #     print(f"Saved PSD plot before ICA cleaning of {edf_file} to {output_path_before}")
        
    #     # Plot the PSD after ICA cleaning
    #     fig_after = raw_clean.compute_psd(picks=eeg_channels, fmin=1, fmax=40).plot(dB=False, show=False)  # Set dB=False here
    #     output_filename_after = f"{os.path.basename(edf_file).replace('.edf', '').replace('.bdf', '')}_after_ICA_psd.png"
    #     output_path_after = os.path.join(output_directory, output_filename_after)
    #     fig_after.savefig(output_path_after)
    #     plt.close(fig_after)
    #     print(f"Saved PSD plot after ICA cleaning of {edf_file} to {output_path_after}")

    #     # Extract events and create epochs
    #     events, event_dict = mne.events_from_annotations(raw, regexp='^(?=.*videos)(?!.*neutralVideo)')
        
    #     # Loop through each event and plot PSD
    #     for i, event in enumerate(events):
    #         event_id = event[-1]
    #         event_name = list(event_dict.keys())[list(event_dict.values()).index(event_id)]
            
    #         # Define the time span for the event
    #         start, stop = event[0] / raw.info['sfreq'], min((event[0] + raw.n_times) / raw.info['sfreq'], raw.times[-1])
    #         start = start + 15
    #         stop = start + 45
            
    #         # Crop the raw data to the event span
    #         cropped_raw = raw_clean.copy().crop(tmin=start, tmax=stop)
            
    #         # Plot the PSD for the cropped raw data
    #         fig = cropped_raw.compute_psd(picks=eeg_channels, fmin=1, fmax=40).plot(dB=False, show=False)  # Set dB=False here
            
    #         # Sanitize event name for the filename
    #         sanitized_event_name = sanitize_filename(event_name)
            
    #         # Determine subfolder based on the first 6 characters of the filename
    #         subfolder_name = os.path.basename(edf_file)[:6]
    #         subfolder_path = os.path.join(output_directory, subfolder_name)
    #         os.makedirs(subfolder_path, exist_ok=True)
            
    #         # Save the plot to a PNG file
    #         output_filename = f"{os.path.basename(edf_file).replace('.edf', '').replace('.bdf', '')}_epoch_{i + 1}_{sanitized_event_name}_psd.png"
    #         output_path = os.path.join(subfolder_path, output_filename)
    #         fig.savefig(output_path)
    #         plt.close(fig)  # Close the figure to free up memory
    #         print(f"Saved PSD plot for epoch {i + 1} ({sanitized_event_name}) of {edf_file} to {output_path}")
        
    # except Exception as e:
    #     print(f"Error processing {edf_file}: {e}")


def find_edf_files(parent_directory): # Self explanatory, let's grab every EDF file and process it
    extensions = ['*.bdf', '*.edf', '*.edf+']
    edf_files = []
    for ext in extensions:
        edf_files.extend(glob.glob(os.path.join(parent_directory, '**', ext), recursive=True))
    return edf_files

def main(parent_directory, output_directory):
    edf_files = find_edf_files(parent_directory) # Grab EDF files

    for edf_file in edf_files: # Process EDF files one at a time
        print(f"Processing file: {edf_file}")
        plot_psd(edf_file, output_directory)

if __name__ == '__main__':
    parent_directory = 'EDF+'
    output_directory = 'PSD_Plots'
    
    # Create a timestamped subfolder in the output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_output_directory = os.path.join(output_directory, sanitize_filename(timestamp + description))
    os.makedirs(timestamped_output_directory, exist_ok=True)
    
    # Save a copy of the script in the output directory for a verbatim record of code that produced the relevant graphs
    save_script_copy(__file__, timestamped_output_directory)
    
    main(parent_directory, timestamped_output_directory)
