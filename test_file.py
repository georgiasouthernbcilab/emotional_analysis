import mne
import matplotlib.pyplot as plt

# Define the mapping of 10-20 names to EMOTIV names (flipped version)
standard_to_emotiv = {
    "Fp1": [""],
    "Fpz": [""],
    "Fp2": [""],
    "AF7": ["LH"],
    "AF3": [""],
    "AFz": [""],
    "AF4": [""],
    "AF8": ["RH"],
    "F9": ["LE"],
    "F7": ["LG"],
    "F3": ["LF"],
    "Fz": ["RB"],
    "F4": ["RF"],
    "F8": ["RG"],
    "F10": ["RE"],
    "FT9": ["CMS"],
    "FC5": ["LD"],
    "FC1": ["LC"],
    "FC2": ["RC"],
    "FC6": ["RD"],
    "FT10": ["DRL"],
    "T7": ["LB"],
    "C3": ["LA"],
    "Cz": ["LK"],
    "C4": ["RA"],
    "T8": ["RK"],
    "TP9": ["LM"],
    "CP5": ["LQ"],
    "CP1": ["LP"],
    "CP2": ["RP"],
    "CP6": ["RQ"],
    "TP10": ["RM"],
    "P3": ["LN"],
    "Pz": ["LJ"],
    "P4": ["RN"],
    "O1": ["LL"],
    "Oz": ["RJ"],
    "O2": ["RL"],
    "TP8": ["RO"],   # Added RO for TP8
    "TP7": ["LO"]    # Added LO for TP7
}

# Create a standard montage and extract all 10-20 positions
montage = mne.channels.make_standard_montage('standard_1020')

# Create a dictionary for the 10-20 positions but only for electrodes with matching EMOTIV pairs
ch_pos_standard = {
    standard: montage.get_positions()['ch_pos'][standard]
    for standard, emotiv_list in standard_to_emotiv.items()
    if emotiv_list[0]  # Ensure that only electrodes with matching EMOTIV names are kept
}

# Create a dictionary for the EMOTIV channel positions with an offset
offset = 0.01  # Offset in the Y direction for EMOTIV electrodes
ch_pos_emotiv = {
    emotiv: (
        montage.get_positions()['ch_pos'][standard][0],  # X position
        montage.get_positions()['ch_pos'][standard][1] + offset,  # Y position (with offset)
        montage.get_positions()['ch_pos'][standard][2]  # Z position
    )
    for standard, emotiv_list in standard_to_emotiv.items()
    for emotiv in emotiv_list
    if standard in montage.get_positions()['ch_pos'] and emotiv  # Only keep valid pairs
}

# Combine both channel positions into one dictionary for the custom montage
combined_ch_pos = {**ch_pos_standard, **ch_pos_emotiv}

# Create a custom montage with both standard and EMOTIV labels
combined_montage = mne.channels.make_dig_montage(combined_ch_pos, coord_frame='head')

# Create a larger figure
plt.figure(figsize=(12, 10))  # Width, Height in inches

# Plot the combined montage with both EMOTIV and standard labels
combined_montage.plot(show_names=True, sphere=(
    0.0,  # X offset
    0.0,  # Y Move head up (positive) or down (negative)
    0.07, # The larger this is, the larger the radius of the dots
    0.2   # Radius of head
))

# Set the plot title
plt.gca().set_title("Electrodes with Matching EMOTIV Pairs", fontsize=16)

# Show the plot
plt.show()
