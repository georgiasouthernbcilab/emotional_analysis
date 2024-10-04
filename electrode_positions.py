import mne
import matplotlib.pyplot as plt

# Define the mapping of EMOTIV names to 10-20 names
emotiv_to_1020 = {
    "CMS": "TP9",
    "DRL": "TP10",
    "LA": "C3",
    "LB": "T7",
    "LC": "FC1",
    "LD": "FC5",
    "LE": "FT9",
    "LF": "F3",
    "LG": "F7",
    "LH": "Fp1",
    "LJ": "Pz",
    "LK": "Cz",
    "LL": "O1",
    "LM": "PO9",
    "LN": "P3",
    "LO": "P7",
    "LP": "CP1",
    "LQ": "CP5",
    "RA": "C4",
    "RB": "Fz",
    "RC": "FC2",
    "RD": "FC6",
    "RE": "FT10",
    "RF": "F4",
    "RG": "F8",
    "RH": "Fp2",
    "RJ": "Oz",
    "RK": "T8",
    "RL": "O2",
    "RM": "PO10",
    "RN": "P4",
    "RO": "P8",
    "RP": "CP2",
    "RQ": "CP6"
}

# Create a montage
montage = mne.channels.make_standard_montage('standard_1020')

# Create a list of the standard 10-20 names and corresponding EMOTIV names
standard_names = []
emotiv_names = []
for emotiv, standard in emotiv_to_1020.items():
    standard_names.append(standard)
    emotiv_names.append(emotiv)

# Plot the montage
fig = montage.plot(show_names=True)

# Add EMOTIV names to the plot
for i, name in enumerate(standard_names):
    pos = montage.get_positions()['ch_pos'][name]
    plt.text(pos, pos, emotiv_names[i], color='red', fontsize=12)

plt.show()
import mne
import matplotlib.pyplot as plt

# Define the mapping of EMOTIV names to 10-20 names
emotiv_to_1020 = {
    "CMS": "TP9",
    "DRL": "TP10",
    "LA": "C3",
    "LB": "T7",
    "LC": "FC1",
    "LD": "FC5",
    "LE": "FT9",
    "LF": "F3",
    "LG": "F7",
    "LH": "Fp1",
    "LJ": "Pz",
    "LK": "Cz",
    "LL": "O1",
    "LM": "PO9",
    "LN": "P3",
    "LO": "P7",
    "LP": "CP1",
    "LQ": "CP5",
    "RA": "C4",
    "RB": "Fz",
    "RC": "FC2",
    "RD": "FC6",
    "RE": "FT10",
    "RF": "F4",
    "RG": "F8",
    "RH": "Fp2",
    "RJ": "Oz",
    "RK": "T8",
    "RL": "O2",
    "RM": "PO10",
    "RN": "P4",
    "RO": "P8",
    "RP": "CP2",
    "RQ": "CP6"
}

# Create a standard montage
montage = mne.channels.make_standard_montage('standard_1020')

# Create a dictionary with the positions of only the relevant channels
ch_pos = {emotiv: montage.get_positions()['ch_pos'][standard] 
          for emotiv, standard in emotiv_to_1020.items() if standard in montage.get_positions()['ch_pos']}

# Create a custom montage with the EMOTIV labels
custom_montage = mne.channels.make_dig_montage(ch_pos, coord_frame='head')

# Create a larger figure
plt.figure(figsize=(12, 10))  # Width, Height in inches

# Plot the custom montage with EMOTIV labels
custom_montage.plot(
    show_names=True, 
    sphere=(
        0,  # X offset
        0,  # Y offset
        0,  # Z offset
        0.12  # Radius of head
    )
)

# # Add a title
# plt.title('EMOTIV Channel Layout', fontsize=16)
# plt.show()
