import pickle

# Path to your .pkl file
# file_path = './data/2018.01/iq_by_mod_snr.pkl'
file_path = './data/RML2016.10a_dict.pkl'

# Load the file
with open(file_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')  # use 'latin1' if original was Python 2

# Check what type of object it is
print("Type:", type(data))

# If it's a dict (common for modulation datasets)
if isinstance(data, dict):
    print("Top-level keys (usually tuples like (modulation, SNR)):")
    mods = set()
    for i, (k, v) in enumerate(data.items()):
        print(f"{k}: {type(v)}")
        if i == 9:  # stop after 10
            break


    
    # Example: inspect one sample
    first_key = list(data.keys())[0]
    print("\nShape of sample under first key:", data[first_key].shape)
    print("First sample:\n", data[first_key][0])
