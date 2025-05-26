import pickle

# Load the pkl file properly in binary mode
with open('rfml/data/2018.01/iq_by_mod_snr.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')  # encoding needed if the file is python2 pickle

# Save it again, but with .dat extension
with open('gold.dat', 'wb') as f:
    pickle.dump(data, f)
