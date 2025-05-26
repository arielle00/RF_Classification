import pickle


# Load the pickle file with latin1 encoding
with open('RML2016.10b_converted.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Print keys and shapes of corresponding values
for key, value in data.items():
    print(f"Key: {key}, Value shape: {value.shape}")
