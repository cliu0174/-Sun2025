import pickle

with open('data/our_data/1-1.pkl', 'rb') as f:
    data = pickle.load(f)

print(type(data))
print(data)
