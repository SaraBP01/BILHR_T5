import pickle
import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
#pickle_path = os.path.join(script_dir, 'blob_shoulder_data_T4.pkl')
#minmax_path = os.path.join(script_dir, 'blob_shoulder_minmax_T4.pkl')
pickle_file = os.path.join(script_dir, 'blob_shoulder_data_T4.pkl')
txt_file = os.path.join(script_dir, 'blob_shoulder_data_T4.txt')
minmax_file = os.path.join(script_dir, 'blob_shoulder_minmax_T4.pkl')

import pickle

# Input and output paths
#pickle_file = 'blob_shoulder_data_T4.pkl'
#txt_file = 'blob_shoulder_data_T4.txt'
#minmax_file = 'blob_shoulder_minmax_T4.pkl'

# Load the pickle data
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

# Find minimum and maximum per axis/joint
mins = [1000] * 4
maxs = [-1000] * 4
for vec in data:
    for k in range(4):
        if vec[k] > maxs[k]:
            maxs[k] = vec[k]
        if vec[k] < mins[k]:
            mins[k] = vec[k]

# Normalize data
for l in range(len(data)):
    vec = data[l]
    data[l] = [(vec[k] - mins[k]) / (maxs[k] - mins[k]) for k in range(4)]

# Write the data to a text file
with open(txt_file, 'w') as f:
    for item in data:
        line = ', '.join(str(x) for x in item)
        f.write(line + '\n')

minmax = {'mins': mins, 'maxs': maxs}
with open('blob_shoulder_minmax_T4.pkl', 'wb') as g:
    pickle.dump(minmax, g)

print(f'Data saved to {txt_file}')
print(mins)
print(maxs)