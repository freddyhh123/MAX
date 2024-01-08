import os
import torchaudio
import csv

directory = 'data'
bad_files = []
fields = ['folder', 'file', 'error'] 

for folder in os.listdir(directory):
    if folder == "checksums":
        continue
    for filename in os.listdir(os.path.join(directory, folder)):
        if filename.endswith('.mp3'):
            full_path = os.path.join(directory, folder, filename)
            try:
                torchaudio.load(full_path, normalize=True)
            except Exception as e:  # Catch the exception to understand the issue
                bad_files.append((folder, filename, str(e)))

with open('bad_files.csv', 'w',newline='') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(bad_files)