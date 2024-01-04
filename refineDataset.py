import csv
import pandas as pd
import numpy as np

# This is from the FMA Github repository, it is used to
# properly remove some features from the dataset
import utils

# Step 1: Read the second file to get valid IDs
valid_ids = set()
with open('fma_data/echonest.csv', 'r',encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if there is one
    for row in reader:
        valid_ids.add(row[0])  # Assuming ID is in the first column

# Step 2: Filter the original CSV file
filtered_rows = []
with open('fma_data/raw_tracks.csv', 'r',encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)  # Store the header
    for row in reader:
        if row[0] in valid_ids:
            filtered_rows.append(row)

# Step 3: Write the filtered data to a new CSV file
with open('refined_tracks.csv', 'w', newline='',encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header
    writer.writerows(filtered_rows)

# Columns to remove
columns_to_remove = [
    "album_url", "artist_name", "artist_url", "artist_website",
    "license_image_file", "license_parent_id", "track_comments", 
    "track_composer", "track_copyright_c", "track_copyright_p", 
    "track_date_created", "track_disc_number", "track_explicit_notes", 
    "track_favorites", "track_file", "track_information", 
    "track_language_code", "track_lyricist", "track_publisher"
]

# Read the CSV
df = pd.read_csv('refined_tracks.csv')

# Drop the specified columns
df.drop(columns=columns_to_remove, inplace=True, errors='ignore')

# Save the modified DataFrame back to CSV
df.to_csv('refined_tracks.csv', index=False)

# Columns to remove
columns_to_remove = [
    "artist_active_year_begin", "artist_active_year_end", "artist_associated_labels",
    "artist_comments", "artist_contact", "artist_date_created", "artist_flattr_name",
    "artist_images", "artist_latitude", "artist_location", "artist_longitude", 
    "artist_website", "artist_wikipedia_page","artist_bio","artist_related_projects"
]

# Read the CSV
df = pd.read_csv('fma_data/raw_artists.csv')

# Drop the specified columns
df.drop(columns=columns_to_remove, inplace=True, errors='ignore')

# Save the modified DataFrame back to CSV
#df.to_csv('refined_artists.csv', index=False)

columns_to_remove = [
    "album_comments", "album_date_created", "album_engineer", 
    "album_images", "album_information", "album_producer", "artist_url"
]

# Read the CSV
df = pd.read_csv('fma_data/raw_albums.csv')

# Drop the specified columns
df.drop(columns=columns_to_remove, inplace=True, errors='ignore')

# Save the modified DataFrame back to CSV
df.to_csv('refined_albums.csv', index=False)


features = utils.load('fma_data/features.csv')
echonest = utils.load('fma_data/echonest.csv')
tracks = utils.load('fma_data/tracks.csv')
np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

echonest['echonest','audio_features'].to_csv('refined_analysis.csv')