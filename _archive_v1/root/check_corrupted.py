import pandas as pd
import cv2
from pathlib import Path

# Load linked events
df = pd.read_csv('data/manifests/linked_events.csv')
df = df[df['linked'] == True]

print(f'Total linked events: {len(df)}')
print(f'Unique videos: {df["video_path"].nunique()}')

# Check each unique video
errors = []
for vpath in df['video_path'].unique():
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        errors.append(vpath)
    else:
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            errors.append(vpath)
    cap.release()

print(f'\nCorrupted videos in linked events: {len(errors)}')
if errors:
    print('\nAffected videos:')
    for e in errors[:10]:
        print(f'  {e}')
    
    # Count affected events
    affected = df[df['video_path'].isin(errors)]
    print(f'\nAffected events: {len(affected)}')
    print(f'By group: {affected["group"].value_counts().to_dict()}')
