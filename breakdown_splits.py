import pandas as pd

tr = pd.read_csv('data/manifests/train.csv')
va = pd.read_csv('data/manifests/val.csv')

# Count events per video
tr_events_per_video = tr.groupby('video_id').size()
va_events_per_video = va.groupby('video_id').size()

print("TRAIN EVENTS BREAKDOWN")
print(f"  Videos: {len(tr_events_per_video)}")
print(f"  Total events: {len(tr)}")
print(f"  Avg events per video: {len(tr) / len(tr_events_per_video):.1f}")
print(f"  Min/Max events: {tr_events_per_video.min()} / {tr_events_per_video.max()}")

print("\nVAL EVENTS BREAKDOWN")
print(f"  Videos: {len(va_events_per_video)}")
print(f"  Total events: {len(va)}")
print(f"  Avg events per video: {len(va) / len(va_events_per_video):.1f}")
print(f"  Min/Max events: {va_events_per_video.min()} / {va_events_per_video.max()}")

print(f"\nTOTAL: {len(tr) + len(va)} events from {len(tr_events_per_video) + len(va_events_per_video)} videos")

# Show original linked events
linked = pd.read_csv('data/manifests/linked_events.csv')
from pathlib import Path
linked['video_id'] = linked['video_path'].map(lambda x: Path(str(x).lower()).name)
linked_per_video = linked.groupby('video_id').size()
print(f"\nORIGINAL linked_events.csv:")
print(f"  Videos: {len(linked_per_video)}")
print(f"  Events: {len(linked)}")
print(f"  Avg per video: {len(linked) / len(linked_per_video):.1f}")
