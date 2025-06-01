import os
import shutil

flat_dir = "data/faces"                   # your current flat directory
grouped_dir = "data/faces_grouped"        # new grouped directory

os.makedirs(grouped_dir, exist_ok=True)

# Replace this with logic based on your source if you know which are real/fake
# For demo, we'll assume all are fake (label = 0)
LABEL_PREFIX = "fake_"  # change to "real_" if needed

grouped = {}

# Organize into prefix → list of frames
for fname in sorted(os.listdir(flat_dir)):
    if not fname.endswith('.jpg'):
        continue
    parts = fname.split("_")
    if len(parts) != 2:
        continue
    prefix, frame = parts[0], parts[1]
    grouped.setdefault(prefix, []).append(fname)

for prefix, frames in grouped.items():
    if len(frames) < 10:
        print(f"Skipping {prefix}: only {len(frames)} frames")
        continue
    # Only take the first 10 frames
    frames = sorted(frames)[:10]

    subfolder = os.path.join(grouped_dir, f"{LABEL_PREFIX}{prefix}")
    os.makedirs(subfolder, exist_ok=True)

    for i, fname in enumerate(frames):
        src = os.path.join(flat_dir, fname)
        dst = os.path.join(subfolder, f"{i}.jpg")
        shutil.copy(src, dst)

print("✅ Grouping complete. Saved to:", grouped_dir)
