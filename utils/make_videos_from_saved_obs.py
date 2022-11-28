from pathlib import Path
import os

existing_mp4s = [x for x in Path(".").glob("*.mp4")]
# delete the existing mp4s
_ = [os.remove(x) for x in existing_mp4s]

images = Path("images")
folders = [x for x in images.iterdir() if x.is_dir()]
print(folders)
for f in folders:
    print(f)
    os.system(f"ffmpeg -i {f}/{f.name}_%d.png  {f.name}.mp4")