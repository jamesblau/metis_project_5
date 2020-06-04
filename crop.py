import os, re
import pandas as pd
from PIL import Image

df = pd.read_csv('labels/regions/fox_frcnn_tags.txt')

df.head()

crop_width = 350
crop_height = 350

for _, (path, x1, y1, x2, y2, _) in df.iterrows():
    width, height = x2 - x1, y2 - y1
    w_diff, h_diff = crop_width - width, crop_height - height
    left_pad, top_pad = w_diff / 2, h_diff / 2
    right_pad, bottom_pad = w_diff - left_pad, h_diff - top_pad
    x1 = x1 - left_pad
    x2 = x2 + right_pad
    y1 = y1 - top_pad
    y2 = y2 + bottom_pad
    frame_num = re.sub(".*frame(.*)\..*", "\\1", path)
    out_file_path = f"data/larger_cropped/{frame_num}.png"
    im = Image.open('data/' + path)
    cropped = im.crop((x1, y1, x2, y2)).convert('RGB')
    cropped.save(out_file_path)
