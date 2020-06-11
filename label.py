import os
from glob import glob
import re
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output

df = pd.read_csv('labels/regions/fox_frcnn_tags.txt')

# best guess fox labels

for _, (path, _, _, _, _, _) in df.head().iterrows():
    image_path = re.sub("(.*),.*", "\\1", path)
    frame_num = re.sub(".*frame(.*)\..*", "\\1", image_path)
    im = Image.open(image_path)
    plt.imshow(im)
    plt.show()
    label = input()
    with open("labels.csv", 'a') as f:
        f.write(f"new_fox_{frame_num}.png,{label}\n")
    clear_output(wait=True)

for image_path in glob("cropped/old_fox*")[:2]:
    filename = image_path.split('/')[1]
    im = Image.open(image_path)
    plt.imshow(im)
    plt.show()
    label = input()
    with open("labels.csv", 'a') as f:
        f.write(f"{filename},{label}\n")
    clear_output(wait=True)

# easy fox labels

for _, (path, _, _, _, _, _) in df.head().iterrows():
    image_path = re.sub("(.*),.*", "\\1", path)
    frame_num = re.sub(".*frame(.*)\..*", "\\1", image_path)
    im = Image.open(image_path)
    plt.imshow(im)
    plt.show()
    label = input()
    with open("easy_labels.csv", 'a') as f:
        f.write(f"new_fox_{frame_num}.png,{label}\n")
    clear_output(wait=True)

for image_path in glob("cropped/old_fox*")[:2]:
    filename = image_path.split('/')[1]
    im = Image.open(image_path)
    plt.imshow(im)
    plt.show()
    label = input()
    with open("easy_labels.csv", 'a') as f:
        f.write(f"{filename},{label}\n")
    clear_output(wait=True)

# best guess falcon labels

for image_path in glob("cropped/old_falcon*")[:2]:
    filename = image_path.split('/')[1]
    im = Image.open(image_path)
    plt.imshow(im)
    plt.show()
    label = input()
    with open("labels.csv", 'a') as f:
        f.write(f"{filename},{label}\n")
    clear_output(wait=True)

# easy falcon labels: TODO

for image_path in glob("cropped/old_falcon*")[:2]:
    filename = image_path.split('/')[1]
    im = Image.open(image_path)
    plt.imshow(im)
    plt.show()
    label = input()
    with open("easy_labels.csv", 'a') as f:
        f.write(f"{filename},{label}\n")
    clear_output(wait=True)

# redo best guess old fox labels from frames: TODO

for image_path in glob("data/wiz_sfat/larger_cropped/*")[:2]:
    filename = image_path.split('/')[-1]
    im = Image.open(image_path)
    plt.imshow(im)
    plt.show()
    label = input()
    with open("labels/classifier/easy_wiz_sfat_larger_cropped_labels.csv", 'a') as f:
        f.write(f"{filename},{label}\n")
    clear_output(wait=True)

# lost some easy labels; redoing on larger_cropped images
