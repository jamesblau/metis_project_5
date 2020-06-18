import re
import pickle
from glob import glob
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras import Model
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import np_utils

from google.colab import drive
drive.mount('/content/gdrive')

project_dir = "gdrive/My Drive/metis_project_5"

# Get easy labels, wiz_sfat

with open(f"{project_dir}/labels/classifier/easy_wiz_sfat_larger_cropped_labels.csv") as f:
    raw_ws_labels = f.read().split('\n')[1:-1]

ws_label_regex = re.compile('^([^.]*)\.png,(.*)$')

ws_label_to_int = {
    'g': 0,
    'a': 1,
    's': 0,
    'aub': 1,
    'gdb': 0,
    'adb': 1,
    'l': 1,
}
ws_num_categories = 2

ws_paths, ws_labels, ws_y = [], [], []
for label in raw_ws_labels:
    frame, label = ws_label_regex.match(label).groups()
    if label in ws_label_to_int.keys():
        int_label = ws_label_to_int[label]
        if int_label:
            ws_labels.append('a')
        else:
            ws_labels.append('g')
        path = f"{project_dir}/data/wiz_sfat/larger_cropped/{frame}.png"
        ws_paths.append(path)
        ws_y.append(int_label)

# Get semi-easy labels, wiz_sfat

with open(f"{project_dir}/labels/classifier/easy_wiz_sfat_larger_cropped_labels.csv") as f:
    raw_se_labels = f.read().split('\n')[1:-1]

se_label_regex = re.compile('^([^.]*)\.png,(.*)$')

se_label_to_int = {
    'g': 0,
    'a': 1,
    's': 2,
    'aub': 3,
    'gdb': 4,
    'adb': 4,
    'l': 5,
}
se_num_categories = 6

se_paths, se_labels, se_y = [], [], []
for label in raw_se_labels:
    frame, label = se_label_regex.match(label).groups()
    se_labels.append(label)
    if label in se_label_to_int.keys():
        int_label = se_label_to_int[label]
        path = f"{project_dir}/data/wiz_sfat/larger_cropped/{frame}.png"
        se_paths.append(path)
        se_y.append(int_label)

# Get hard labels, wiz_sfat

with open(f"{project_dir}/labels/classifier/hard_wiz_sfat_labels.csv") as f:
    raw_hws_labels = f.read().split('\n')[1:-1]

hws_label_regex = re.compile('^([^\d]*)([^.]*)\.png,.,(.*)$')

hws_label_to_int = {
    'a': 0,
    'adb': 1,
    'anb': 2,
    'aub': 3,
    'ba': 4,
    'd': 5,
    'da': 6,
    'dt': 7,
    'g': 8,
    'gdb': 9,
    'h': 10,
    'ha': 12,
    'hg': 12,
    'j': 13,
    'l': 14,
    'na': 15,
    'p': 16,
    's': 17,
    'sd': 18,
    't': 19,
    'ua': 20,
    'ut': 21,
}
hws_num_categories = 22

hws_paths, hws_labels, hws_y = [], [], []
for label in raw_hws_labels:
    (category, frame, label) = hws_label_regex.match(label).groups()
    int_label = hws_label_to_int.get(label)
    hws_labels.append(label)
    if int_label and category == 'new_fox_':
        int_label = hws_label_to_int[label]
        path = f"{project_dir}/data/wiz_sfat/larger_cropped/{frame}.png"
        hws_paths.append(path)
        hws_y.append(int_label)

# Get easy labels, tbh7_purp_fox_vgbc

with open(f"{project_dir}/labels/classifier/easy_tbh7_purp_fox_vgbc_labels.csv") as f:
    raw_tbh_labels = f.read().split('\n')[1:-1]

tbh_label_regex = re.compile('^([^.]*)\.png,(.*)$')

tbh_label_to_int = {
    'g': 0,
    'a': 1,
    's': 0,
    'aub': 1,
    'gdb': 0,
    'adb': 1,
    'l': 1,
}
tbh_num_categories = 2

tbh_paths, tbh_labels, tbh_y = [], [], []
for label in raw_tbh_labels:
    frame, label = tbh_label_regex.match(label).groups()
    tbh_labels.append(label)
    if label in tbh_label_to_int.keys():
        int_label = tbh_label_to_int[label]
        path = f"{project_dir}/data/tbh7_purp_fox_vgbc/larger_cropped/{frame}.png"
        tbh_paths.append(path)
        tbh_y.append(int_label)

img_hw = (350, 350)
img_dimensions = (350, 350, 3)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=img_hw)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

# Prepare test/train data, wiz_sfat easy

ws_X = np.array([prepare_image(path)[0] for path in ws_paths])

ws_X_tr, ws_X_te, ws_y_tr, ws_y_te = train_test_split(ws_X, ws_y, test_size=0.2, random_state=42)

# Prepare test/train data, wiz_sfat semi-easy

se_X = np.array([prepare_image(path)[0] for path in se_paths])

se_X_tr, se_X_te, se_y_tr, se_y_te = train_test_split(se_X, se_y, test_size=0.2, random_state=42)

# Prepare test/train data, wiz_sfat hard

hws_X = np.array([prepare_image(path)[0] for path in hws_paths])

hws_X_tr, hws_X_te, hws_y_tr, hws_y_te = train_test_split(hws_X, hws_y, test_size=0.2, random_state=42)

# Prepare test/train data, tbh7

tbh_X = np.array([prepare_image(path)[0] for path in tbh_paths])

tbh_X_tr, tbh_X_te, tbh_y_tr, tbh_y_te = train_test_split(tbh_X, tbh_y, test_size=0.2, random_state=42)

pickles_dir = project_dir + "/pickles"

# Dump easy wiz_sfat data

# with open(f'{pickles_dir}/ws_X_tr.pickle', 'wb') as f:
#     pickle.dump(ws_X_tr, f)

# with open(f'{pickles_dir}/ws_X_te.pickle', 'wb') as f:
#     pickle.dump(ws_X_te, f)

# with open(f'{pickles_dir}/ws_y_tr.pickle', 'wb') as f:
#     pickle.dump(ws_y_tr, f)

# with open(f'{pickles_dir}/ws_y_te.pickle', 'wb') as f:
#     pickle.dump(ws_y_te, f)

# Load easy wiz_sfat data

with open(f'{pickles_dir}/ws_X_tr.pickle', 'rb') as f:
    ws_X_tr = pickle.load(f)

with open(f'{pickles_dir}/ws_X_te.pickle', 'rb') as f:
    ws_X_te = pickle.load(f)

with open(f'{pickles_dir}/ws_y_tr.pickle', 'rb') as f:
    ws_y_tr = pickle.load(f)

with open(f'{pickles_dir}/ws_y_te.pickle', 'rb') as f:
    ws_y_te = pickle.load(f)

# Dump semi-easy wiz_sfat data

# with open(f'{pickles_dir}/se_X_tr.pickle', 'wb') as f:
#     pickle.dump(se_X_tr, f)

# with open(f'{pickles_dir}/se_X_te.pickle', 'wb') as f:
#     pickle.dump(se_X_te, f)

# with open(f'{pickles_dir}/se_y_tr.pickle', 'wb') as f:
#     pickle.dump(se_y_tr, f)

# with open(f'{pickles_dir}/se_y_te.pickle', 'wb') as f:
#     pickle.dump(se_y_te, f)

# Load semi-easy wiz_sfat data

with open(f'{pickles_dir}/se_X_tr.pickle', 'rb') as f:
    se_X_tr = pickle.load(f)

with open(f'{pickles_dir}/se_X_te.pickle', 'rb') as f:
    se_X_te = pickle.load(f)

with open(f'{pickles_dir}/se_y_tr.pickle', 'rb') as f:
    se_y_tr = pickle.load(f)

with open(f'{pickles_dir}/se_y_te.pickle', 'rb') as f:
    se_y_te = pickle.load(f)

# Dump hard wiz_sfat data

# with open(f'{pickles_dir}/hws_X_tr.pickle', 'wb') as f:
#     pickle.dump(hws_X_tr, f)

# with open(f'{pickles_dir}/hws_X_te.pickle', 'wb') as f:
#     pickle.dump(hws_X_te, f)

# with open(f'{pickles_dir}/hws_y_tr.pickle', 'wb') as f:
#     pickle.dump(hws_y_tr, f)

# with open(f'{pickles_dir}/hws_y_te.pickle', 'wb') as f:
#     pickle.dump(hws_y_te, f)

# Load hard wiz_sfat data

with open(f'{pickles_dir}/hws_X_tr.pickle', 'rb') as f:
    hws_X_tr = pickle.load(f)

with open(f'{pickles_dir}/hws_X_te.pickle', 'rb') as f:
    hws_X_te = pickle.load(f)

with open(f'{pickles_dir}/hws_y_tr.pickle', 'rb') as f:
    hws_y_tr = pickle.load(f)

with open(f'{pickles_dir}/hws_y_te.pickle', 'rb') as f:
    hws_y_te = pickle.load(f)

# Dump easy tbh data

# with open(f'{pickles_dir}/tbh_X_tr.pickle', 'wb') as f:
#     pickle.dump(tbh_X_tr, f)

# with open(f'{pickles_dir}/tbh_X_te.pickle', 'wb') as f:
#     pickle.dump(tbh_X_te, f)

# with open(f'{pickles_dir}/tbh_y_tr.pickle', 'wb') as f:
#     pickle.dump(tbh_y_tr, f)

# with open(f'{pickles_dir}/tbh_y_te.pickle', 'wb') as f:
#     pickle.dump(tbh_y_te, f)

# Load easy tbh data

with open(f'{pickles_dir}/tbh_X_tr.pickle', 'rb') as f:
    tbh_X_tr = pickle.load(f)

with open(f'{pickles_dir}/tbh_X_te.pickle', 'rb') as f:
    tbh_X_te = pickle.load(f)

with open(f'{pickles_dir}/tbh_y_tr.pickle', 'rb') as f:
    tbh_y_tr = pickle.load(f)

with open(f'{pickles_dir}/tbh_y_te.pickle', 'rb') as f:
    tbh_y_te = pickle.load(f)

# Choose actual training and test sets

# Train on semi-easy labels with wiz_sfat images
X_tr = se_X_tr
y_tr = se_y_tr
X_te = se_X_te
y_te = se_y_te
num_categories = se_num_categories

# Train on tbh images (larger set) and test on wiz_sfat images
X_tr = np.concatenate([tbh_X_tr, tbh_X_te])
y_tr = tbh_y_tr + tbh_y_te
X_te, y_te = ws_X_te, ws_y_te
num_categories = tbh_num_categories

# Train on both train sets and test on both test sets
X_tr = np.concatenate([tbh_X_tr, ws_X_tr])
y_tr = tbh_y_tr + ws_y_tr
X_te = np.concatenate([tbh_X_te, ws_X_te])
y_te = tbh_y_te + ws_y_te
num_categories = ws_num_categories

# Train on hard labels with wiz_sfat images
X_tr = hws_X_tr
y_tr = hws_y_tr
X_te = hws_X_te
y_te = hws_y_te
num_categories = hws_num_categories

# If multi-class, make categorical targets
if num_categories > 2:
  y_tr_cat_compare = np_utils.to_categorical(y_tr)
  num_tr = len(y_tr)
  y = np.concatenate([y_tr, y_te])
  y_cat = np_utils.to_categorical(y)
  y_tr_cat = y_cat[:num_tr]
  y_te_cat = y_cat[num_tr:]
  assert(len(np.concatenate([y_tr_cat, y_te_cat])) == len(y))
  assert(len(y_tr) == len(y_tr_cat))
  assert(len(y_te) == len(y_te_cat))
  assert((y_tr_cat == y_tr_cat_compare).all())
else:
  y_tr_cat = y_tr
  y_te_cat = y_te

# Fetch pretrained model

pretrained = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=img_dimensions,
)

# Build model

for layer in pretrained.layers[:17]:
    layer.trainable = False

model = Sequential()
model.add(pretrained)
model.add(Flatten())

if num_categories > 2:
    model.add(Dense(num_categories))
    loss='categorical_crossentropy',
else:
    model.add(Dense(1, activation='sigmoid'))
    loss='binary_crossentropy',

model.compile(
    loss=loss,
    optimizer='adam',
    metrics=['accuracy'],
)

# Image augmentation
tr_datagen = image.ImageDataGenerator()
te_datagen = tr_datagen

# Training and test gerators
tr_gen = tr_datagen.flow(X_tr, y_tr_cat)
te_gen = te_datagen.flow(X_te, y_te_cat)
epochs=100

# Callbacks
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1,
                   restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.2,
                   patience=10, cooldown=5, min_lr=0.0002)

# Train model
model.fit_generator(tr_gen, steps_per_epoch=10, epochs=epochs, callbacks=[es, reduce_lr], 
                 verbose=1, validation_data=te_gen, validation_steps=20)

# Save / load model

models_dir = project_dir + "/models"

models = glob(models_dir + '/*')
latest_model_num = int(re.sub(".*model_(\d*)\..*", "\\1", sorted(models)[-1]))
next_model_num = latest_model_num + 1

model.save_weights(f'{models_dir}/model_{next_model_num}.hdf5')

model.load_weights(f'{models_dir}/model_{next_model_num}.hdf5')
# model.load_weights(f'{models_dir}/model_{latest_model_num}.hdf5')
