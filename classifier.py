import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing import image
from keras.utils import np_utils

# Get easy labels

with open('labels/classifier/easy_labels.csv') as f:
    raw_easy_labels = f.read().split('\n')[1:-1]

easy_label_regex = re.compile('^([^\d]*)([^.]*)\.png,(.*)$')

easy_label_to_int = {'a': 0, 'g': 1, 'db': 2, 'ub': 3, 's': 4}

easy_paths, easy_y = [], []
for label in raw_easy_labels:
    (category, frame, label) = label_regex.match(label).groups()
    int_label = label_to_int.get(label)
    if int_label and category == 'new_fox_':
        path = f"data/larger_cropped/{frame}.png"
        easy_paths.append(path)
        easy_y.append(int_label)

easy_num_categories = 5

# Get new easy_larger_cropped labels

with open('labels/classifier/easy_larger_cropped_labels.csv') as f:
    raw_elc_labels = f.read().split('\n')[1:-1]

elc_label_regex = re.compile('^([^.]*)\.png,(.*)$')

elc_label_to_int = {
    'g': 0,     # count: 200
    'a': 1,     # count: 128
    's': 0,     # count: 7
    'aub': 1,   # count: 5
    'gdb': 0,   # count: 5
    'adb': 1,   # count: 2
    # 'l': 1,   # count: 1
}

elc_paths, elc_y = [], []
for label in raw_elc_labels:
    frame, label = elc_label_regex.match(label).groups()
    if label in elc_label_to_int.keys():
        int_label = elc_label_to_int[label]
        path = f"data/larger_cropped/{frame}.png"
        elc_paths.append(path)
        elc_y.append(int_label)

elc_num_categories = 2

# Get hard labels

with open('labels/classifier/labels.csv') as f:
    raw_hard_labels = f.read().split('\n')[1:-1]

hard_label_regex = re.compile('^([^\d]*)([^.]*)\.png,.,(.*)$')

hard_label_to_int = {
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
    'ha': 11,
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

hard_paths, hard_y = [], []
for label in raw_hard_labels:
    (category, frame, label) = hard_label_regex.match(label).groups()
    int_label = label_to_int.get(label)
    if int_label and category == 'new_fox_':
        labels.add(label)
        path = f"data/larger_cropped/{frame}.png"
        paths.append(path)
        hard_y.append(label)

hard_num_categories = 22

# Choose labels

paths = elc_paths
y = elc_y
num_categories = elc_num_categories

# Prepare test/train data

img_hw = (350, 350)
img_dimensions = (350, 350, 3)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=img_hw)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = mobilenet_v2.preprocess_input(x)
    return x

X = np.array([prepare_image(path)[0] for path in paths])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

y_tr_cat = np_utils.to_categorical(y_tr)

# Build model

pretrained = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=img_dimensions,
)

for layer in pretrained.layers[:10]:
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

# Train model

if num_categories > 2:
    model.fit(X_tr, y_tr_cat)
else:
    model.fit(X_tr, y_tr)

# with open('pickles/model_1.pickle', 'wb') as f:
    # pickle.dump(model, f)

with open('pickles/model_1.pickle', 'rb') as f:
    model = pickle.load(f)

# Predict and score

y_pr = model.predict_classes(X_te)[:,0]

len(y_pr[y_pr == y_te]) / len(y_pr)

len(y_pr), len(y_te)

y_tr_pr = model.predict_classes(X_tr)

len(y_tr_pr[y_tr_pr == y_tr]) / len(y_tr_pr)

len(y_tr_pr), len(y_tr)
