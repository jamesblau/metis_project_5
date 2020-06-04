import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing import image

with open('labels/classifier/easy_labels.csv') as f:
    raw_labels = f.read().split('\n')[1:-1]

label_regex = re.compile('^[^\d]*([^.]*)\.png,(.*)$')

label_to_int = {'a': 0, 'g': 1, 'db': 2, 'ub': 3, 's': 4}

labels = []
for label in raw_labels:
    (frame, label) = label_regex.match(label).groups()
    int_label = label_to_int.get(label)
    if int_label:
        path = f"data/larger_cropped/{frame}.png"
        labels += [(path, int_label)]

img_dimensions = (350, 350)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=img_dimensions)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = mobilenet_v2.preprocess_input(x)
    return x

x = prepare_image(labels[0][0])

pretrained = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=img_dimensions + (3,))

for layer in pretrained.layers[:10]:
    layer.trainable = False

model = Sequential()
model.add(pretrained)
model.add(Flatten())
model.add(Dense(5))

model.predict(x)
