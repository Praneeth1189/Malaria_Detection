import tensorflow as tf
import tensorflow_datasets as tfds
from keras import BinaryCrossentropy
from keras import Adam
from Model.CNN_Resnet import model

dataset, dataset_info = tfds.load('malaria', as_supervised = True, with_info = True,
                                   split=["train"], shuffle_files=True)

def split(dataset, train_split, val_split):
  n = len(dataset)
  train_dataset = dataset.take(int(train_split*n))
  X = dataset.skip(int(train_split*n))
  val_dataset = X.take(int(val_split*n))
  test_dataset = X.skip(int(val_split*n))
  return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = split(dataset[0],0.8,0.1)
image_size = 224

def resize_rescale(image, label):
  return tf.image.resize(image,(image_size,image_size))/255. , label

train_dataset = train_dataset.map(resize_rescale)
val_dataset = val_dataset.map(resize_rescale)
test_dataset = test_dataset.map(resize_rescale)

batch_size = 32
train_dataset = train_dataset.shuffle(buffer_size=8,
                                      reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size)

model = model()
model.compile(Adam(learning_rate=0.001),
              loss = BinaryCrossentropy(),
              metrics = 'accuracy')
history = model.fit(train_dataset, validation_data = val_dataset, epochs = 5, verbose = 1)
