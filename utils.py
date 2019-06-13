import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import shuffle


def process_dataset(train_dir, val_dir, size):
  
  # training set
  
  directory = os.listdir(train_dir + '/dogs')
  train_dogs = []
  train_dogs_labels = []
  for img in directory:
    train_dogs.append(cv2.resize(cv2.imread(train_dir + '/dogs/' + img), (size, size)).reshape(-1))
    train_dogs_labels.append(1)
    
  train_dogs = np.array(train_dogs)
  train_dogs_labels = np.array(train_dogs_labels)
  directory = os.listdir(train_dir + '/cats')
  train_cats = []
  train_cats_labels = []
  for img in directory:
    train_cats.append(cv2.resize(cv2.imread(train_dir + '/cats/' + img), (size, size)).reshape(-1))
    train_cats_labels.append(0)

  train_cats = np.array(train_cats)
  train_cats_labels = np.array(train_cats_labels)
  
  # validation set
  
  directory = os.listdir(val_dir + '/dogs')
  val_dogs = []
  val_dogs_labels = []
  for img in directory:
    val_dogs.append(cv2.resize(cv2.imread(val_dir + '/dogs/' + img), (size, size)).reshape(-1))
    val_dogs_labels.append(1)
    
  val_dogs = np.array(val_dogs)
  val_dogs_labels = np.array(val_dogs_labels)
  
  directory = os.listdir(val_dir + '/cats')
  val_cats = []
  val_cats_labels = []
  for img in directory:
    val_cats.append(cv2.resize(cv2.imread(val_dir + '/cats/' + img), (size, size)).reshape(-1))
    val_cats_labels.append(0)

  val_cats = np.array(val_cats)
  val_cats_labels = np.array(val_cats_labels)
  
  
  x_train = np.concatenate((train_dogs, train_cats))
  y_train = np.concatenate((train_dogs_labels, train_cats_labels))
  
  x_test = np.concatenate((val_dogs, val_cats))
  y_test = np.concatenate((val_dogs_labels, val_cats_labels))
  
  x_train, y_train = shuffle(x_train, y_train, random_state=123)
  x_test, y_test = shuffle(x_test, y_test, random_state=123)
  
  x_train = x_train.astype('float32') / 255.
  x_test = x_test.astype('float32') / 255.
  
  pca = PCA(n_components=200)
  pca.fit(x_train)

  x_train = pca.transform(x_train)
  x_test = pca.transform(x_test)
  
  return x_train, y_train, x_test, y_test