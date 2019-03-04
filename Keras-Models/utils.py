import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

from collections import defaultdict

from document_imgaug import document_imgaug

from keras.utils import np_utils
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator

def normalize_images(images):
    '''
    Channel-wise normalization of the input images: subtracted by mean and divided by std
    Args:
        images: 3-D array
    Returns:
        normalized images: 2-D array
    '''
    H, W = 28, 28
    images = np.reshape(images, (-1, H * W))
    numerator = images - np.expand_dims(np.mean(images, 1), 1)
    denominator = np.expand_dims(np.std(images, 1), 1)
    return np.reshape(numerator / (denominator + 1e-7), (-1, H, W))

def shuffle_dataset(x,y, shuffle=True):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42,
                                                        shuffle=shuffle)
    x_train = np.concatenate((x_train, x_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)
    return x_train, y_train



def load_mnist(image_size, channels):
    '''
    Load mnist data sets for training, validation, and test.
    Args:
        None
    Returns:
        (x_train, y_train): (4-D array, 2-D array)
        (x_val, y_val): (4-D array, 2-D array)
        (x_test, y_test): (4-D array, 2-D array)
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # if K.image_data_format() == 'channels_first':
    #     x_train = x_train.reshape(x_train.shape[0], channels, x_train.shape[1], x_train.shape[2])
    #     x_test = x_test.reshape(x_test.shape[0], channels, x_test.shape[1], x_test.shape[2])
    # else:
    #     x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], channels)
    #     x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], channels)


    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    nb_train_samples = 20000  # 3000 training samples
    nb_teste_samples = 10000

    # Resize images
    if K.image_dim_ordering() == 'th':
        x_train = np.array([cv2.resize(img, (image_size,image_size)) for img in x_train[:nb_train_samples,:,:]])
        x_test = np.array([cv2.resize(img, (image_size,image_size)) for img in x_test[:nb_teste_samples,:,:]])
        x_train = x_train.reshape(-1, channels, image_size, image_size)
        x_test = x_test.reshape(-1, channels, image_size, image_size)
    else:
        x_train = np.array([cv2.resize(img, (image_size,image_size)) for img in x_train[:nb_train_samples,:,:]])
        x_test = np.array([cv2.resize(img, (image_size,image_size)) for img in x_test[:nb_teste_samples,:,:]])
        x_train = x_train.reshape(-1, image_size, image_size, channels)
        x_test = x_test.reshape(-1, image_size, image_size, channels)



    y_train = np_utils.to_categorical(y_train[:nb_train_samples])  # encode one-hot vector
    y_test = np_utils.to_categorical(y_test[:nb_teste_samples])

    nb_valid_samples = 1000
    x_val = x_train[:nb_valid_samples]
    y_val = y_train[:nb_valid_samples]
    x_train = x_train[nb_valid_samples:]
    y_train = y_train[nb_valid_samples:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def load_cifar10(image_size, channels):
    '''
    Load mnist data sets for training, validation, and test.
    Args:
        None
    Returns:
        (x_train, y_train): (4-D array, 2-D array)
        (x_val, y_val): (4-D array, 2-D array)
        (x_test, y_test): (4-D array, 2-D array)
    '''
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    nb_train_samples = 2000 # 3000 training samples
    nb_teste_samples = 1000

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    print(x_train.shape)
    print(x_test.shape)
     
    if K.image_data_format() == 'channels_first':
    #    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    #    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    #    input_shape = (3, img_rows, img_cols)
        print("Entrou no 1");
    else:
    #    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    #    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    #    input_shape = (img_rows, img_cols, 3)
        print("Entrou no 2");

    # Resize images
    if K.image_dim_ordering() == 'th':
        x_train = np.array([cv2.resize(img.transpose(1,2,0), (image_size,image_size)).transpose(2,0,1) for img in x_train[:nb_train_samples,:,:,:]])
        x_test = np.array([cv2.resize(img.transpose(1,2,0), (image_size,image_size)).transpose(2,0,1) for img in x_test[:nb_teste_samples,:,:,:]])
        x_train = x_train.reshape(-1, channels, image_size, image_size)
        x_test = x_test.reshape(-1, channels, image_size, image_size)
      
    else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        x_train = np.array([cv2.resize(img, (image_size,image_size)) for img in x_train[:nb_train_samples,:,:,:]])
        x_test = np.array([cv2.resize(img, (image_size,image_size)) for img in x_test[:nb_teste_samples,:,:,:]])
        x_train = x_train.reshape(-1, image_size, image_size, channels)
        x_test = x_test.reshape(-1, image_size, image_size,channels)

    y_train = np_utils.to_categorical(y_train[:nb_train_samples]) # encode one-hot vector
    y_test = np_utils.to_categorical(y_test[:nb_teste_samples])

    num_of_test_data = 1000
    x_val = x_train[:num_of_test_data]
    y_val = y_train[:num_of_test_data]
    x_train = x_train[num_of_test_data:]
    y_train = y_train[num_of_test_data:]


    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def load_cifar100(image_size, channels):
    '''
    Load mnist data sets for training, validation, and test.
    Args:
        None
    Returns:
        (x_train, y_train): (4-D array, 2-D array)
        (x_val, y_val): (4-D array, 2-D array)
        (x_test, y_test): (4-D array, 2-D array)
    '''
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    #nb_train_samples = 42000 # 3000 training samples
    #nb_teste_samples = 1000

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    
    # Resize images
    if K.image_dim_ordering() == 'th':
        x_train = np.array([cv2.resize(img.transpose(1,2,0), (image_size,image_size)).transpose(2,0,1) for img in x_train[:,:,:,:]])
        x_test = np.array([cv2.resize(img.transpose(1,2,0), (image_size,image_size)).transpose(2,0,1) for img in x_test[:,:,:,:]])
    else:
        x_train = np.array([cv2.resize(img, (image_size,image_size)) for img in x_train[:,:,:,:]])
        x_test = np.array([cv2.resize(img, (image_size,image_size)) for img in x_test[:,:,:,:]])

    x_train = x_train.reshape(-1, image_size, image_size, channels)
    x_test = x_test.reshape(-1, image_size, image_size,channels)
    y_train = np_utils.to_categorical(y_train) # encode one-hot vector
    y_test = np_utils.to_categorical(y_test)

    num_of_test_data = 5000
    x_val = x_train[:num_of_test_data]
    y_val = y_train[:num_of_test_data]
    x_train = x_train[num_of_test_data:]
    y_train = y_train[num_of_test_data:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def load_fashion_mnist(image_size, channels):
    '''
    Load mnist data sets for training, validation, and test.
    Args:
        None
    Returns:
        (x_train, y_train): (4-D array, 2-D array)
        (x_val, y_val): (4-D array, 2-D array)
        (x_test, y_test): (4-D array, 2-D array)
    '''
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    nb_train_samples = 20000
    nb_teste_samples = 10000

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    
    # Resize images
    if K.image_dim_ordering() == 'th':
        x_train = np.array([cv2.resize(img, (image_size,image_size)) for img in x_train[:nb_train_samples,:,:]])
        x_test = np.array([cv2.resize(img, (image_size,image_size)) for img in x_test[:nb_teste_samples,:,:]])
        x_train = x_train.reshape(-1, channels, image_size, image_size)
        x_test = x_test.reshape(-1, channels, image_size, image_size)
    else:
        x_train = np.array([cv2.resize(img, (image_size,image_size)) for img in x_train[:nb_train_samples,:,:]])
        x_test = np.array([cv2.resize(img, (image_size,image_size)) for img in x_test[:nb_teste_samples,:,:]])
        x_train = x_train.reshape(-1, image_size, image_size, channels)
        x_test = x_test.reshape(-1, image_size, image_size, channels)


    y_train = np_utils.to_categorical(y_train[:nb_train_samples]) # encode one-hot vector
    y_test = np_utils.to_categorical(y_test[:nb_teste_samples])

    nb_valid_samples = 1000
    x_val = x_train[:nb_valid_samples]
    y_val = y_train[:nb_valid_samples]
    x_train = x_train[nb_valid_samples:]
    y_train = y_train[nb_valid_samples:]


    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def resize_images(x_train, y_train, x_val, y_val, x_test, y_test, image_size):

    # Resize images
    if K.image_dim_ordering() == 'th':
        x_train = np.array([cv2.resize(img.transpose(1,2,0), (image_size,image_size)).transpose(2,0,1) for img in x_train[:,:,:,:]])
        x_test = np.array([cv2.resize(img.transpose(1,2,0), (image_size,image_size)).transpose(2,0,1) for img in x_test[:,:,:,:]])
    else:
        x_train = np.array([cv2.resize(img, (image_size,image_size)) for img in x_train[:,:,:,:]])
        x_test = np.array([cv2.resize(img, (image_size,image_size)) for img in x_test[:,:,:,:]])

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def generate_aug(x,y):
    x_aug, y_aug = document_imgaug(x,y)
    return x_aug, y_aug

def get_train_generator(x_train, y_train, batch_size = 32, aug_size = 1):
    '''
    Return augmented training data.
    Args:
        x_train: 4-D array
        y_train: 2-D array
        batch_size: integer
    Returns:
        Instance of ImageDataGenerator
        (See: https://keras.io/preprocessing/image/ )
    '''
    train_datagen = ImageDataGenerator(rotation_range = 3,
                                       width_shift_range = 0.1,
                                       height_shift_range = 0.1,
                                       shear_range = 0.2,
                                       zoom_range = 0.1)
    train_datagen.fit(x = x_train,
                augment=True,
                rounds=aug_size)

    return train_datagen.flow(x_train, y_train, batch_size = batch_size)

def get_val_generator(x_val, y_val, batch_size = 32):
    '''
    Return augmented validation data.
    Args:
        x_train: 4-D array
        y_train: 2-D array
        batch_size: integer
    Returns:
        Instance of ImageDataGenerator
        (See: https://keras.io/preprocessing/image/ )
    '''
    val_datagen = ImageDataGenerator()

    return val_datagen.flow(x_val, y_val, batch_size = batch_size, shuffle = False)

def get_test_generator(x_test, y_test, **kwars):
    '''
    Same function as get_val_generator()
    '''
    return get_val_generator(x_test, y_test, **kwars)

def plot(history, path, title = None):
    '''
    Plot the trends of loss and metrics during training
    Args:
        history: History.history attribute. It is a return value of fit method.
        title: string
    Returns:
        None
    '''
    dhist = defaultdict(lambda: None) # just in case history doesn't have validation info
    dhist.update(history.history)

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(dhist['loss'], 'y', label='training loss')
    if dhist['val_loss']:
        loss_ax.plot(dhist['val_loss'], 'r', label='validation loss')

    acc_ax.plot(dhist['acc'], 'b', label='training acc')
    if dhist['val_acc']:
        acc_ax.plot(dhist['val_acc'], 'g', label='validation acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    if title:
        plt.title(title)
    plt.savefig(path, dpi = fig.dpi)