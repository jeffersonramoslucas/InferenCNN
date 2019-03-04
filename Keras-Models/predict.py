from keras import backend as K
from keras import optimizers
from keras.models import model_from_json

from vgg16 import VGG16
from inception_v3 import Inception
from AlexNet import AlexNet

from utils import load_mnist
from utils import load_fashion_mnist
from utils import load_cifar10
from utils import load_cifar100


import os
import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")


def load_model(model_name):

	# load json and create model
    json_file = open(f"files/{model_name}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

	# load weights into new model
    model.load_weights(f"files/{model_name}.h5")

    return model

def main():
    '''
    Predict the model defined above.
    '''

    dataset_name = 'cifar_10'
    model_name = 'MobileNet'
    image_size = 128
    channels = 3
    batch_size = 1



    if(dataset_name == 'mnist'):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist(image_size, channels)
    elif(dataset_name == 'cifar_10'):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10(image_size, channels)
    elif(dataset_name == 'cifar_100'):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar100(image_size, channels)
    elif(dataset_name == 'fashion_mnist'):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_fashion_mnist(image_size, channels)


    model = load_model(f'{dataset_name}_{model_name}')

    optimizer = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0001,nesterov=False)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])


    start = time.time()
    predict = model.predict(x_val[:32], verbose=1, batch_size=batch_size)
    end = time.time()


    print(str(end-start))



if __name__ == '__main__':
    main()







