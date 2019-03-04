#https://github.com/Curt-Park/handwritten_digit_recognition
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D,
                          GlobalAveragePooling2D, Activation, Dense, DepthwiseConv2D)
from keras.models import Model
from keras import optimizers
from base_model import BaseModel

from keras.models import model_from_json

from utils import load_mnist
from utils import load_fashion_mnist
from utils import load_cifar10
from utils import load_cifar100
from utils import generate_aug
from utils import resize_images
#from train import train

import numpy as np
import os, time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

ALPHA = 0.25
MODEL_NAME = f'MobileNet' # This should be modified when the model name changes.

def print_conv_layer_W(file, layers):
    file.write('%d %d %d %d %s %d %d\n' % (layers.get_weights()[0].shape[0],
                                           layers.get_weights()[0].shape[1],  # kernel_x
                                           layers.get_weights()[0].shape[3],  # kernel_y
                                           layers.get_weights()[0].shape[2],  # kernel_qtd
                                           layers.padding,  # padding
                                           layers.strides[0],  # stride_x
                                           layers.strides[1]))  # stride_y

    arr = layers.get_weights()[0]

    for filter_ in range(arr.shape[3]):
        extracted_filter = arr[:, :, :, filter_]
        for filter_p in range(extracted_filter.shape[2]):
            extracted_filter_p = extracted_filter[:, :, filter_p]
            for k in extracted_filter_p:
                for l in k:
                    file.write('%.15f ' % l)
                file.write("\n")
    return


def print_conv_layer_b(file, layer):
    arr = layer.get_weights()[1]

    for k in arr:
        file.write('%.15f\n' % k)
    return


def print_hidden_soft_layer_W(file, layer, conf):
    conf.write('%s 1 %s\n' % (layer.name, layer.input.name))

    if len(layer.get_weights()[0].shape) == 2:
        file.write('%d %d\n' % (layer.get_weights()[0].shape[1],
                                layer.get_weights()[0].shape[0]))
    else:
        file.write('%d %d\n' % (layer.get_weights()[0].shape[0],
                                layer.get_weights()[0].shape[0]))

    arr = layer.get_weights()[0]

    for filter_ in range(arr.shape[1]):
        extracted_filter = arr[:, filter_]
        for k in extracted_filter:
            file.write('%.15f ' % k)
        file.write("\n")

    return


def print_hidden_soft_layer_b(file, layer):
    arr = layer.get_weights()[1]

    for k in arr:
        file.write('%.15f\n' % k)

    return;


def busca_em_profun(model, layers, len_layers, conf, list_mark, idlay):
    print("Input: " + layers.name)
    if isinstance((layers.input), list):

        for lay in layers.input:
            for i in range(0, len_layers):
                if (lay.name.find(model.layers[i].name) >= 0 and list_mark[idlay] == 0):
                    busca_em_profun(model, model.layers[i], len_layers, conf, list_mark, i)
                    conf.write('flatten 1 %s\n' % model.layers[i].name)

        if list_mark[idlay] == 0:
            conf.write('%s %d' % (layers.name, len(layers.input)))
            for lay in layers.input:
                conf.write(' %s' % lay.name)
            conf.write('\n')
            list_mark[idlay] = 1

    else:
        if (layers.input.name.find(layers.name)):

            for i in range(0, len_layers):
                if (layers.input.name.find(model.layers[i].name) >= 0 and list_mark[idlay] == 0):

                    busca_em_profun(model, model.layers[i], len_layers, conf, list_mark, i)

                    conf.write('%s 1 %s\n' % (layers.name, model.layers[i].name))

                    list_mark[idlay] = 1;
                    print(list_mark)

                    if 'depth' in layers.name:
                        print("ENTROU")
                        with open('files/ConvLayersW.txt', 'a+') as file:
                            print_conv_layer_W(file, layers)
                    elif 'conv' in layers.name:
                        with open('files/ConvLayersW.txt', 'a+') as file:
                            print_conv_layer_W(file, layers)
                        with open('files/ConvLayersB.txt', 'a+') as file:
                            print_conv_layer_b(file, layers)
                    elif 'pool' in layers.name:
                        with open('files/ConvLayersW.txt', 'a+') as file:
                            file.write('%d %d %s %d %d\n' % (
                            layers.pool_size[0], layers.pool_size[1], layers.padding, layers.strides[0],
                            layers.strides[1]))
                    elif 'zero' in layers.name:
                        with open('files/ConvLayersW.txt', 'a+') as file:
                            file.write('%d %d %d %d\n' % (
                            layers.padding[0][0], layers.padding[0][1], layers.padding[1][0], layers.padding[1][1]))
                    elif 'batch_norm' in layers.name:
                        with open('files/ConvLayersW.txt', 'a+') as file:
                            file.write('%s\n' % len(layers.get_weights()))
                            for params in layers.get_weights():
                                for k in params:
                                    file.write('%.15f ' % k)
                                file.write('\n')

    return


class MobileNet(BaseModel):
    '''
    1. ZeroPadding2D (2, 2)
    2. 3X3 Conv2D 32
    3. Densewise separable convolution block X 13
    4. GlobalAveragePooling2D
    5. FC 10 + Softmax
    '_build()' is only modified when the model changes.
    HowToUse:
        model = MobileNet()
        * all funtionalities are written in BaseModel.py
    '''
    def __init__(self, input_size, channels, classes):
        '''
        - Reference for hyperparameters
          => https://github.com/Zehaos/MobileNet/issues/13
        '''
        self.input_size = input_size
        self.channels = channels
        self.classes = classes
        callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                       patience = 30, verbose = 1)]
        #optimizer = optimizers.SGD(lr=0.0001, momentum=0.9)#
        optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #optimizer = optimizers.RMSprop(lr = 0.01)
        BaseModel.__init__(self, model = self._build(), optimizer = optimizer,
                           callbacks = callbacks)

    def _build(self):
        '''
        Builds MobileNet.
        - MobileNets (https://arxiv.org/abs/1704.04861)
          => Depthwise Separable convolution
          => Width multiplier
        - Implementation in Keras
          => https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet.py
        - How Depthwise conv2D works
          => https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d
        Returns:
            MobileNet model
        '''
        alpha = ALPHA # 0 < alpha <= 1
        x = Input(shape = (self.input_size, self.input_size, self.channels))
        y = ZeroPadding2D(padding = (1, 1))(x) # matching the image size of CIFAR-10

        # some layers have different strides from the papers considering the size of mnist
        y = Conv2D(int(32 * alpha), (3, 3),padding='valid',strides = (2, 2))(y) # strides = (2, 2) in the paper
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = self._densewise_sep_conv(y, 64, alpha) # spatial size: 32 x 32
        y = self._densewise_sep_conv(y, 128, alpha, strides = (2, 2)) # spatial size: 32 x 32
        y = self._densewise_sep_conv(y, 128, alpha) # spatial size: 16 x 16
        y = self._densewise_sep_conv(y, 256, alpha, strides = (2, 2)) # spatial size: 8 x 8
        y = self._densewise_sep_conv(y, 256, alpha) # spatial size: 8 x 8
        y = self._densewise_sep_conv(y, 512, alpha, strides = (2, 2)) # spatial size: 4 x 4
        for _ in range(5):
            y = self._densewise_sep_conv(y, 512, alpha) # spatial size: 4 x 4
        y = self._densewise_sep_conv(y, 1024, alpha, strides = (2, 2)) # spatial size: 2 x 2
        y = self._densewise_sep_conv(y, 1024, alpha) # strides = (2, 2) in the paper
        y = GlobalAveragePooling2D()(y)
        y = Dense(1024, activation='relu')(y)
        y = Dense(self.classes, activation='softmax')(y)
        #y = Dense(units = 10)(y)
        #y = Activation('softmax')(y)

        return Model(x, y, name = MODEL_NAME)

    def _densewise_sep_conv(self, x, filters, alpha, strides = (1, 1)):
        '''
        Creates a densewise separable convolution block
        Args:
            x - input
            filters - the number of output filters
            alpha - width multiplier
            strides - the stride length of the convolution
        Returns:
            A densewise separable convolution block
        '''
        y = ZeroPadding2D(padding = (1, 1))(x)
        y = DepthwiseConv2D((3, 3), padding='valid', use_bias=False,strides = strides)(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(int(filters * alpha), (1, 1), padding = 'same', strides=(1, 1))(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        return y

#def main():
    '''
    Train the model defined above.
    '''
dataset_model = 'cifar_10'
image_size = 128
channels = 3
n_classes = 10
epochs =300
batch_size = 32
layers_conv, layers_hidden = 3, 1

if(dataset_model == 'mnist'):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist(image_size, channels)
elif(dataset_model == 'cifar_10'):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10(image_size, channels)
elif(dataset_model == 'cifar_100'):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar100(image_size, channels)
elif(dataset_model == 'fashion_mnist'):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_fashion_mnist(image_size, channels)

#print('x_train shape:', x_train.shape)
#print('x_val shape:', x_val.shape)
#print('x_test shape:', x_test.shape)
#print('y_train shape:', y_train.shape)
#    print('y_val shape:', y_val.shape)
#    print('y_test shape:', y_test.shape)
#    print('Loaded Images ', type(x_train[0][0][0][0]))


    #x_train, y_train = generate_aug(x_train, y_train)
   

    #train_generator = get_train_generator()


json_file = open('files/cifar_10_MobileNet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("files/cifar_10_MobileNet.h5")
print("Loaded model from disk")
loaded_model.summary()

'''then = time.time()
preds = loaded_model.predict(x_test[:10])
now = time.time()
diff = now - then
print(diff % 60)
print(preds[0])'''

optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
loaded_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy',
                         metrics = ['accuracy'])
x = loaded_model.evaluate(x_test[:10], y_test[:10])
print(x)

'''with open('files/ConvLayersW.txt', 'a+') as file:
    file.write('%d\n' % (layerss_conv));
with open('files/HiddenLayersW.txt', 'a+') as file:
    file.write('%d\n' % (layers_hidden));

with open('files/conf.txt', 'a+') as conf:
    id_layer = 12
    list_mark = [0] * len(loaded_model.layers)
    layers = loaded_model.layers[id_layer]
    busca_em_profun(loaded_model, layers, len(loaded_model.layers), conf, list_mark, id_layer)

    conf.write('flatten 1 %s\n' % layers.name)

    layers = loaded_model.layers[14]
    with open('files/HiddenLayersW.txt', 'a+') as f:
        print_hidden_soft_layer_W(f, layers, conf)
        #layers = mobile_model.get_model().layers[35]
        #print_hidden_soft_layer_W(f, layers, conf)

    layers = loaded_model.layers[14]
    with open('files/HiddenLayersB.txt', 'a+') as f:
        print_hidden_soft_layer_b(f, layers)
        #layers = mobile_model.get_model().layers[35]
        #print_hidden_soft_layer_b(f, layers)



    layers = loaded_model.layers[15]
    with open('files/SmrLayerW.txt', 'a+') as f:
        print_hidden_soft_layer_W(f, layers, conf)
    with open('files/SmrLayerB.txt', 'a+') as f:
        print_hidden_soft_layer_b(f, layers)

    conf.write('endconf 1 %s\n' % layers.name)
exit()'''

callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                       patience = 30, verbose = 1)]
#optimizer = optimizers.SGD(lr=0.0001, momentum=0.9)#
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
loaded_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

layer = 69

test = np.array([x_test[0]])
get_1rd_layer_output = K.function([loaded_model.layers[0].input] + [K.learning_phase()],
                                      [loaded_model.layers[layer-1].output])
layer_output = get_1rd_layer_output([test, 1.])[0]

#print(layer_output.shape)
arr = np.array(layer_output)
#print(layer_output)
with open('outputLayers' + str(layer) + '.txt', 'w') as file:
        
    '''for filter_ in range(arr.shape[0]):
        extracted_filter = arr[filter_, :, :, :]
        for filter_p in range(extracted_filter.shape[0]):
            extracted_filter_p = extracted_filter[filter_p, :, :]
            for k in extracted_filter_p:
                for l in k:
                    file.write('%f ' % l)
                file.write("\n")
            file.write("\n")
        '''
    for filter_ in range(arr.shape[1]):
        extracted_filter = arr[:, filter_]
        for k in extracted_filter:
            file.write('%.15f ' % k)
        file.write("\n")
    

#if __name__ == '__main__':
#    main()
