#https://github.com/Curt-Park/handwritten_digit_recognition
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D,
                          GlobalAveragePooling2D, Activation, Dense, Flatten, DepthwiseConv2D)
from keras.models import Model
from keras import optimizers
from base_model import BaseModel
from vgg16 import VGG16
from inception_v3 import Inception
from AlexNet import AlexNet

from utils import load_mnist
from utils import load_fashion_mnist
from utils import load_cifar10
from utils import load_cifar100
from utils import generate_aug
from utils import resize_images

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


'''import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.visible_device_list = '2'
set_session(tf.Session(config=config))'''

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


if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

ALPHA = 1

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
    def __init__(self, MODEL_NAME, input_size, channels, classes):
        '''
        - Reference for hyperparameters
          => https://github.com/Zehaos/MobileNet/issues/13
        '''
        self.MODEL_NAME = MODEL_NAME
        self.input_size = input_size
        self.channels = channels
        self.classes = classes
        #callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
        #                               patience = 30, verbose = 1)]
        optimizer = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0001,nesterov=False)#
        #optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #optimizer = optimizers.RMSprop(lr = 0.01)
        BaseModel.__init__(self, model = self._build(), optimizer = optimizer, callbacks = self.__callbacks())
    
    def __callbacks(self):

        tensorboard = TensorBoard(log_dir=f'files/logs/')

        checkpoint = ModelCheckpoint(f'files/{self.MODEL_NAME}.h5', 
                                    monitor='val_loss', verbose=2, 
                                    save_best_only=True, mode='min')

        early_stoping = EarlyStopping(monitor='val_loss', patience=10, 
                                    verbose=2, restore_best_weights=True)

        return [tensorboard, checkpoint, early_stoping]

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
        x = Input(shape = ( self.channels, self.input_size, self.input_size))
        y = ZeroPadding2D(padding = (1, 1), name='zero_cnv_block')(x) # matching the image size of CIFAR-10

        # some layers have different strides from the papers considering the size of mnist
        y = Conv2D(32, (3, 3),padding='valid', strides=(2, 2), activation='relu', name='conv1_cnv_block')(y) # strides = (2, 2) in the paper
        '''*32 * alpha'''

        y = BatchNormalization(center=0, name='batch_norm_cnv_block')(y)
        y = Activation('relu', name='relu_cnv_block')(y)
        ##y = Conv2D(10, (11, 11), padding='same', activation='relu', name='conv2')(y)  # strides = (2, 2) in the paper
        ##y = Conv2D(10, (11, 11), padding='valid', activation='relu', name='conv3')(y)  # strides = (2, 2) in the paper
        ##y = Conv2D(10, (11, 11), padding='valid', activation='relu', name='conv4')(y)  # strides = (2, 2) in the paper
        
        # y = DepthwiseConv2D((3, 3), padding='valid',  depth_multiplier=3,  use_bias=False, name='depth1')(y)
        ##y = BatchNormalization(center=0)(y)

        y = self._densewise_sep_conv(y, 64, alpha, block_id=1) # spatial size: 32 x 32
        y = self._densewise_sep_conv(y, 128, alpha, strides = (2, 2), block_id=2) # spatial size: 32 x 32
        y = self._densewise_sep_conv(y, 128, alpha, block_id=3) # spatial size: 16 x 16
        y = self._densewise_sep_conv(y, 256, alpha, strides = (2, 2), block_id=4) # spatial size: 8 x 8
        y = self._densewise_sep_conv(y, 256, alpha, block_id=5) # spatial size: 8 x 8
        y = self._densewise_sep_conv(y, 512, alpha, strides = (2, 2), block_id=6) # spatial size: 4 x 4
        
        bid=7
        for _ in range(5):
            y = self._densewise_sep_conv(y, 512, alpha, block_id=bid) # spatial size: 4 x 4
            bid+=1
        y = self._densewise_sep_conv(y, 1024, alpha, strides = (2, 2), block_id=12) # spatial size: 2 x 2
        y = self._densewise_sep_conv(y, 1024, alpha, block_id=13) # strides = (2, 2) in the paper
        ##y = GlobalAveragePooling2D()(y)
        y = Flatten('channels_last')(y)
        y = Dense(100, activation='relu')(y)
        y = Dense(self.classes, activation='softmax')(y)
        ####y = Dense(units = 10)(y)
        ####y = Activation('softmax')(y)

        return Model(x, y, name = self.MODEL_NAME)

    def _densewise_sep_conv(self, x, filters, alpha, strides = (1, 1), block_id=0, depth_multiplier=1):
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
        y = ZeroPadding2D(	padding = (1, 1),
        					name='zero_%d_densewise_block' % block_id)(x)
        y = DepthwiseConv2D((3, 3), padding='valid',
        					depth_multiplier=depth_multiplier, 
        					use_bias=False,strides = strides,
        					name='depth_%d_densewise_block' % block_id)(y)
        y = BatchNormalization(	center=0,
        						name='batch_norm1_%d_densewise_block' % block_id)(y)
        y = Activation('relu', name='relu1_%d_densewise_block' % block_id)(y)
        y = Conv2D(	int(filters * alpha), (1, 1),
        			activation='relu', padding = 'valid',
        			name='conv_%d_densewise_block' % block_id)(y)
        y = BatchNormalization(	center=0,
        						name='batch_norm2_%d_densewise_block' % block_id)(y)
        y = Activation('relu', name='relu2_%d_densewise_block' % block_id)(y)
        return y

def main():
    '''
    Train the model defined above.
    '''

    dataset_name = 'cifar_10'
    model_name = 'MobileNet'
    image_size = 224
    channels = 3
    n_classes = 10
    epochs = 40
    batch_size = 32

    layers_conv, layers_hidden = 14, 1


    if(dataset_name == 'mnist'):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist(image_size, channels)
    elif(dataset_name == 'cifar_10'):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10(image_size, channels)
    elif(dataset_name == 'cifar_100'):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar100(image_size, channels)
    elif(dataset_name == 'fashion_mnist'):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_fashion_mnist(image_size, channels)

    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_val shape:', y_val.shape)
    print('y_test shape:', y_test.shape)
    print('Loaded Images ')


    #x_train, y_train = generate_aug(x_train, y_train)
    #train_generator = get_train_generator()

    #mobile_model = AlexNet(f"{dataset_name}_{model_name}", image_size, channels, n_classes)    
    mobile_model = MobileNet(f"{dataset_name}_{model_name}", image_size, channels, n_classes)
    #mobile_model = VGG16(f"{dataset_name}_{model_name}", image_size, channels, n_classes)
    #mobile_model = Inception(f"{dataset_name}_{model_name}", image_size, channels, n_classes)
    mobile_model.summary()

    mobile_model.compile()
    mobile_model.fit((x_train, y_train), (x_val, y_val), epochs, batch_size)

    '''
    loss_and_metrics = mobile_model.evaluate((x_train, y_train), batch_size = batch_size)
    print('\n\n[Evaluation on the train dataset]\n', loss_and_metrics, '\n\n')
    mobile_model.predict(x_train, y_train, batch_size = batch_size)

    
    loss_and_metrics = mobile_model.evaluate((x_val, y_val), batch_size = batch_size)
    print('[Evaluation on the vel dataset]\n', loss_and_metrics, '\n\n')
    mobile_model.predict(x_val, y_val, batch_size = batch_size)
    
    loss_and_metrics = mobile_model.evaluate((x_test, y_test), batch_size = batch_size)
    print('[Evaluation on the test dataset]\n', loss_and_metrics, '\n\n')
    mobile_model.predict(x_test, y_test, batch_size = batch_size)
    '''
    mobile_model.save_model(f"{dataset_name}_{model_name}")

    mobile_model.get_model().load_weights(f"files/{dataset_name}_{model_name}.h5")
    mobile_model.compile()

    import numpy as np
    predictions_valid = mobile_model.get_model().predict(x_test,  batch_size=batch_size, verbose=1)
    np.set_printoptions(precision=4)
    # np.set_printoptions(suppress=True)
    
    with open('files/Preds.txt', 'w+') as f:
        for i in predictions_valid:
            for j in i:
                f.write('%.4f ' % j)
            f.write("\n")
    

    with open('files/ConvLayersW.txt', 'a+') as file:
        file.write('%d\n' % (layers_conv));
    with open('files/HiddenLayersW.txt', 'a+') as file:
        file.write('%d\n' % (layers_hidden));


    c, h, s = 95, 97, 98
    #AlexNet 8, 10, 11
    #Mobilenet 95, 97, 98
    #Inception 87, 93, 96
    #VGG16 31, 33, 34
    with open('files/conf.txt', 'a+') as conf:
        id_layer = c
        list_mark = [0] * len(mobile_model.get_model().layers)
        layers = mobile_model.get_model().layers[id_layer]
        busca_em_profun(mobile_model.get_model(), layers, len(mobile_model.get_model().layers), conf, list_mark, id_layer)

        conf.write('flatten 1 %s\n' % layers.name)

        layers = mobile_model.get_model().layers[h]
        with open('files/HiddenLayersW.txt', 'a+') as f:
            print_hidden_soft_layer_W(f, layers, conf)
            #layers = mobile_model.get_model().layers[35]
            #print_hidden_soft_layer_W(f, layers, conf)

        layers = mobile_model.get_model().layers[h]
        with open('files/HiddenLayersB.txt', 'a+') as f:
            print_hidden_soft_layer_b(f, layers)
            #layers = mobile_model.get_model().layers[35]
            #print_hidden_soft_layer_b(f, layers)



        layers = mobile_model.get_model().layers[s]
        with open('files/SmrLayerW.txt', 'a+') as f:
            print_hidden_soft_layer_W(f, layers, conf)
        with open('files/SmrLayerB.txt', 'a+') as f:
            print_hidden_soft_layer_b(f, layers)

        conf.write('endconf 1 %s\n' % layers.name)

if __name__ == '__main__':
    main()