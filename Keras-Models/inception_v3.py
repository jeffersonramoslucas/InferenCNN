# -*- coding: utf-8 -*-


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, concatenate
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers

from base_model import BaseModel
from keras.models import Model

class Inception(BaseModel):

    def __init__(self, MODEL_NAME, input_size, channels, classes):
        '''
        - Reference for hyperparameters
          => https://github.com/Zehaos/MobileNet/issues/13
        '''

        self.MODEL_NAME = MODEL_NAME
        self.input_size = input_size
        self.channels = channels
        self.classes = classes
        callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                       patience = 30, verbose = 1)]
        optimizer = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001,nesterov=False)#
        #optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #optimizer = optimizers.RMSprop(lr = 0.01)
        BaseModel.__init__(self, model = self._build(), optimizer = optimizer)

    def _build(self):
        """VGG 16 Model for Keras

        Model Schema is based on
        https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

        ImageNet Pretrained Weights
        https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing

        Parameters:
          img_rows, img_cols - resolution of inputs
          channel - 1 for grayscale, 3 for color
          num_classes - number of categories for our classification task
        """

        x = Input(shape=(self.channels, self.input_size, self.input_size))

        conv1_7x7_s2 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1_7x7_s2')(x)
        zero_pad_7x7_s2 = ZeroPadding2D(padding=(1, 1), name='zero_pad_7x7_s2')(conv1_7x7_s2)

        pool1_3x3_s2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pool1_3x3_s2')(zero_pad_7x7_s2)

        conv2_reduce_3x3 = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv2_reduce_3x3')(pool1_3x3_s2)
        conv2_3x3 = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv2_3x3')(conv2_reduce_3x3)
        zero_pad_3x3 = ZeroPadding2D(padding=(1, 1), name='zero_pad_3x3')(conv2_3x3)

        pool2_3x3_s2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pool2_3x3_s2')(zero_pad_3x3)

        ######################### INCEPTION 1 #########################
        conv1_inception_3a_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv1_inception_3a_1x1')(
            pool2_3x3_s2)

        conv2_inception_3a_reduce_3x3 = Conv2D(96, (1, 1), padding='same', activation='relu',
                                               name='conv2_inception_3a_reduce_3x3')(pool2_3x3_s2)
        conv2_inception_3a_3x3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_inception_3a_3x3')(
            conv2_inception_3a_reduce_3x3)

        conv3_inception_3a_reduce_5x5 = Conv2D(16, (1, 1), padding='same', activation='relu',
                                               name='conv3_inception_3a_reduce_5x5')(pool2_3x3_s2)
        conv3_inception_3a_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv3_inception_3a_5x5')(
            conv3_inception_3a_reduce_5x5)

        inception_3a_pool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='inception_3a_pool_3x3')(
            pool2_3x3_s2)
        conv4_inception_3a_1x1 = Conv2D(32, (1, 1), padding='same', activation='relu', name='conv4_inception_3a_1x1')(
            inception_3a_pool_3x3)

        inception_3a_output = concatenate(
            [conv1_inception_3a_1x1, conv2_inception_3a_3x3, conv3_inception_3a_5x5, conv4_inception_3a_1x1],
            axis=1, name='concatenate_inception_3a')

        ######################### INCEPTION 2 #########################
        conv1_inception_3b_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='conv1_inception_3b_1x1')(
            inception_3a_output)

        conv2_inception_3b_reduce_3x3 = Conv2D(128, (1, 1), padding='same', activation='relu',
                                               name='conv2_inception_3b_reduce_3x3')(inception_3a_output)
        conv2_inception_3b_3x3 = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv2_inception_3b_3x3')(
            conv2_inception_3b_reduce_3x3)

        conv3_inception_3b_reduce_5x5 = Conv2D(32, (1, 1), padding='same', activation='relu',
                                               name='conv3_inception_3b_reduce_5x5')(inception_3a_output)
        conv3_inception_3b_5x5 = Conv2D(96, (5, 5), padding='same', activation='relu', name='conv3_inception_3b_5x5')(
            conv3_inception_3b_reduce_5x5)

        inception_3b_pool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='inception_3b_pool_3x3')(
            inception_3a_output)
        conv4_inception_3b_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv4_inception_3b_1x1')(
            inception_3b_pool_3x3)

        inception_3b_output = concatenate(
            [conv1_inception_3b_1x1, conv2_inception_3b_3x3, conv3_inception_3b_5x5, conv4_inception_3b_1x1],
            axis=1, name='concatenate_inception_3b')

        inception_zero_pad_3b_output = ZeroPadding2D(padding=(1, 1), name='inception_zero_pad_3b_output')(
            inception_3b_output)
        pool3_3x3_s2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pool3_3x3_s2')(
            inception_zero_pad_3b_output)

        ######################### INCEPTION 3 #########################
        conv1_inception_4a_1x1 = Conv2D(192, (1, 1), padding='same', activation='relu', name='conv1_inception_4a_1x1')(
            pool3_3x3_s2)

        conv2_inception_4a_reduce_3x3 = Conv2D(96, (1, 1), padding='same', activation='relu',
                                               name='conv2_inception_4a_reduce_3x3')(pool3_3x3_s2)
        conv2_inception_4a_3x3 = Conv2D(208, (3, 3), padding='same', activation='relu', name='conv2_inception_4a_3x3')(
            conv2_inception_4a_reduce_3x3)

        conv3_inception_4a_reduce_5x5 = Conv2D(16, (1, 1), padding='same', activation='relu',
                                               name='conv3_inception_4a_reduce_5x5')(pool3_3x3_s2)
        conv3_inception_4a_5x5 = Conv2D(48, (5, 5), padding='same', activation='relu', name='conv3_inception_4a_5x5')(
            conv3_inception_4a_reduce_5x5)

        inception_4a_pool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='inception_4a_pool_3x3')(
            pool3_3x3_s2)
        conv4_inception_4a_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv4_inception_4a_1x1')(
            inception_4a_pool_3x3)

        inception_4a_output = concatenate(
            [conv1_inception_4a_1x1, conv2_inception_4a_3x3, conv3_inception_4a_5x5, conv4_inception_4a_1x1],
            axis=1, name='concatenate_inception_4a')

        ######################### INCEPTION 4 #########################
        conv1_inception_4b_1x1 = Conv2D(160, (1, 1), padding='same', activation='relu', name='conv1_inception_4b_1x1')(
            inception_4a_output)

        conv2_inception_4b_reduce_3x3 = Conv2D(112, (1, 1), padding='same', activation='relu',
                                               name='conv2_inception_4b_reduce_3x3')(inception_4a_output)
        conv2_inception_4b_3x3 = Conv2D(224, (3, 3), padding='same', activation='relu', name='conv2_inception_4b_3x3')(
            conv2_inception_4b_reduce_3x3)

        conv3_inception_4b_reduce_5x5 = Conv2D(24, (1, 1), padding='same', activation='relu',
                                               name='conv3_inception_4b_reduce_5x5')(inception_4a_output)
        conv3_inception_4b_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='conv3_inception_4b_5x5')(
            conv3_inception_4b_reduce_5x5)

        inception_4b_pool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='inception_4b_pool_3x3')(
            inception_4a_output)
        conv4_inception_4b_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv4_inception_4b_1x1')(
            inception_4b_pool_3x3)

        inception_4b_output = concatenate(
            [conv1_inception_4b_1x1, conv2_inception_4b_3x3, conv3_inception_4b_5x5, conv4_inception_4b_1x1],
            axis=1, name='concatenate_inception_4b')

        ######################### LOSS 1 #########################
        loss1_pool_5x5 = MaxPooling2D((5, 5), strides=(3, 3), name='loss1_pool_5x5')(inception_4b_output)
        loss1_conv = Conv2D(128, (1, 1), padding='same', activation='relu', name='loss1_conv')(loss1_pool_5x5)
        loss1_flat = Flatten('channels_last')(loss1_conv)
        loss1_fc = Dense(1024, activation='relu', name='loss1_fc')(loss1_flat)
        out1_classifier_act = Dense(10, activation='softmax', name='out1_classifier_act')(loss1_fc)

        ######################### INCEPTION 5 #########################
        conv1_inception_4c_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='conv1_inception_4c_1x1')(
            inception_4b_output)

        conv2_inception_4c_reduce_3x3 = Conv2D(128, (1, 1), padding='same', activation='relu',
                                               name='conv2_inception_4c_reduce_3x3')(inception_4b_output)
        conv2_inception_4c_3x3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv2_inception_4c_3x3')(
            conv2_inception_4c_reduce_3x3)

        conv3_inception_4c_reduce_5x5 = Conv2D(24, (1, 1), padding='same', activation='relu',
                                               name='conv3_inception_4c_reduce_5x5')(inception_4b_output)
        conv3_inception_4c_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='conv3_inception_4c_5x5')(
            conv3_inception_4c_reduce_5x5)

        inception_4c_pool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='inception_4c_pool_3x3')(
            inception_4b_output)
        conv4_inception_4c_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv4_inception_4c_1x1')(
            inception_4c_pool_3x3)

        inception_4c_output = concatenate(
            [conv1_inception_4c_1x1, conv2_inception_4c_3x3, conv3_inception_4c_5x5, conv4_inception_4c_1x1],
            axis=1, name='concatenate_inception_4c')

        ######################### INCEPTION 6 #########################
        conv1_inception_4d_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='conv1_inception_4d_1x1')(
            inception_4c_output)

        conv2_inception_4d_reduce_3x3 = Conv2D(128, (1, 1), padding='same', activation='relu',
                                               name='conv2_inception_4d_reduce_3x3')(inception_4c_output)
        conv2_inception_4d_3x3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv2_inception_4d_3x3')(
            conv2_inception_4d_reduce_3x3)

        conv3_inception_4d_reduce_5x5 = Conv2D(24, (1, 1), padding='same', activation='relu',
                                               name='conv3_inception_4d_reduce_5x5')(inception_4c_output)
        conv3_inception_4d_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='conv3_inception_4d_5x5')(
            conv3_inception_4d_reduce_5x5)

        inception_4d_pool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='inception_4d_pool_3x3')(
            inception_4c_output)
        conv4_inception_4d_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv4_inception_4d_1x1')(
            inception_4d_pool_3x3)

        inception_4d_output = concatenate(
            [conv1_inception_4d_1x1, conv2_inception_4d_3x3, conv3_inception_4d_5x5, conv4_inception_4d_1x1],
            axis=1, name='concatenate_inception_4d')

        ######################### LOSS 2 #########################
        loss2_pool_5x5 = MaxPooling2D((5, 5), strides=(3, 3), name='loss2_pool_5x5')(inception_4d_output)
        loss2_conv = Conv2D(128, (1, 1), padding='same', activation='relu', name='loss2_conv')(loss2_pool_5x5)
        loss2_flat = Flatten('channels_last')(loss2_conv)
        loss2_fc = Dense(1024, activation='relu', name='loss2_fc')(loss2_flat)
        out2_classifier_act = Dense(10, activation='softmax', name='out2_classifier_act')(loss2_fc)

        ######################### INCEPTION 7 #########################
        conv1_inception_4e_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu', name='conv1_inception_4e_1x1')(
            inception_4d_output)

        conv2_inception_4e_reduce_3x3 = Conv2D(160, (1, 1), padding='same', activation='relu',
                                               name='conv2_inception_4e_reduce_3x3')(inception_4d_output)
        conv2_inception_4e_3x3 = Conv2D(320, (3, 3), padding='same', activation='relu', name='conv2_inception_4e_3x3')(
            conv2_inception_4e_reduce_3x3)

        conv3_inception_4e_reduce_5x5 = Conv2D(32, (1, 1), padding='same', activation='relu',
                                               name='conv3_inception_4e_reduce_5x5')(inception_4d_output)
        conv3_inception_4e_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='conv3_inception_4e_5x5')(
            conv3_inception_4e_reduce_5x5)

        inception_4e_pool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='inception_4e_pool_3x3')(
            inception_4d_output)
        conv4_inception_4e_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='conv4_inception_4e_1x1')(
            inception_4e_pool_3x3)

        inception_4e_output = concatenate(
            [conv1_inception_4e_1x1, conv2_inception_4e_3x3, conv3_inception_4e_5x5, conv4_inception_4e_1x1],
            axis=1, name='concatenate_inception_4e')

        inception_zero_pad_4e_output = ZeroPadding2D(padding=(1, 1), name='inception_zero_pad_4e_output')(
            inception_4e_output)
        pool4_3x3_s2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pool4_3x3_s2')(
            inception_zero_pad_4e_output)

        ######################### INCEPTION 8 #########################
        conv1_inception_5a_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu', name='conv1_inception_5a_1x1')(
            pool4_3x3_s2)

        conv2_inception_5a_reduce_3x3 = Conv2D(160, (1, 1), padding='same', activation='relu',
                                               name='conv2_inception_5a_reduce_3x3')(pool4_3x3_s2)
        conv2_inception_5a_3x3 = Conv2D(320, (3, 3), padding='same', activation='relu', name='conv2_inception_5a_3x3')(
            conv2_inception_5a_reduce_3x3)

        conv3_inception_5a_reduce_5x5 = Conv2D(32, (1, 1), padding='same', activation='relu',
                                               name='conv3_inception_5a_reduce_5x5')(pool4_3x3_s2)
        conv3_inception_5a_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='conv3_inception_5a_5x5')(
            conv3_inception_5a_reduce_5x5)

        inception_5a_pool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='inception_5a_pool_3x3')(
            pool4_3x3_s2)
        conv4_inception_5a_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='conv4_inception_5a_1x1')(
            inception_5a_pool_3x3)

        inception_5a_output = concatenate(
            [conv1_inception_5a_1x1, conv2_inception_5a_3x3, conv3_inception_5a_5x5, conv4_inception_5a_1x1],
            axis=1, name='concatenate_inception_5a')

        ######################### INCEPTION 9 #########################
        conv1_inception_5b_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu', name='conv1_inception_5b_1x1')(
            inception_5a_output)

        conv2_inception_5b_reduce_3x3 = Conv2D(160, (1, 1), padding='same', activation='relu',
                                               name='conv2_inception_5b_reduce_3x3')(inception_5a_output)
        conv2_inception_5b_3x3 = Conv2D(320, (3, 3), padding='same', activation='relu', name='conv2_inception_5b_3x3')(
            conv2_inception_5b_reduce_3x3)

        conv3_inception_5b_reduce_5x5 = Conv2D(32, (1, 1), padding='same', activation='relu',
                                               name='conv3_inception_5b_reduce_5x5')(inception_5a_output)
        conv3_inception_5b_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='conv3_inception_5b_5x5')(
            conv3_inception_5b_reduce_5x5)

        inception_5b_pool_3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='inception_5b_pool_3x3')(
            inception_5a_output)
        conv4_inception_5b_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='conv4_inception_5b_1x1')(
            inception_5b_pool_3x3)

        inception_5b_output = concatenate(
            [conv1_inception_5b_1x1, conv2_inception_5b_3x3, conv3_inception_5b_5x5, conv4_inception_5b_1x1],
            axis=1, name='concatenate_inception_5b')
        ####

        ######################### DENSE LAYERS #########################
        output_flat = Flatten('channels_last')(inception_5b_output)
        output_fc = Dense(200, activation='relu', name='output_fc')(output_flat)
        output_classifier_act = Dense(10, activation='softmax', name='output_classifier_act')(output_fc)

        return Model(inputs=x, outputs=[out1_classifier_act,out2_classifier_act,output_classifier_act], name = self.MODEL_NAME)