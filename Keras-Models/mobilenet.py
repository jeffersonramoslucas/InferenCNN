# -*- coding: utf-8 -*-


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import optimizers

from base_model import BaseModel
from keras.models import Model

class MobileNet(BaseModel):

    def __init__(self, model_name, input_size, channels, classes):
        '''
        - Reference for hyperparameters
          => https://github.com/Zehaos/MobileNet/issues/13
        '''

        self.model_name = model_name
        self.input_size = input_size
        self.channels = channels
        self.classes = classes
        #self.callbacks = #[ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                     #                  patience = 30, verbose = 1)]
        optimizer = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0001,nesterov=False)#
        #optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #optimizer = optimizers.RMSprop(lr = 0.01)
        BaseModel.__init__(self, model = self.__build(), optimizer = optimizer, callbacks = self.__callbacks())


    def __callbacks(self):

        tensorboard = TensorBoard(log_dir=f'files/logs/')

        checkpoint = ModelCheckpoint(f'files/{self.model_name}.h5', 
                                    monitor='val_loss', verbose=2, 
                                    save_best_only=True, mode='min')

        early_stoping = EarlyStopping(monitor='val_loss', patience=5, 
                                    verbose=2, restore_best_weights=True)

        return [tensorboard, checkpoint, early_stoping]

    def __build(self):
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

        y = BatchNormalization(center=0, name='batch_norm_cnv_block')(y)

        # some layers have different strides from the papers considering the size of mnist
        y = Conv2D(32, (3, 3),padding='valid', strides=(2, 2), activation='relu', name='conv1_cnv_block')(y) # strides = (2, 2) in the paper
        '''*32 * alpha'''

        ##y = Activation('relu')(y)
        ##y = Conv2D(10, (11, 11), padding='same', activation='relu', name='conv2')(y)  # strides = (2, 2) in the paper
        ##y = Conv2D(10, (11, 11), padding='valid', activation='relu', name='conv3')(y)  # strides = (2, 2) in the paper
        ##y = Conv2D(10, (11, 11), padding='valid', activation='relu', name='conv4')(y)  # strides = (2, 2) in the paper
        
        # y = DepthwiseConv2D((3, 3), padding='valid',  depth_multiplier=3,  use_bias=False, name='depth1')(y)
        ##y = BatchNormalization(center=0)(y)

        y = self._densewise_sep_conv(y, 64, alpha, block_id=1) # spatial size: 32 x 32
        #y = self._densewise_sep_conv(y, 128, alpha, strides = (2, 2), block_id=2) # spatial size: 32 x 32
        #y = self._densewise_sep_conv(y, 128, alpha, block_id=3) # spatial size: 16 x 16
        #y = self._densewise_sep_conv(y, 256, alpha, strides = (2, 2), block_id=4) # spatial size: 8 x 8
        #y = self._densewise_sep_conv(y, 256, alpha, block_id=5) # spatial size: 8 x 8
        #y = self._densewise_sep_conv(y, 512, alpha, strides = (2, 2), block_id=6) # spatial size: 4 x 4
        
        #bid=7
        #for _ in range(5):
        #    y = self._densewise_sep_conv(y, 512, alpha, block_id=bid) # spatial size: 4 x 4
        #    bid+=1
        #y = self._densewise_sep_conv(y, 1024, alpha, strides = (2, 2), block_id=12) # spatial size: 2 x 2
        #y = self._densewise_sep_conv(y, 1024, alpha, block_id=13) # strides = (2, 2) in the paper
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
        y = ZeroPadding2D(  padding = (1, 1),
                            name='zero_%d_densewise_block' % block_id)(x)
        y = DepthwiseConv2D((3, 3), padding='valid',
                            depth_multiplier=depth_multiplier, 
                            use_bias=False,strides = strides,
                            name='depth_%d_densewise_block' % block_id)(y)
        y = BatchNormalization( center=0,
                                name='batch_norm1_%d_densewise_block' % block_id)(y)
        #y = Activation('relu')(y)
        y = Conv2D( int(filters * alpha), (1, 1),
                    activation='relu', padding = 'valid',
                    name='conv_%d_densewise_block' % block_id)(y)
        y = BatchNormalization( center=0,
                                name='batch_norm2_%d_densewise_block' % block_id)(y)
        #y = Activation('relu')(y)
        return y




