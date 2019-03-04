# -*- coding: utf-8 -*-


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import optimizers

from base_model import BaseModel
from keras.models import Model

class VGG16(BaseModel):

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

        y = ZeroPadding2D((1, 1), name='zero1_stage1')(x)
        y = Conv2D(64, (3, 3), activation='relu', name='conv1_stage1')(y)
        y = ZeroPadding2D((1, 1), name='zero2_stage1')(y)
        y = Conv2D(64, (3, 3), activation='relu', name='conv2_stage1')(y)
        y = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_stage1')(y)

        y = ZeroPadding2D((1, 1), name='zero1_stage2')(y)
        y = Conv2D(128, 3, 3, activation='relu', name='conv1_stage2')(y)
        y = ZeroPadding2D((1, 1), name='zero2_stage2')(y)
        y = Conv2D(128, 3, 3, activation='relu', name='conv2_stage2')(y)
        y = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_stage2')(y)

        y = ZeroPadding2D((1, 1), name='zero1_stage3')(y)
        y = Conv2D(256, 3, 3, activation='relu', name='conv1_stage3')(y)
        y = ZeroPadding2D((1, 1), name='zero2_stage3')(y)
        y = Conv2D(256, 3, 3, activation='relu', name='conv2_stage3')(y)
        y = ZeroPadding2D((1, 1), name='zero3_stage3')(y)
        y = Conv2D(256, 3, 3, activation='relu', name='conv3_stage3')(y)
        y = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_stage3')(y)

        y = ZeroPadding2D((1, 1), name='zero1_stage4')(y)
        y = Conv2D(512, 3, 3, activation='relu', name='conv1_stage4')(y)
        y = ZeroPadding2D((1, 1), name='zero2_stage4')(y)
        y = Conv2D(512, 3, 3, activation='relu', name='conv2_stage4')(y)
        y = ZeroPadding2D((1, 1), name='zero3_stage4')(y)
        y = Conv2D(512, 3, 3, activation='relu', name='conv3_stage4')(y)
        y = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_stage4')(y)

        y = ZeroPadding2D((1, 1), name='zero1_stage5')(y)
        y = Conv2D(512, 3, 3, activation='relu', name='conv1_stage5')(y)
        y = ZeroPadding2D((1, 1), name='zero2_stage5')(y)
        y = Conv2D(512, 3, 3, activation='relu', name='conv2_stage5')(y)
        y = ZeroPadding2D((1, 1), name='zero3_stage5')(y)
        y = Conv2D(512, 3, 3, activation='relu', name='conv3_stage5')(y)
        y = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_stage5')(y)
        
        # Add Fully Connected Layer
        y = Flatten('channels_last')(y)
        y = Dense(100, activation='relu')(y)
        # model.add(Dropout(0.5)
        #y = Dense(4096, activation='relu')(y)
        # model.add(Dropout(0.5))
        y = Dense(self.classes, activation='softmax')(y)
        # Loads ImageNet pre-trained data
        # model.load_weights('vgg16_weights_th_dim_ordering_th_kernels.h5')
        # Truncate and replace softmax layer for transfer learning
        # model.layers.pop()
        # model.outputs = [model.layers[-1].output]
        # model.layers[-1].outbound_nodes = []
        # x = Dense(num_classes, activation='softmax')(x)
        # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
        # for layer in model.layers[:10]:
        #    layer.trainable = False
        # Learning rate is changed to 0.001
        return Model(x, y, name = self.model_name)