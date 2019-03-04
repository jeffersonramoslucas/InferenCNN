from keras.models import Sequential

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import optimizers

from base_model import BaseModel
from keras.models import Model

class AlexNet(BaseModel):

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

        Parameters:s
          img_rows, img_cols - resolution of inputs
          channel - 1 for grayscale, 3 for color
          num_classes - number of categories for our classification task
        """
        #model = Sequential()

        x = Input(shape=(self.channels, self.input_size, self.input_size))


        # 1st Convolutional Layer
        y = Conv2D(filters=96, kernel_size=(11,11),strides=(4,4), padding='valid', activation='relu')(x)
		# Pooling 
        y = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(y)
		# Batch Normalisation before passing it to the next layer
		#model.add(BatchNormalization())

		# 2nd Convolutional Layer
        y = Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid', activation='relu')(y)
		# Pooling
        y = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(y)
		# Batch Normalisation
		#model.add(BatchNormalization())

		# 3rd Convolutional Layer
        y = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(y)
		# Batch Normalisation
		#model.add(BatchNormalization())

		# 4th Convo
        y = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(y)
		# Batch Normalisation
		#model.add(BatchNormalization())

		# 5th Convolutional Layer
        y = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(y)
		# Pooling
        y = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(y)
		# Batch Normalisation
		#model.add(BatchNormalization())
        

        # Passing it to a dense layer
        y = Flatten('channels_last')(y)
		# 1st Dense Layer
        y = Dense(4096, activation='relu')(y)
		# Add Dropout to prevent overfitting
		#model.add(Dropout(0.4))
		# Batch Normalisation
		#model.add(BatchNormalization())

		# 2nd Dense Layer
		#model.add(Dense(4096, activation='relu'))
		# Add Dropout
		#model.add(Dropout(0.4))
		# Batch Normalisation
		#model.add(BatchNormalization())

		# 3rd Dense Layer
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