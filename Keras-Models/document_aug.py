import keras
from keras.preprocessing.image import ImageDataGenerator



def document_aug(
			x,
			y,
			aug_size,
			aug_out_dir
			):

	train_datagen = ImageDataGenerator(
					    featurewise_center=False,  # set input mean to 0 over the dataset
					    samplewise_center=False,  # set each sample mean to 0
					    featurewise_std_normalization=False,  # divide inputs by std of the dataset
					    samplewise_std_normalization=False,  # divide each input by its std
					    zca_whitening=False,  # apply ZCA whitening
					    rotation_range=2,  # randomly rotate images in the range (degrees, 0 to 180)
					    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
					    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
					    zoom_range=[.8, 1],
					    channel_shift_range=30
					    )

	train_datagen.fit(x = x,
			    augment=True,
			    rounds=aug_size)

	train_generator = train_datagen.flow(x, y, batch_size=64, seed=11, save_to_dir=aug_out_dir, save_prefix='data_aug',save_format='png')
	#train_generator = train_datagen.flow_from_directory(directory='train/',target_size=(image_size,image_size),classes=['false_images','crlv'],batch_size=64,seed=11,save_to_dir = 'output/', save_prefix='train')

	return train_generator

