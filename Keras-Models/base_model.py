from sklearn.metrics import confusion_matrix
from keras.utils import plot_model
import numpy as np
import utils

class BaseModel(object):
    def __init__(self, model, optimizer, callbacks = None):
        self.model = model
        self.callbacks = callbacks
        self.optimizer = optimizer

    def summary(self):
        self.model.summary()

    def load_weights(self, path):
        self.model.load_weights(path)

    def save(self, path):
        self.model.save(path)

    def save_model(self, model_name):
        model_json = self.model.to_json()
        with open(f'files/{model_name}.json', "w") as json_file:
                 json_file.write(model_json)

        # serialize weights to HDF5
        #self.model.save_weights(f'files/{model_name}.h5')
        #print("Saved model to disk")

    def get_model(self):
        return self.model

    def compile(self):
        self.model.compile(optimizer = self.optimizer, loss = 'categorical_crossentropy',
                         metrics = ['accuracy'])

    def fit(self, training_data, validation_data, epochs, batch_size):
        x_train, y_train = training_data
        x_val, y_val = validation_data

        hist = self.model.fit(x_train, y_train, epochs = epochs,
                              batch_size = batch_size,
                              validation_data = (x_val, y_val), callbacks = self.callbacks)
        return hist

    def fit_generator(self, training_data, validation_data, epochs, batch_size):
        x_train, y_train = training_data
        x_val, y_val = validation_data

        train_datagen = utils.get_train_generator(x_train, y_train,
                                                  batch_size = batch_size)
        val_datagen = utils.get_val_generator(x_val, y_val,
                                              batch_size = batch_size)
        hist = self.model.fit_generator(train_datagen,
                                        callbacks = self.callbacks,
                                        steps_per_epoch = x_train.shape[0] // batch_size,
                                        epochs = epochs, validation_data = val_datagen,
                                        validation_steps = x_val.shape[0] // batch_size)
        return hist

    def evaluate(self, eval_data, batch_size = 32):
        x, y = eval_data
        loss_and_metrics = self.model.evaluate(x, y,
                                               batch_size = batch_size)
        return loss_and_metrics

    def predict(self, x, y, batch_size = None, verbose = 0, steps = None):
        predictions = self.model.predict(x, batch_size, verbose, steps)
        predictions_classes = np.argmax(predictions,axis=1)
        labels = np.argmax(y,axis=1)
        conf_mat = confusion_matrix(labels,predictions_classes)
        print(conf_mat)
        

    def save_model_as_image(self, path):
        plot_model(self.model, to_file = path, show_shapes = True)