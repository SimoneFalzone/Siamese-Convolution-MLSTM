from tensorflow.keras.layers import LSTM, Lambda, Embedding, Conv1D, MaxPooling1D, Dropout, BatchNormalization, \
     Dense
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K
import os
from tensorflow.keras.initializers import RandomNormal


class CNN_MLSTM():
    history = None
    callbacks = []
    checkpoint_path = './checkpoints/cp.ckpt'
    saved_weights = './checkpoints/my_checkpoint'
    model_folder = './model_save/'

    def __init__(self, earlystopping: bool = False, kernel_regularizer=None, embedding_layer: Embedding = None,
                 n_hidden_states: int = 50, input_dimension: int = 300):

        if earlystopping:
            # Which parameters to use?! Paper only mentions early stopping without parameters :O
            callback = EarlyStopping(monitor='val_pearson_correlation', mode='max', patience=20)
            self.callbacks.append(callback)
        elif not earlystopping:
            pass
        # self.callbacks.append(ModelCheckpoint(filepath=self.checkpoint_path,
        #                                       save_weights_only=True,
        #                                       verbose=1))

        # create left and right inputs of siamese architecture
        left_input = Input(shape=(input_dimension,), name='input_left')
        right_input = Input(shape=(input_dimension,), name='input_right')

        left_embedding = embedding_layer(left_input)
        right_embedding = embedding_layer(right_input)

        # 1st CNN layer with max-pooling
        conv1 = Conv1D(256, 7, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),
                       bias_initializer=RandomNormal(mean=0.0, stddev=0.05), activation='relu')
        conv1_left = conv1(left_embedding)
        conv1_right = conv1(right_embedding)
        pool1 = MaxPooling1D(pool_size=3)
        pool1_left = pool1(conv1_left)
        pool1_right = pool1(conv1_right)

        # 2nd CNN layer with max-pooling
        conv2 = Conv1D(256, 7, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),
                       bias_initializer=RandomNormal(mean=0.0, stddev=0.05), activation='relu')
        conv2_left = conv2(pool1_left)
        conv2_right = conv2(pool1_right)
        pool2 = MaxPooling1D(pool_size=3)
        pool2_left = pool2(conv2_left)
        pool2_right = pool2(conv2_right)

        # 3rd CNN layer without max-pooling
        conv3 = Conv1D(256, 3, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),
                       bias_initializer=RandomNormal(mean=0.0, stddev=0.05), activation='relu')
        conv3_left = conv3(pool2_left)
        conv3_right = conv3(pool2_right)

        lstm = LSTM(n_hidden_states, name="lstm_layer",  dropout=0.2, recurrent_dropout=0.3)
        lstm_left = lstm(conv3_left)
        lstm_right = lstm(conv3_right)

        similarity_function = lambda x: K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        merge = Lambda(function=similarity_function, output_shape=lambda x: x[0], name='similarity')(
            [lstm_left, lstm_right])

        output = Dense(1, activation='sigmoid')(merge)

        self.model = Model([left_input, right_input], output)
        self.__compile()
        print(self.model.summary())

    def __compile(self):
        learning_rate = 0.001
        momentum = 0.9
        optimizer = optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        self.model.compile(loss=losses.BinaryCrossentropy(), optimizer=optimizer,
                           metrics=['accuracy'])

    def fit(self, left_data, right_data, targets, validation_data, epochs=5, batch_size=32):
        # lrate = LearningRateScheduler(step_decay)
        # self.callbacks.append(lrate)

        self.history = self.model.fit([left_data, right_data], targets, batch_size=batch_size, epochs=epochs,
                                      callbacks=self.callbacks,
                                      validation_data=validation_data)

    def save(self):
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(self.model_folder + 'model.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(self.model_folder + 'model.h5')
        print('Saved model to disk')

    def load(self):
        self.model.load_weights(self.saved_weights)

    def predict(self, left_data, right_data):
        return self.model.predict([left_data, right_data])

    def load_pretrained_weights(self, model_weights_path=None):
        if model_weights_path is None:
            if os.path.exists(self.saved_weights):
                model_weights_path = self.saved_weights
            else:
                return
        self.model.load_weights(model_weights_path)
        self.__compile()
