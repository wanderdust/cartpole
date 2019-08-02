from dqn.model import ModelVanilla
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras import backend as K
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,\
    Activation, BatchNormalization, GlobalAveragePooling2D, Lambda, Input


class DuelingNet(ModelVanilla):
    
    def build_model(self):
        model = Sequential()
        
        input = Input(shape=(self.state_size,))

        fc1 = Dense(512, activation='relu')(input)

        fc2 = Dense(256, activation='relu')(fc1)
        
        # Build the dueling net
        x = Dense(self.action_size + 1, activation='linear')(fc2)
        policy = Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True),
                        output_shape=(self.action_size,))(x)

        model = Model(input, policy)

        # compile the model
        model.compile(loss='mean_squared_error',
                  optimizer=RMSprop(lr=self.learning_rate,
                                    rho=0.95,
                                    epsilon=0.01),
                  metrics=['accuracy'])

        return model