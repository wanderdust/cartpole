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
        
        input = Input((self.state_size))

        conv1 = Conv2D(filters=32, kernel_size=8, padding='valid', activation='relu', 
                        strides=(4,4), input_shape=self.state_size)(input)

        conv2 = Conv2D(filters=64, kernel_size=4, padding='valid', activation='relu',
                        strides=(2,2))(conv1)

        conv3 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',
                        strides=(1,1))(conv2)

        flatten = Flatten()(conv3)

        fc1 = Dense(512, activation='relu')(flatten)
        
        # Build the dueling net
        x = Dense(self.action_size + 1, activation='linear')(fc1)
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