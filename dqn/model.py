from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import RMSprop
import numpy as np


class ModelVanilla:
  def __init__(self, state_size, action_size, learning_rate):
    self.state_size = state_size # size tuple
    self.action_size = action_size
    self.learning_rate = learning_rate

    self.model = self.build_model()

  def build_model(self):
    model = Sequential()
    
    model.add(Dense(512, activation='relu', input_dim=self.state_size))
    model.add(Dense(256, activation='relu'))

    model.add(Dense(self.action_size))

    # compile the model
    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop(lr=self.learning_rate,
                                    rho=0.95,
                                    epsilon=0.01),
                  metrics=['accuracy'])

    return model

  @staticmethod
  def state_to_tensor(state):
    """
    Convert state to tensor
    """
    state = np.asarray(state)
    state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])

    return state



