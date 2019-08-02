from dqn.replayBuffer import ReplayBuffer
from dqn.model import ModelVanilla
from dqn.dueling import DuelingNet
import numpy as np

class DQNAgent:
  def __init__(self, dueling=False, action_size=6):

    # Hyperparameters
    self.learning_rate = 0.00025
    self.gamma = 0.99 # discount rate
    self.epsilon = 1 # exploration
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.1
    
    # Model's parameters
    self.k_frames = 4
    self.state_size = 4
    self.action_size = action_size

    # Model with fixed weights w-'s parameters
    self.c_steps = 10000 # how often w- gets updated
    self.c_steps_counter = 0

    # Replay memory
    self.buffer_size = 100000
    self.batch_size = 64
    self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

    # Check if we are going to be using dueling nets
    if(dueling):
      self.model = DuelingNet(self.state_size, self.action_size, self.learning_rate).model
      # Model with fixed weights w-
      self.model_f = DuelingNet(self.state_size, self.action_size, self.learning_rate).model
    else:
      self.model = ModelVanilla(self.state_size, self.action_size, self.learning_rate).model
      # Model with fixed weights w-
      self.model_f = ModelVanilla(self.state_size, self.action_size, self.learning_rate).model


  def act(self, state):
    """
    Choose action using greedy policy
    """
    if np.random.rand() <= self.epsilon:
      # select random action
      return np.random.choice(np.arange(self.action_size))

    # select greedy action
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])

  def update_model_f(self):
    """
    Updates fixed weights of the 'model_f' every C steps.
    """
    if self.c_steps_counter == self.c_steps:
      self.model_f.set_weights(self.model.get_weights()) 
      self.c_steps_counter = 0
    else:
      self.c_steps_counter += 1

  def learn(self):
    """
    1. Obtain a batch of random samples
    2. Set target y = r + gamma*max q(s,a,w-)
    3. Update weights (forward pass -> Gradient descent)
    4. Update fixed w after C steps w- <- w
    """
    if len(self.memory.memory) < self.batch_size:
      return 
      
    minibatch = self.memory.sample(self.batch_size)
    
    for state, action, reward, next_state, done in minibatch:
      target = reward
      if not done:
        target = reward + self.gamma * np.amax(self.model_f.predict(next_state)[0])

      # make the agent to approximately map
      # the current state to future discounted reward
      target_f = self.model.predict(state)
      target_f[0][action] = target

      self.model.fit(state, target_f, epochs=1, verbose=0)
      self.update_model_f()

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay  

  def save_weights(self, filename="best_model"):
    self.model.save_weights('saved_models/'+ filename + '.h5')

  def load_weights(self, filename):
    self.model.load_weights('saved_models/'+ filename + '.h5')
    self.model_f.load_weights('saved_models/'+ filename + '.h5')
