import tensorflow as tf
import time
tf.config.run_functions_eagerly(True)

class NeuralNetworkTf(tf.keras.Sequential):

  def __init__(self, sizes, random_state=1):
    super().__init__()
    self.sizes = sizes
    self.random_state = random_state
    tf.random.set_seed(random_state)
    
    for i in range(0, len(sizes)):
      if i == len(sizes) - 1:
        self.add(tf.keras.layers.Dense(sizes[i], activation='softmax')) # chaning the output layer activation function to softmax
      else:
        self.add(tf.keras.layers.Dense(sizes[i], activation='sigmoid')) # changing the hidden layers activation function to sigmoid

  def compile_and_fit(self, x_train, y_train, x_val=None, y_val=None, epochs=50, learning_rate=0.01, batch_size=1): # adding x_val and y_val as parameters to be able to extract the metrics of the model on the validation data as well
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    eval_metrics = ['accuracy']

    self.compile(optimizer=optimizer, loss=loss_function, metrics=eval_metrics)

    start_time = time.time()
    history = {'accuracy':[], 'val_accuracy':[]} # creating a history dict. to store the metrics of the model

    for epoch in range(epochs):
      train_loss, train_accuracy = self.train_on_batch(x_train, y_train)
      history['accuracy'].append(train_accuracy)
      if x_val is not None:
        val_loss, val_accuracy = self.evaluate(x_val, y_val)
        history['val_accuracy'].append(val_accuracy)
      current_learning_rate = learning_rate if not isinstance(learning_rate, TimeBasedLearningRate) else learning_rate(epoch)
      print(f'Epoch: {epoch + 1}, '
            f'learning rate: {current_learning_rate:.4f}, '
            f'train accuracy: {train_accuracy:.4f}, '
            f'val accuracy: {val_accuracy:.4f}')
        
    return history

class TimeBasedLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
  '''TODO: Implement a time-based learning rate that takes as input a 
  positive integer (initial_learning_rate) and at each step reduces the
  learning rate by 1 until minimal learning rate of 1 is reached.
    '''
  def __init__(self, initial_learning_rate):
    super().__init__()
    self.initial_learning_rate = initial_learning_rate # setting the initial learning rate

  def __call__(self, step):
    return tf.math.maximum(1, self.initial_learning_rate - step) # as per the task, ensuring the learning rate doesn't go below 1
  
  def get_config(self):
    return {'initial_learning_rate': self.initial_learning_rate} 
  
  
  
  '''Commenting on the mistakes and what was missing in the Network class:
  1. The activation functions for the hidden layers and output layer were missplaced. As per usual methods and also our classification case, the output layer should have a softmax function and for the hidden layers should be sigmoid functions. This allows for the output layer to show us how likely a data point is to be in a certain class and the hidden layers to show us how likely a data point is to be in a certain class given the previous layer.
  2. The loss function previously was Binary Cross Entropy, which is used for binary classification. For multi-class classification, we should use Categorical Cross Entropy. BCE calculates the average of the loss for each data point, while CCE calculates the sum of the loss for each data point, which is what we want since we are dealing with a multi-class classification problem.
  3. The validation data was substituted with x_val and y_val in order to be able to extract the metrics of the model on the validation data as well. In addition a history dictionary was created in order to store the metrics of the model on the training and validation data. In addition x_train and x_val were flattened in order to be able to be fed into the model.
  '''

    
