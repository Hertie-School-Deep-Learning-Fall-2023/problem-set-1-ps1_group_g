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
        self.add(tf.keras.layers.Dense(sizes[i], activation='softmax'))
      else:
        self.add(tf.keras.layers.Dense(sizes[i], activation='sigmoid'))

  def compile_and_fit(self, x_train, y_train, x_val=None, y_val=None, epochs=50, learning_rate=0.01, batch_size=1):
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    eval_metrics = ['accuracy']

    self.compile(optimizer=optimizer, loss=loss_function, metrics=eval_metrics)

    start_time = time.time()
    history = {'accuracy':[], 'val_accuracy':[]}

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
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step):
    return tf.math.maximum(1, self.initial_learning_rate - step)
  
  def get_config(self):
    return {'initial_learning_rate': self.initial_learning_rate}
  
  
  
  '''Commenting on the mistakes in the Network class:
  1. The activation function in the Network class: Considering that we are dealing with a multi-class classification problem, the activation function of the output layer should be softmax, whearas the activation function of the hidden layers can be either sigmoid or relu. The softmax activation function is usually implemented in the output layers as its result can be interpreted as a probability distribution.
  2. The loss function in the Network class: As we are dealing with a multi-class classification problem, the most appropriate loss function would be categorical crossentropy. The current loss function - binary crossentropy is used for binary classification problems and if implemented in a multi-class classification problem, it would result in a very high loss value and would not allow the model to learn or perform well.
  3. Validation data: Initially, the validation data argument was not passed into the fit method. This would result in the model not being validated and consecutively not allowing the model to learn well.'''

    
