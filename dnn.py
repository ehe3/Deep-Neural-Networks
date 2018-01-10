from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

data = input_data.read_data_sets('MNIST_data', one_hot=True)

# DNN constants
IMAGE_SIZE = 784
HIDDEN_LAYER_1_SIZE = 100
HIDDEN_LAYER_2_SIZE = 25
CLASS_SIZE = 10
RELU_LEAK = 0.1
NUM_TRAIN_IMAGES = 55000
NUM_TEST_IMAGES = 10000
INIT_MEAN = 0
INIT_VAR = 0.1

class WeightsAndBiases:
  def __init__(self):
    # weight matrix and bias vector that acts on the first layer
    self.w1 = np.random.normal(INIT_MEAN, INIT_VAR, (IMAGE_SIZE, HIDDEN_LAYER_1_SIZE))
    self.b1 = np.random.normal(INIT_MEAN, INIT_VAR, (1, HIDDEN_LAYER_1_SIZE))
    # weight matrix and bias vector that acts on the second layer
    self.w2 = np.random.normal(INIT_MEAN, INIT_VAR, (HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE))
    self.b2 = np.random.normal(INIT_MEAN, INIT_VAR, (1, HIDDEN_LAYER_2_SIZE))
    # weight matrix and bias vector that acts on the third layer
    self.w3 = np.random.normal(INIT_MEAN, INIT_VAR, (HIDDEN_LAYER_2_SIZE, CLASS_SIZE))
    self.b3 = np.random.normal(INIT_MEAN, INIT_VAR, (1, CLASS_SIZE))

class LayerOutputs:
  def __init__(self, l1, r1, l2, r2, l3):
    # output after first layer before and after nonlinearity
    self.l1 = l1
    self.r1 = r1
    # output after second layer before and after nonlinearity
    self.l2 = l2
    self.r2 = r2
    # output after third layer before softmax
    self.l3 = l3

def sample_training_data(sample_size):
  idx = np.arange(NUM_TRAIN_IMAGES)
  np.random.shuffle(idx)
  return idx[:sample_size]

def leaky_relu(x):
  return max(RELU_LEAK * x, x)

def soft_max(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

def compute_cross_entropy(predicted, actual):
  predicted_probabilities = [np.log(x) for x in np.multiply(predicted, actual).sum(axis=1)]
  return -1 * sum(predicted_probabilities) / len(predicted_probabilities)

def forward_pass(images, labels, wb, full_pass=True, batch_size=None):
  if not full_pass:
    assert batch_size is not None
    sample_idx = sample_training_data(batch_size)
    sample_images, sample_labels = images[sample_idx,], labels[sample_idx,]
  else:
    sample_images, sample_labels = images, labels

  leaky_relu_vec = np.vectorize(leaky_relu)
  # first layer feedforward
  l1 = np.dot(sample_images, wb.w1) + wb.b1
  r1 = leaky_relu_vec(l1)
  # second layer feedforward
  l2 = np.dot(r1, wb.w2) + wb.b2
  r2 = leaky_relu_vec(l2)
  # third layer feedforward
  l3 = np.apply_along_axis(soft_max, 1, np.dot(r2, wb.w3) + wb.b3)
  # return values
  lo = LayerOutputs(l1, r1, l2, r2, l3)
  return sample_images, sample_labels, lo, compute_cross_entropy(l3, sample_labels)

def backward_pass(sample_images, sample_labels, wb, lo, batch_size, step_size):
  # third layer gradient calculations
  sm_grad = (lo.l3 - sample_labels)
  Db_3 = sm_grad.mean(axis=0)
  Dw_3 = np.dot(lo.r2.transpose(), sm_grad) / batch_size
  # third layer backpropogation
  wb.w3 -= step_size * Dw_3
  wb.b3 -= step_size * Db_3
  # second layer gradient calculations
  l2_grad = np.dot(sm_grad, wb.w3.transpose())
  r2_grad = np.multiply(np.where(lo.l2 > 0, 1, RELU_LEAK), l2_grad)
  Db_2 = r2_grad.mean(axis=0)
  Dw_2 = np.dot(lo.r1.transpose(), r2_grad) / batch_size
  # second layer backpropogation
  wb.w2 -= step_size * Dw_2
  wb.b2 -= step_size * Db_2
  # first layer gradient calculations
  l1_grad = np.dot(l2_grad, wb.w2.transpose())
  r1_grad = np.multiply(np.where(lo.l1 > 0, 1, RELU_LEAK), l1_grad)
  Db_1 = r1_grad.mean(axis=0)
  Dw_1 = np.dot(sample_images.transpose(), r1_grad) / batch_size
  # first layer backpropogation
  wb.w1 -= step_size * Dw_1
  wb.b1 -= step_size * Db_1
  # return updated weights
  return wb

# ceu: cross entropy update, prints cross entropy every ceu iterations
def train(iterations, images, labels, wb, batch_size, step_size, ceu=10):
  accumulated_cross_entropy = 0
  for i in range(1, iterations + 1):
    sample_images, sample_labels, lo, cross_entropy = forward_pass(images, labels, wb, full_pass=False, batch_size=batch_size)
    accumulated_cross_entropy += cross_entropy
    if i % ceu == 0:
      print('{}: {}'.format(i, accumulated_cross_entropy / ceu))
      accumulated_cross_entropy = 0
    wb = backward_pass(sample_images, sample_labels, wb, lo, batch_size, step_size)
  return wb

def calculate_error(wb):
  _, _, lo, cross_entropy = forward_pass(data.test.images, data.test.labels, wb)
  predictions = np.array([prob.argmax() for prob in lo.l3])
  correct_labels = np.array([label.argmax() for label in data.test.labels])
  accuracy = np.array(predictions == correct_labels)
  return (1 - (accuracy.sum() / len(accuracy))) * 100

def main():
  wb = WeightsAndBiases()
  train(200000, data.train.images, data.train.labels, wb, 64, 1e-4, ceu=100)
  print('Error rate: {0:.2f}%'.format(calculate_error(wb)))

if __name__ == "__main__":
  main()