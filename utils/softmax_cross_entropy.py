from scipy.special import softmax
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions))/N
    return ce

def cross_entropy_with_logits(logits, labels):
    ce = []
    for z,y in zip(logits, labels):
        loss = -np.sum(y * (z - np.max(z) - np.log(np.sum(np.exp(z - np.max(z))))))
        ce.append(loss)
    return ce

# X = np.arange(12).reshape(3, 4)
# y = one_hot(np.arange(3), 4)
X = np.array([[ -2.2225,   1.9737],
        [ -1.7488,   1.5965],
        [ -0.9455,   0.9125],
        [ -0.6766,   0.5128],
        [ -1.3384,   1.1606],
        [ -0.9431,   0.8055],
        [ -2.0609,   1.8724],
        [ -2.4676,   2.4401],
        [ -2.8094,   2.7259],
        [-58.2041,  57.5718]])
y = one_hot(np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0]), 2)

print("X: \n", X)
print("y: \n", y)

ce = cross_entropy(softmax(X, axis=-1), y)
ce2 = cross_entropy_with_logits(X, y)
# print("sf_pred: ", softmax(X, axis=-1))

print("CE: ", ce)
print("CE2: ", ce2)


## Tensorflow version ##
input = tf.placeholder(tf.float32, None)
label = tf.placeholder(tf.float32, None)
loss = tf.losses.softmax_cross_entropy(label, input)
nn_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=input)
sf_pred = tf.nn.softmax(logits=input)

with tf.Session() as sess:
    tf_pred = sess.run(sf_pred, {input: X})
    ce_tf1, ce_tf2 = sess.run([loss, nn_loss], {input: X, label: y})
# print("sf_pred_tf: ", tf_pred)
print("CE_tf1: ", ce_tf1)
print("CE_tf2: ", ce_tf2)


## PyTorch version ##
X_torch = torch.from_numpy(X.astype(np.float32))
y_torch = torch.from_numpy(np.argmax(y, axis=-1).astype(np.int64))
# loss1 = F.binary_cross_entropy_with_logits(X_torch, y_torch)   # combines sigmoid and nll_loss in a single function
# logit_A = F.softmax(X_torch, dim=1)
# loss2 = F.cross_entropy(logit_A, y_torch)
loss2 = F.cross_entropy(X_torch, y_torch)   # combines log_softmax and BCE_loss in a single function
print("CE_torch: ", loss2)

print("")