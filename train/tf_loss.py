from utils import lovasz_losses_tf as L

import tensorflow as tf
import numpy as np


def weightedLoss(class_weights, onehot_labels, logits):
    '''
    cross_entropy loss function for imbalanced data by adding class weights

    :param class_weights: weights for the different classes in case of multi-class imbalance, shape: (n,) or (1, n)
    :param onehot_labels: labels in onehot format, shape: (?, n)  #n is the number of classes
    :param logits: logits output from the networks, shape: (?, n)
    :return:
    '''

    # your class weights/ class_weights = tf.constant([[1.0, 2.0, 3.0]])
    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

    # deduce weights for batch samples based on their true label
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)
    # compute your (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(onehot_labels, logits)
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses)

    return loss


def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def _get_cost(self, logits, cost_name, cost_kwargs):
    """
    Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
    Optional arguments are:
    class_weights: weights for the different classes in case of multi-class imbalance
    regularizer: power of the L2 regularizers added to the loss function
    """

    with tf.name_scope("cost"):
        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.y, [-1, self.n_class])
        if cost_name == "cross_entropy":
            class_weights = cost_kwargs.pop("class_weights", None)

            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)

                loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                      labels=flat_labels)
                weighted_loss = tf.multiply(loss_map, weight_map)

                loss = tf.reduce_mean(weighted_loss)

            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                 labels=flat_labels))
        elif cost_name == "dice_coefficient":
            eps = 1e-5
            prediction = pixel_wise_softmax(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss = -(2 * intersection / (union))

        elif cost_name == "lovasz_loss":
            loss = L.lovasz_hinge(logits, self.y)

        else:
            raise ValueError("Unknown cost function: " % cost_name)

        regularizer = cost_kwargs.pop("regularizer", None)
        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss += (regularizer * regularizers)

    return loss



