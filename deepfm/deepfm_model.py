#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@file: deepfm_model.py
@function:
@modify:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build DeepFM model with binary_classification_head."""

    num_dense_features = params['num_dense_features']
    num_sparse_features = params['num_sparse_features']
    num_dense_fields = params['num_dense_fields']
    num_sparse_fields = params['num_sparse_fields']

    learning_rate = params.get('learning_rate', 0.001)

    deep_layers = params.get('deep_layers', [100, 100])
    num_factors = params.get('num_factors', 4)

    dense_indices = tf.reshape(features['dense_indices'], shape=[-1, num_dense_fields])
    dense_values = tf.reshape(features['dense_values'], shape=[-1, num_dense_fields])

    sparse_indices = tf.reshape(features['sparse_indices'], shape=[-1, num_sparse_fields])
    sparse_values = tf.reshape(features['sparse_values'], shape=[-1, num_sparse_fields])

    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.reshape(labels, shape=[-1, 1])

    with tf.variable_scope('first_order_part'):
        # num_features * 1
        w = tf.get_variable('w', shape=[num_dense_features+num_sparse_features, 1], initializer=tf.initializers.glorot_normal())
        # batch * num_fields
        indices = tf.concat([dense_indices, sparse_indices], axis=1)
        # batch * num_fields * 1
        first_order_part = tf.nn.embedding_lookup(w, indices)
        # batch * num_fields
        values = tf.concat([dense_values, sparse_values], axis=1)
        # batch * num_fields * 1
        first_order_part = tf.multiply(first_order_part, tf.reshape(values, shape=[-1, num_dense_fields+num_sparse_fields, 1]))
        # batch * num_fields
        first_order_part = tf.reduce_sum(first_order_part, axis=2)

    with tf.variable_scope('second_order_part'):
        # num_features * num_factors
        w = tf.get_variable('w', shape=[num_dense_features+num_sparse_features, num_factors], initializer=tf.initializers.glorot_normal())
        # batch * num_fields
        indices = tf.concat([dense_indices, sparse_indices], axis=1)
        # batch * num_fields * num_factors
        embeddings = tf.nn.embedding_lookup(w, indices)
        # batch * num_fields
        values = tf.concat([dense_values, sparse_values], axis=1)
        # batch * num_fields * num_factors
        embeddings = tf.multiply(embeddings, tf.reshape(values, shape=[-1, num_dense_fields+num_sparse_fields, 1]))
        # batch * num_factors
        sum_square = tf.square(tf.reduce_sum(embeddings, axis=1))
        # batch * num_factors
        square_sum = tf.reduce_sum(tf.square(embeddings), axis=1)
        # batch * num_factors
        second_order_part = 0.5 * (sum_square - square_sum)

    with tf.variable_scope('deep_part'):
        deep_part = tf.reshape(embeddings, shape=[-1, (num_dense_fields+num_sparse_fields)*num_factors])
        for i in range(len(deep_layers)):
            deep_part = tf.layers.dense(deep_part, deep_layers[i], activation=tf.nn.relu)
            deep_part = tf.layers.dropout(deep_part, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(tf.concat([first_order_part, second_order_part, deep_part], axis=1), 1, activation=None)

    my_head = tf.contrib.estimator.binary_classification_head()
    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
        logits=logits
    )



if __name__ == "__main__":
    pass