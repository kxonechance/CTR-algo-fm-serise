#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@file: nfm_model.py
@function:
@modify:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build Neural Factorization Machine(NFM) model with binary_classification_head."""

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

    with tf.variable_scope('linear_part'):
        # num_features * 1
        w = tf.get_variable('w', shape=[num_dense_features+num_sparse_features, 1], initializer=tf.initializers.glorot_normal())
        # batch * num_fields
        indices = tf.concat([dense_indices, sparse_indices], axis=1)
        # batch * num_fields * 1
        embeddings = tf.nn.embedding_lookup(w, indices)
        # batch * num_fields
        values = tf.concat([dense_values, sparse_values], axis=1)
        # batch * num_fields * 1
        linear_part = tf.multiply(embeddings, tf.reshape(values, shape=[-1, num_dense_fields+num_sparse_fields, 1]))
        # batch * 1
        linear_part = tf.reduce_sum(linear_part, axis=1)

    with tf.variable_scope('bi_inter_part'):
        # num_features * num_factors
        feat_emb = tf.get_variable('emb', shape=[num_dense_features+num_sparse_features, num_factors], initializer=tf.initializers.glorot_normal())
        # batch * num_fields
        indices = tf.concat([dense_indices, sparse_indices], axis=1)
        # batch * num_fields * num_factors
        embeddings = tf.nn.embedding_lookup(feat_emb, indices)
        # batch * num_fields
        values = tf.concat([dense_values, sparse_values], axis=1)
        # batch * num_fields * num_factors
        embeddings = tf.multiply(embeddings, tf.reshape(values, [-1, num_dense_fields+num_sparse_fields, 1]))
        # batch * num_factors
        sum_square = tf.square(tf.reduce_sum(embeddings, axis=1))
        # batch * num_factors
        square_sum = tf.reduce_sum(tf.square(embeddings), axis=1)
        # batch * num_factors
        bi_inter_part = 0.5 * tf.subtract(sum_square, square_sum)

    with tf.variable_scope('deep_part'):
        # dropout
        deep_part = tf.layers.dropout(bi_inter_part, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

        for i in range(len(deep_layers)):
            deep_part = tf.layers.dense(deep_part, deep_layers[i], activation=tf.nn.relu)
            deep_part = tf.layers.dropout(deep_part, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

        deep_part = tf.layers.dense(deep_part, 1, activation=None)
        # batch * 1
        deep_part = tf.reshape(deep_part, shape=[-1, 1])

    with tf.variable_scope('nfm_output'):
        bias = tf.get_variable('bias', shape=[1], initializer=tf.constant_initializer(0.0))
        # batch * 1
        bias = bias * tf.ones_like(deep_part, dtype=tf.float32)
        # batch * 1
        logits = linear_part + deep_part + bias

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