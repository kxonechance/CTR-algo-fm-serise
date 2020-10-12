#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
@author: kx
@file: afm_model.py
@function:
@modify:
@reference:
https://github.com/cnfsll/Attentional-Neural-Factorization-Machine/blob/master/AFM_Model.py
https://nirvanada.github.io/2017/09/18/AFM/
https://github.com/hexiangnan/attentional_factorization_machine/blob/master/code/AFM.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def model_fn(features, labels, mode, params):
    """Build Neural Factorization Machine(NFM) model with binary_classification_head."""

    num_dense_features = params['num_dense_features']
    num_sparse_features = params['num_sparse_features']
    num_dense_fields = params['num_dense_fields']
    num_sparse_fields = params['num_sparse_fields']

    num_features = num_dense_features + num_sparse_features
    num_fields = num_dense_fields + num_sparse_fields

    learning_rate = params.get('learning_rate', 0.001)

    att_factors = params.get('att_factors', 16)
    num_factors = params.get('num_factors', 4)

    dense_indices = tf.reshape(features['dense_indices'], shape=[-1, num_dense_fields])
    dense_values = tf.reshape(features['dense_values'], shape=[-1, num_dense_fields])

    sparse_indices = tf.reshape(features['sparse_indices'], shape=[-1, num_sparse_fields])
    sparse_values = tf.reshape(features['sparse_values'], shape=[-1, num_sparse_fields])

    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.reshape(labels, shape=[-1, 1])

    with tf.variable_scope('embedding_part'):
        # num_features * num_factors
        w = tf.get_variable('w', shape=[num_features, num_factors], initializer=tf.initializers.glorot_normal())
        # batch * num_fields
        indices = tf.concat([dense_indices, sparse_indices], axis=1)
        values = tf.concat([dense_values, sparse_values], axis=1)
        # batch * num_fields * num_factors
        embeddings = tf.nn.embedding_lookup(w, indices)

    with tf.variable_scope('pair-wise-interaction_part'):
        # interaction
        pair_wise_list = []
        num_inter_fields = int(num_fields * (num_fields - 1) / 2)
        for i in range(num_fields):
            for j in range(i+1, num_fields):
                # num_inter_fields * (batch * 1 * num_factors)
                pair_wise_list.append(tf.multiply(embeddings[:, i, :], embeddings[:, j, :]))
        # num_inter_fields * batch * num_factors
        pair_wise = tf.stack(pair_wise_list)
        # batch * num_inter_fields * num_factors
        pair_wise = tf.transpose(pair_wise, perm=[1, 0, 2])

    with tf.variable_scope('attention_part'):
        # num_factors * attention_factors
        w = tf.get_variable('w', shape=[num_factors, att_factors], initializer=tf.initializers.glorot_normal())
        # 1 * attention_factors
        b = tf.get_variable('b', shape=[1, att_factors], initializer=tf.initializers.glorot_normal())
        # attention_factors
        p = tf.get_variable('p', shape=[att_factors], initializer=tf.initializers.glorot_normal())
        # batch * num_inter_fields * att_factors
        attention_mul = tf.reshape(tf.matmul(tf.reshape(pair_wise, shape=[-1, num_factors]), w), shape=[-1, num_inter_fields, att_factors])
        # batch * num_inter_fields * 1
        attention_relu = tf.reduce_sum(tf.multiply(p, tf.nn.relu(attention_mul + b)), axis=2, keepdims=True)
        # batch * num_inter_fields * 1
        attention_out = tf.nn.softmax(attention_relu)
        # batch * num_inter_fields * 1
        attention_out = tf.layers.dropout(attention_out, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('pair_wise_attention_part'):
        # batch * num_factors
        attention_pair_wise = tf.reduce_sum(tf.multiply(pair_wise, attention_out), axis=1)
        # batch * num_factors
        attention_pair_wise = tf.layers.dropout(attention_pair_wise, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('afm_output_part'):
        # num_factors * 1
        w = tf.Variable(np.ones((num_factors, 1), dtype=np.float32))
        # batch * 1
        bi_linear = tf.matmul(attention_pair_wise, w)
        # 1 * 1
        bias = tf.get_variable('bias', shape=[1], initializer=tf.constant_initializer(0.0))
        # batch * 1
        bias = bias * tf.ones_like(bi_linear, dtype=tf.float32)
        # num_features * 1
        feat_w = tf.get_variable('feat_w', shape=[num_features, 1], initializer=tf.initializers.glorot_normal())
        # batch * num_fields
        indices = tf.concat([dense_indices, sparse_indices], axis=1)
        values = tf.concat([dense_values, sparse_values], axis=1)
        # batch * num_fields * 1
        emb = tf.nn.embedding_lookup(feat_w, indices)
        # batch * num_fields * 1
        feat_part = tf.multiply(emb, tf.reshape(values, shape=[-1, num_fields, 1]))
        # batch * 1
        feat_part = tf.reduce_sum(feat_part, axis=1)
        # batch * 1
        logits = bi_linear + feat_part + bias

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