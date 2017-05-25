"""Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf

class tf_model(object):
  def __init__(self, words, Initial_We, rel, Initial_Rel, placeholder, FLAGS):
    self.lr = FLAGS.learning_rate
    self.reg = FLAGS.regularization
    self.eps = FLAGS.eps
    self.margin = FLAGS.margin
    self.alpha = FLAGS.alpha
    self.beta1 = FLAGS.beta1
    self.beta2 = FLAGS.beta2
    self.batchsize = FLAGS.batch_size
    self.peephole = FLAGS.peephole
    self.embed_dim = FLAGS.embed_dim
    self.hidden_dim = FLAGS.hidden_dim
    self.vocab_size = len(words)

    self.word_emb = tf.Variable(tf.random_uniform([self.vocab_size, FLAGS.embed_dim], FLAGS.lower_scale, FLAGS.higher_scale), trainable = True, name = 'word_emb')
    

    # self.We = tf.abs(tf.Variable(Initial_We, trainable = True, name = 'word_embed'))
    # self.Rel = tf.abs(tf.Variable(Initial_Rel, trainable = True, name = 'rel_embed'))
    self.We = tf.Variable(tf.random_uniform([self.vocab_size, FLAGS.embed_dim], FLAGS.lower_scale, FLAGS.higher_scale), trainable = True, name = 'word_embed')
    self.Rel = tf.Variable(tf.random_uniform([FLAGS.rel_size, FLAGS.embed_dim],  FLAGS.lower_scale, FLAGS.higher_scale), trainable = True, name = 'rel_embed')

    t1x = placeholder['t1_idx_placeholder']
    t1mask = placeholder['t1_msk_placeholder']
    t1length = placeholder['t1_length_placeholder']
    t2x = placeholder['t2_idx_placeholder']
    t2mask = placeholder['t2_msk_placeholder']
    t2length = placeholder['t2_length_placeholder'] 
    rel = placeholder['rel_placeholder']
    label = placeholder['label_placeholder']
    relmsk = placeholder['rel_msk_placeholder']

    rang=tf.range(0,FLAGS.batch_size,1)
    # nt1x = (rang+1)%FLAGS.batch_size
    # nt1mask, nt1length = 
    # nt2x = (rang+2)%FLAGS.batch_size
    # nt2mask, nt2length = 
    nlabels = tf.zeros_like(label)
    nrel = rel
    nrelmsk = relmsk
    # nt1x = placeholder['nt1_idx_placeholder']
    # nt1mask = placeholder['nt1_msk_placeholder']
    # nt1length = placeholder['nt1_length_placeholder'] 
    # nt2x = placeholder['nt2_idx_placeholder']
    # nt2mask = placeholder['nt2_msk_placeholder']
    # nt2length = placeholder['nt2_length_placeholder']
    # nrel = placeholder['nrel_placeholder']
    # nlabels = placeholder['nlabel_placeholder']
    # nrelmsk = placeholder['nrel_msk_placeholder'] #batch_size

    self.data_shard = placeholder['data_shard']
    self.row_indices = placeholder['row_indices']
    self.real_batch_size = placeholder['real_batch_size']

    self.data_shard_var = tf.Variable(tf.zeros([FLAGS.data_shard_rows,FLAGS.data_shard_cols], dtype = tf.int32),trainable=False)
    self.assign_data_shard_var = tf.assign(self.data_shard_var,self.data_shard)

    if FLAGS.tuple_model == 'ave':
        embed_t1 = self.tuple_embedding(t1x, t1mask, t1length, self.We)
        embed_t2 = self.tuple_embedding(t2x, t2mask, t2length, self.We)
        self.embed_nt1 = embed_nt1 = tf.nn.embedding_lookup(embed_t1,(rang+1)%FLAGS.batch_size)
        self.embed_nt2 = embed_nt2 = tf.nn.embedding_lookup(embed_t2,(rang+2)%FLAGS.batch_size)
        # embed_nt1 = self.neg_tuple_embedding(nt1x, nt1mask, nt1length, self.We)
        # embed_nt2 = self.neg_tuple_embedding(nt2x, nt2mask, nt2length, self.We)

    elif FLAGS.tuple_model == 'lstm':
        with tf.variable_scope('term_embed'):
            embed_t1 = self.tuple_lstm_embedding(t1x, t1mask, t1length, self.We) 
        with tf.variable_scope('term_embed', reuse = True): 
            embed_t2 = self.tuple_lstm_embedding(t2x, t2mask, t2length, self.We)
        with tf.variable_scope('term_embed', reuse = True): 
            # embed_nt1 = self.tuple_lstm_embedding(nt1x, nt1mask, nt1length, self.We)
            self.embed_nt1 = embed_nt1 = tf.nn.embedding_lookup(embed_t1,(rang+1)%FLAGS.batch_size)
        with tf.variable_scope('term_embed', reuse = True): 
            # embed_nt2 = self.tuple_lstm_embedding(nt2x, nt2mask, nt2length, self.We)
            self.embed_nt2 = embed_nt2 = tf.nn.embedding_lookup(embed_t2, (rang+2)%FLAGS.batch_size)
    else:
        print('Sorry, currently only support lstm terms and average terms')


    embed_rel = self.rel_embedding(self.Rel, rel, relmsk)
    self.embed_nrel = embed_nrel = self.rel_embedding(self.Rel, nrel, nrelmsk)

    #fix neg
    # pos_costs= tf.multiply(self.hierarchical_error(embed_t1, embed_rel, embed_t2, self.eps, self.batchsize, self.embed_dim), label) #batch_size
    # # neg_costs = tf.multiply(tf.maximum(tf.zeros([1], tf.float32), margin - self.hierarchical_error(embed_nt1, embed_nrel, embed_nt2, eps, self.batchsize, self.embed_dim)),(1-nlabels))#batch_size
    # google_neg_costs = tf.multiply(self.neg_hier_error(embed_nt1, embed_nrel, embed_nt2, self.eps, self.margin, self.batchsize, self.embed_dim), (1-nlabels))
    # neg_costs = google_neg_costs
    # neg_costs += pos_costs


    # google objective
    # pos_costs = tf.multiply(self.hierarchical_error(embed_t1, embed_rel, embed_t2, self.eps, self.batchsize, self.embed_dim), label)
    # neg_costs = tf.multiply(self.neg_hier_error(), (1-nlabels))
    # neg_costs +=pos_costs
    # pos_costs = tf.multiply(self.hierarchical_error(embed_t1, embed_rel, embed_t2, self.eps, self.batchsize, self.embed_dim), label)
    # neg_costs = tf.multiply(tf.maximum(tf.zeros([1], tf.float32), self.margin - self.hierarchical_error(embed_nt1, embed_nrel, embed_nt2, self.eps, self.batchsize, self.embed_dim)),(1-nlabels))
    # neg_costs +=pos_costs

    # #random neg
    # pos_costs= self.hierarchical_error(embed_t1, embed_rel, embed_t2, eps, batchsize, embed_dim) #batch_size
    # neg_costs = tf.maximum(tf.zeros([1], tf.float32), margin - self.hierarchical_error(embed_nt1, embed_nrel, embed_nt2, eps, batchsize, embed_dim)) #batch_size
    # pos_costs= tf.mul(self.hierarchical_error(embed_t1, embed_rel, embed_t2, eps, batchsize, embed_dim), label) #batch_size
    # neg_costs = tf.mul(tf.maximum(tf.zeros([1], tf.float32), margin - self.hierarchical_error(embed_nt1, embed_nrel, embed_nt2, eps, batchsize, embed_dim)), (1-nlabels)) #batch_size

    # self.costs = tf.reduce_sum(pos_costs, 0) + tf.reduce_sum(neg_costs, 0)


    neg_costs = tf.maximum(tf.zeros([1], tf.float32), self.margin + tf.multiply(self.hierarchical_error(embed_t1, embed_rel, embed_t2, self.eps, self.batchsize, self.embed_dim), label) - tf.multiply(self.hierarchical_error(embed_nt1, embed_nrel, embed_nt2, self.eps, self.batchsize, self.embed_dim), (1-nlabels))) 
    # wordnet
    # neg_costs = tf.maximum(tf.zeros([1], tf.float32), margin + tf.mul(self.hierarchical_error(embed_t1, embed_rel, embed_t2, eps, batch_size, embed_size), label) - tf.mul(self.hierarchical_error(embed_t1, embed_rel, embed_t2, eps, batch_size, embed_size), (1-label)))#batch_size

    neg_costs = neg_costs/tf.to_float(self.batchsize)
    # neg_costs = neg_costs
    l2_loss = FLAGS.regularization * (tf.nn.l2_loss(embed_t1) + tf.nn.l2_loss(embed_t2) + tf.nn.l2_loss(embed_rel))
    # tf.assign(embed_rel,tf.clip_by_norm(embed_rel, clip_norm=1, axes=0), name='renorm')
    neg_costs += l2_loss
    
    self.kb_loss = tf.reduce_sum(neg_costs, 0)
    # self.tvars= tvars = tf.trainable_variables()
    # print(self.tvars)
    # self.raw_kb_grads=raw_kb_grads = tf.gradients(self.kb_loss,self.tvars,colocate_gradients_with_ops=True)
    # print('grad',raw_kb_grads)
    # kb_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # self.kbc_train_op = kb_optimizer.apply_gradients(zip(raw_kb_grads, tvars))

    self.word2vec_loss = self.word2vec_loss_func(FLAGS, self.We, self.We, self.data_shard_var, self.row_indices, self.real_batch_size)
    # self.raw_word2vec_grads= raw_word2vec_grads= tf.gradients(self.word2vec_loss,self.tvars)
    # optimizer = tf.train.AdamOptimizer(FLAGS.cbow_learning_rate)
    # self.word2vec_train_op = optimizer.apply_gradients(zip(raw_word2vec_grads, self.tvars))


    self.costs = FLAGS.beta1*tf.reduce_sum(neg_costs, 0)+FLAGS.beta2 * self.word2vec_loss
 

  def word2vec_loss_func(self, hps, word_emb, classifier_emb, data_shard, row_indices, real_batch_size):
    self.data_shard=data_shard
    # word_emb=variables.word_emb
    # classifier_emb=variables.classifier_emb

    cur_tok_idx_seqs=tf.nn.embedding_lookup(self.data_shard,row_indices)

    word_emb_seqs=tf.nn.embedding_lookup(word_emb,cur_tok_idx_seqs)
    word_emb_seqs=tf.abs(word_emb_seqs)
    classifier_emb_seqs=tf.nn.embedding_lookup(classifier_emb,cur_tok_idx_seqs)
    classifier_emb_seqs=tf.abs(classifier_emb_seqs)
    
    filter=tf.tile(tf.reshape(
      tf.one_hot(hps.window_size,1+2*hps.window_size,on_value=0.0,off_value=1.0/(hps.window_size*2.0)),
      [1,1+2*hps.window_size,1,1]),
      [1,1,hps.embed_dim,1])

    cbows=tf.nn.depthwise_conv2d(tf.expand_dims(word_emb_seqs,1),filter,[1,1,1,1],padding='SAME')
    cbows=tf.reshape(cbows,[hps.word2vec_batch_size,hps.data_shard_cols,hps.embed_dim])
    
    neg_sample_indices=tf.range(0,hps.word2vec_batch_size)
    neg_sample_indices=tf.tile(tf.expand_dims(neg_sample_indices, 1),[1,hps.num_neg_samples])
    neg_sample_indices=neg_sample_indices+tf.tile(tf.expand_dims(tf.range(1,hps.num_neg_samples+1), 0), [hps.word2vec_batch_size, 1])
    neg_sample_indices=neg_sample_indices%hps.word2vec_batch_size

    neg_classifier_emb_seqs=tf.reshape(tf.nn.embedding_lookup(tf.reshape(classifier_emb_seqs,[hps.word2vec_batch_size,-1]),neg_sample_indices),
                                       [hps.word2vec_batch_size,hps.num_neg_samples,hps.data_shard_cols,hps.embed_dim])
    neg_classifier_emb_seqs=tf.abs(neg_classifier_emb_seqs)
    

    # l2 distance
    # classifier_emb_seq_sq_norms=tf.reduce_sum(classifier_emb_seqs*classifier_emb_seqs,2)
    # cbow_sq_norms=tf.reduce_sum(cbows*cbows,2)
    # neg_classifier_emb_seq_sq_norms=tf.reshape(tf.nn.embedding_lookup(classifier_emb_seq_sq_norms,neg_sample_indices),[hps.word2vec_batch_size,hps.num_neg_samples,hps.data_shard_cols])
    # pos_dots=tf.reduce_sum(cbows*classifier_emb_seqs,2)
    # pos_distances = cbow_sq_norms+classifier_emb_seq_sq_norms-2.0*pos_dots
    # neg_dots=tf.reduce_sum(tf.expand_dims(cbows,1)*neg_classifier_emb_seqs,3)
    # neg_distances = tf.expand_dims(cbow_sq_norms, 1)+neg_classifier_emb_seq_sq_norms-2.0*neg_dots
    # print('pos_distances', pos_distances.get_shape())
    # print('neg_distances', neg_distances.get_shape())
    
    # l1 distance
    # print('cbow',cbows.get_shape())
    # print('classfier',classifier_emb_seqs.get_shape())
    # print('cbow_neg', tf.expand_dims(cbows,1).get_shape())
    # print('neg_classifier_emb_seqs',neg_classifier_emb_seqs.get_shape())
    pos_distances = tf.reduce_sum(tf.abs(tf.subtract(cbows, classifier_emb_seqs)), 2)
    neg_distances = tf.reduce_sum(tf.abs(tf.subtract(tf.expand_dims(cbows,1), neg_classifier_emb_seqs)), 3)
    # print('pos_distances', pos_distances.get_shape())
    # print('neg_distances', neg_distances.get_shape())


    # just prob
    #distance_diffs=tf.expand_dims(pos_distances,1)-neg_distances
    #prob_pos_distance_greater_than_neg = tf.sigmoid(distance_diffs)

    # log prob 
    #distance_diffs=neg_distances-tf.expand_dims(pos_distances,1)
    #labels = tf.ones([hps.word2vec_batch_size, hps.num_neg_samples, hps.embed_dim], dtype = tf.float32)
    #prob_pos_distance_greater_than_neg = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=distance_diffs)

    # max(0, distance) same as order 
    distance_diffs=hps.margin+tf.expand_dims(pos_distances,1)-neg_distances
    prob_pos_distance_greater_than_neg = tf.maximum(tf.zeros([hps.word2vec_batch_size, hps.num_neg_samples, hps.data_shard_cols], tf.float32), distance_diffs)

    self.word2vec_loss = tf.reduce_sum(prob_pos_distance_greater_than_neg)/tf.to_float(real_batch_size*hps.num_neg_samples*hps.data_shard_cols)
    # print('word2vec_loss', self.word2vec_loss)
    return self.word2vec_loss

  # def hierarchical_error(self, embed_t1, embed_rel, embed_t2, eps, batch_size, embed_size):
  #   # specific = embed_t1 + embed_rel
  #   specific = embed_t1
  #   general = embed_t2
  #   error = tf.reduce_sum(tf.nn.softplus(general - specific), axis = 1)
  #   # error = tf.reduce_sum(tf.nn.softplus(specific - general), axis = 1)
  #   return error

  # def test_hierarchical_error(self, embed_t1, embed_rel, embed_t2, eps, batch_size, embed_size):
  #   # specific = embed_t1 + embed_rel
  #   specific = embed_t1
  #   general = embed_t2
  #   error = tf.reduce_sum(tf.pow(tf.maximum(tf.zeros_like(general, dtype = tf.float32), general-specific),2),1) #batch_size
  #   return error
 
  # def neg_hier_error(self, embed_t1, embed_rel, embed_t2, eps, margin, batch_size, embed_size):
  #   # specific = embed_t1 + embed_rel
  #   specific = embed_t1
  #   general = embed_t2
  #   dir1 = tf.nn.softplus(tf.reduce_min(general - specific, axis=1))
  #   dir2 = tf.nn.softplus(tf.reduce_min(specific - general, axis=1))
  #   error = dir1+dir2
  #   return error

  def loss(self):
    return self.costs

  def kbc_loss(self):
    return self.kb_loss

  def cbow_loss(self):
    return self.word2vec_loss

  # def neg_hier_error(self):
  #   # specific = embed_t1 + embed_rel
  #   specific = self.embed_nt1 
  #   general = self.embed_nt2
  #   # print(tf.reduce_min(general-specific + margin, axis=1).get_shape())
  #   dir1 = tf.pow(tf.maximum(tf.zeros_like(tf.reduce_min(general, axis=1), dtype = tf.float32), tf.reduce_min(general - specific + self.margin, axis=1)),2)
  #   dir2 = tf.pow(tf.maximum(tf.zeros_like(tf.reduce_min(general, axis=1), dtype = tf.float32), tf.reduce_min(specific - general + self.margin, axis=1)),2)

  #   # error = tf.reduce_sum(dir1+dir2, axis = 1) #batch_size
  #   error = dir1+dir2
  #   return error

  def neg_hier_error(self):
    return self.hierarchical_error(self.embed_nt1, self.embed_nrel, self.embed_nt2, self.eps, self.batchsize, self.embed_dim)

  def hierarchical_error(self, embed_t1, embed_rel, embed_t2, eps, batch_size, embed_size):
    #transE
    # embed_t1_rel = tf.reshape(embed_t1, (-1, 1, embed_size))
    # specific = tf.batch_matmul(embed_t1_rel, embed_rel)
    # specific = tf.reshape(specific, (-1, embed_size))
    # specific = embed_t1 + embed_rel
    specific = embed_t1
    general = embed_t2
    w1 = 1.0
    w2 = 10.0
    #L1 error
    # error = tf.reduce_sum(tf.abs(specific - general), 1)
    #L2 error
    # error = tf.sqrt(tf.reduce_sum(tf.pow((specific - general), 2), 1) + 1e-9)
    #  order embeddings(lab meeting result)
    # error = tf.reduce_sum(tf.pow(tf.maximum(tf.zeros_like(general, dtype = tf.float32), specific-general + eps), 2), 1)  #batch_size
    #adding regulazer 1
    # error = tf.reduce_sum(tf.pow(tf.maximum(tf.zeros_like(general, dtype = tf.float32), general-specific + eps), 2) + (self.alpha * tf.pow((specific-general+eps), 2)) - (self.alpha * tf.pow(embed_t1, 2)) + (self.alpha * tf.pow(embed_t2,2)), 1)
    #adding regulazer 2
    # error = tf.reduce_sum(tf.pow(tf.maximum(tf.zeros_like(general, dtype = tf.float32), specific-general + eps), 2) - (self.alpha * tf.pow((embed_t1+embed_t2), 2)), 1)
    # haw shuang error
    # error1 = w1 * tf.reduce_sum(tf.pow(tf.maximum(tf.zeros_like(general, dtype = tf.float32), general-specific + eps),2), 1) #batch_size
    # error2 = w2 * tf.reduce_sum(tf.pow(tf.maximum(tf.zeros_like(general, dtype = tf.float32), specific-general + eps),2), 1)
    # error = error1 + error2

    # for testing, change the model from specific - general to general - specific, (02.24.4:49pm). please mark when change it back
    # more experiment, change it back to specific-general
    # but something is wrong, seems like the right direction should be general specific.
    # so change specific- general as rev direction
    
    error = tf.reduce_sum(tf.pow(tf.maximum(tf.zeros_like(general, dtype = tf.float32), general-specific + eps),2), 1) #batch_size
    # error = tf.reduce_sum(tf.maximum(tf.zeros_like(general, dtype = tf.float32), general-specific + eps), 1) #batch_size

    return error


  def tuple_lstm_embedding(self, x, x_mask, x_length, We):
    print('Using LSTM to compose term vectors')
    embed = tf.nn.embedding_lookup(We, x) #batchsize * maxlength *  embed_size
    x_length = tf.reshape(x_length, [-1])
    # embed = embed * x_mask #x_mask: batchsize * maxlength * 1
    # term_rnn = tf.nn.rnn_cell.BasicRNNCell(self.embed_dim)
    term_rnn = tf.contrib.rnn.LSTMCell(self.hidden_dim, use_peepholes=self.peephole,num_proj=self.embed_dim, state_is_tuple=True)
    output, state = tf.nn.dynamic_rnn(term_rnn, embed, dtype=tf.float32, sequence_length = x_length)
    
    # select relevant vectors
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + x_length-1
    flat = tf.reshape(output, [-1, out_size])
    # not adding abs, don't know which works better
    relevant = tf.gather(flat, index)
    # relevant = tf.nn.softplus(relevant)
    relevant = tf.abs(relevant)
    return relevant

  def tuple_embedding(self, x, x_mask, x_length, We):
    print('Averaging vectors to get term vectos')
    embed = tf.nn.embedding_lookup(We, x) #batchsize * length *  embed_size
    x_mask = tf.cast(x_mask, tf.float32)
    embed = embed * x_mask #x_mask: batchsize * length * 1
    embed = tf.reduce_sum(embed, 1)
    x_length = tf.cast(x_length, tf.float32)
    embed = embed/x_length
    # embed = tf.nn.softplus(embed)
    embed = tf.abs(embed)
    return embed



  def rel_embedding(self, Rel, rel, relmsk):
    embed_rel = tf.nn.embedding_lookup(Rel, rel)
    # embed_rel = tf.abs(embed_rel * relmsk)
    embed_rel = embed_rel * relmsk
    return embed_rel

  def training(self, loss, epsilon, learning_rate):
    tf.summary.scalar(loss.op.name, loss)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

  def getWe(self):
    return self.We

  def getRel(self):
    return self.Rel

  def test(self, Rel, rel, relmsk):
    embed_rel = tf.nn.embedding_lookup(Rel, rel)
    # embed_rel = tf.abs(embed_rel * relmsk)
    embed_rel = embed_rel * relmsk
    return embed_rel
