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

"""Trains and Evaluates the network."""
from __future__ import division
from __future__ import print_function
# from tensorflow.python.client import timeline
import time
import random
import tensorflow as tf
import plus_input_data as input_data
import numpy as np 
from plus_model import tf_model
import plus_eval_model 
import map_eval 
import plus_eval_model as eval_model
from collections import defaultdict
import pickle

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-2, 'Initial learning rate.')
flags.DEFINE_float('cbow_learning_rate',1e-3, 'cbow learning rate')
flags.DEFINE_float('regularization', 0.0, 'l2 regularization parameters')
flags.DEFINE_boolean('save', False, 'Save the model')
flags.DEFINE_boolean('update_embed', True, 'Update the embeddings')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 800, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', '../TransE_Order/data', 'Directory to put the data.')
flags.DEFINE_integer('cbow_step', 1000, 'Number of steps to run cbow trainer.')
flags.DEFINE_integer('embed_dim', 50, 'word embedding dimension')
flags.DEFINE_float('eps', 0., 'hierarchical error threshold')
flags.DEFINE_float('margin', 5, 'hinge loss margin')
flags.DEFINE_boolean('overfit', False, 'Over fit the dev data to check model')
flags.DEFINE_float('lower_scale', 0, 'lower initialize value for embeddings')
flags.DEFINE_float('higher_scale', 0.1, 'higher initialize value for embeddings')
flags.DEFINE_boolean('kb_only', True, 'whether to train kb only')
# flags.DEFINE_boolean('special_neg_sample', False, 'Whether to find negative examples from the not relation')
flags.DEFINE_integer('print_every',100,'Every 20 step, print out the evaluation results')
flags.DEFINE_float('alpha',0.01, 'regularization on error Function')
flags.DEFINE_boolean('rel_acc', True, 'check the different relation accurancy for test dataset')
flags.DEFINE_boolean('error_analysis', True, 'do error analysis for evaluation data')
flags.DEFINE_string('params_file', './params/','file to save parameters')
flags.DEFINE_string('error_file','./error_analysis/','dictionary to save error analysis result')
flags.DEFINE_string('ouput_file', './result/', 'print the result to this file')
flags.DEFINE_string('train', 'new_isa', 'training on both noisa and isa relatiosn')
flags.DEFINE_string('test','new_isa', 'test on isa relations')
flags.DEFINE_string('eval', 'acc', 'evaluate on MAP')
flags.DEFINE_integer('rel_size', 35, 'relation_size')
# lstm parameters
flags.DEFINE_integer('hidden_dim',100, 'lstm hidden layer dimension')
flags.DEFINE_boolean('peephole',True, 'whether to use peephole in lstm layer')
flags.DEFINE_string('tuple_model', 'ave', 'how to compose term vector, can choose from ave or lstm')

# word2vec parameters
flags.DEFINE_float('epsilon', 1e-6, 'epsilon for optimizor')
flags.DEFINE_float('beta1',1.0, 'weight on order_embedding loss')
flags.DEFINE_float('beta2',1.0, 'Weight on word2vec loss')
flags.DEFINE_string("word2vec_train_data", 'text8', "Training text file.")
flags.DEFINE_integer('word2vec_batch_size', 256, 'Batch size. Must divide evenly into the dataset sizes.') #256 #512
flags.DEFINE_integer('data_shard_rows', 256*600, 'num text "lines" for training in one shard') #256*600
flags.DEFINE_integer('data_shard_cols', 100, 'num tokens per text line') #100
flags.DEFINE_integer('vocab_size', 80000, 'vocab_size')
flags.DEFINE_float('num_neg_samples', 30, 'num_neg_samples')
flags.DEFINE_integer("window_size", 5, "The number of words to predict to the left and right ")

# nearest neighbor parameters
flags.DEFINE_integer('knn', 10, 'how many neighbors want to check')
# big wiki
# flags.DEFINE_string("word2vec_train_data", '../acl_cbow/data/binary-wackypedia-1-4-ukwac-', "Training text file.")
# flags.DEFINE_integer('word2vec_batch_size', 512, 'Batch size. Must divide evenly into the dataset sizes.') #256 #512
# flags.DEFINE_integer('data_shard_rows', 512*600, 'num text "lines" for training in one shard') #256*600
# flags.DEFINE_integer('data_shard_cols', 200, 'num tokens per text line') #100

def placeholder_inputs(batch_size):
  placeholder = {}
  #positive example term1
  placeholder['t1_idx_placeholder'] = tf.placeholder(tf.int32, shape=(None, None))
  placeholder['t1_msk_placeholder'] = tf.placeholder(tf.int32, shape=(None, None, 1))
  placeholder['t1_length_placeholder'] = tf.placeholder(tf.int32, shape=(None, 1))
  # positive example term2
  placeholder['t2_idx_placeholder'] = tf.placeholder(tf.int32, shape=(None, None))
  placeholder['t2_msk_placeholder'] = tf.placeholder(tf.int32, shape=(None, None, 1))
  placeholder['t2_length_placeholder'] = tf.placeholder(tf.int32, shape=(None, 1))
  #negative example term1
  # placeholder['nt1_idx_placeholder'] = tf.placeholder(tf.int32, shape=(None, None))
  # placeholder['nt1_msk_placeholder'] = tf.placeholder(tf.int32, shape=(None, None, 1))
  # placeholder['nt1_length_placeholder'] = tf.placeholder(tf.int32, shape=(None, 1))
  #negative exmaple term2
  # placeholder['nt2_idx_placeholder'] = tf.placeholder(tf.int32, shape=(None, None))
  # placeholder['nt2_msk_placeholder'] = tf.placeholder(tf.int32, shape=(None, None, 1))
  # placeholder['nt2_length_placeholder'] = tf.placeholder(tf.int32, shape=(None, 1))
  #positive relation
  placeholder['rel_placeholder'] = tf.placeholder(tf.int32, shape=[None])
  placeholder['rel_msk_placeholder'] = tf.placeholder(tf.float32, shape=[None, 1])
  #negative relation
  # placeholder['nrel_placeholder'] = tf.placeholder(tf.int32, shape=[None])
  # placeholder['nrel_msk_placeholder'] = tf.placeholder(tf.float32, shape=[None, 1])
  #positive label
  placeholder['label_placeholder'] = tf.placeholder(tf.float32, shape=[None])
  #negative label
  # placeholder['nlabel_placeholder'] = tf.placeholder(tf.float32, shape=[None])
  #word2vec input
  placeholder['row_indices'] = tf.placeholder(tf.int32, shape = [FLAGS.word2vec_batch_size])
  placeholder['real_batch_size'] = tf.placeholder(tf.int32, shape = [])
  placeholder['data_shard'] = tf.placeholder(tf.int32, shape=[FLAGS.data_shard_rows, FLAGS.data_shard_cols])

  return   placeholder

def fill_feed_dict(data_set, placeholder, row_indices, rel):

  r_idx, t1_idx, t2_idx, labels = data_set.next_batch(FLAGS.batch_size)
  t1x, t1mask, t1length= input_data.prepare_data(t1_idx)
  t2x, t2mask, t2length = input_data.prepare_data(t2_idx)
  # print('r_idx', r_idx.shape)
  relmsk = input_data.rel_msk(r_idx, rel)
  
  #random find negative examples from the same batch
  # nr_idx, nt1_idx, nt2_idx, nlabels = input_data.find_neg(r_idx, t1_idx, t2_idx, labels)
  # nt1x, nt1mask, nt1length= input_data.prepare_data(nt1_idx)
  # nt2x, nt2mask, nt2length = input_data.prepare_data(nt2_idx)
  # nrelmsk = input_data.rel_msk(nr_idx, rel)

  feed_dict = {
      placeholder['t1_idx_placeholder']: t1x,
      placeholder['t1_msk_placeholder']: t1mask, 
      placeholder['t1_length_placeholder']: t1length,
      placeholder['t2_idx_placeholder']: t2x,
      placeholder['t2_msk_placeholder']: t2mask,
      placeholder['t2_length_placeholder']: t2length,
      # placeholder['nt1_idx_placeholder']: nt1x,
      # placeholder['nt1_msk_placeholder']: nt1mask,
      # placeholder['nt1_length_placeholder']: nt1length,
      # placeholder['nt2_idx_placeholder']: nt2x,
      # placeholder['nt2_msk_placeholder']: nt2mask,
      # placeholder['nt2_length_placeholder']: nt2length,
      placeholder['rel_placeholder']: r_idx,
      # placeholder['nrel_placeholder']: nr_idx,
      placeholder['label_placeholder']: labels,
      # placeholder['nlabel_placeholder']: nlabels,
      placeholder['rel_msk_placeholder']: relmsk,
      # placeholder['nrel_msk_placeholder']: nrelmsk,
      placeholder['row_indices']: row_indices,
      placeholder['real_batch_size']: len(row_indices),
  }

  return feed_dict

def partial_word2vec_fill_feed_dict(placeholder, data_shared):
  feed_dict = {
         placeholder['data_shard']: data_shared,
         }
  return feed_dict



def run_training():
  accu_list = []
  train_accu_list = []
  test_accu_list = []
  curr_best = 0
  outfile = open(FLAGS.ouput_file+'newtask_abs_l1_learning_rate_'+str(FLAGS.learning_rate)+'_word2vec_batch'+str(FLAGS.word2vec_batch_size)+'_word2vec'+'_kb_only'+str(FLAGS.kb_only)+'_cbow_step'+str(FLAGS.cbow_step)+'_batchsize'+str(FLAGS.batch_size)+'_eps'+str(FLAGS.eps)+'_tuplemodel'+str(FLAGS.tuple_model)+'_peephole'+str(FLAGS.peephole)+'_hiddendim'+str(FLAGS.hidden_dim)+'_train'+str(FLAGS.train)+'_test'+str(FLAGS.test)+'_eval'+str(FLAGS.eval)+'_margin'+str(FLAGS.margin)+'_reg'+str(FLAGS.regularization)+'_dim'+str(FLAGS.embed_dim)+'_steps'+str(FLAGS.max_steps)+'.txt', 'wt')

  error_file_name = FLAGS.error_file+'newtask_abs_l1_learning_rate'+str(FLAGS.learning_rate)+'_word2vec_batch'+str(FLAGS.word2vec_batch_size)+'_word2vec'+'_kb_only'+str(FLAGS.kb_only)+'_cbow_step'+str(FLAGS.cbow_step)+'_batchsize'+str(FLAGS.batch_size)+'_eps'+str(FLAGS.eps)+'_tuplemodel'+str(FLAGS.tuple_model)+'_peephole'+str(FLAGS.peephole)+'_hiddendim'+str(FLAGS.hidden_dim)+'_eval'+str(FLAGS.eval)+'_train'+str(FLAGS.train)+'_test'+str(FLAGS.test)+'_margin'+str(FLAGS.margin)+'_reg'+str(FLAGS.regularization)+'_dim'+str(FLAGS.embed_dim)+'_steps'+str(FLAGS.max_steps)

  fname = FLAGS.params_file+'newtask_abs_l1_learning_rate'+str(FLAGS.learning_rate)+'_word2vec_batch'+str(FLAGS.word2vec_batch_size)+'_word2vec'+'_kb_only'+str(FLAGS.kb_only)+'_cbow_step'+str(FLAGS.cbow_step)+'_batchsize'+str(FLAGS.batch_size)+'_eps'+str(FLAGS.eps)+'_tuplemodel'+str(FLAGS.tuple_model)+'_peephole'+str(FLAGS.peephole)+'_hiddendim'+str(FLAGS.hidden_dim)+'_eval'+str(FLAGS.eval)+'_train'+str(FLAGS.train)+'_test'+str(FLAGS.test)+'_margin'+str(FLAGS.margin)+'_reg'+str(FLAGS.regularization)+'_dim'+str(FLAGS.embed_dim)+'_steps'+str(FLAGS.max_steps)+'.pkl'
  data_sets = input_data.read_data_sets(FLAGS, outfile)
  # special_neg_sample = FLAGS.special_neg_sample
  if FLAGS.overfit:
    train_data = data_sets.dev
  else:
    train_data = data_sets.train

    

  with tf.Graph().as_default():
    placeholder = placeholder_inputs(FLAGS.batch_size)
    print('Build Model...', file = outfile)
    model = tf_model(data_sets.words, data_sets.We, data_sets.rel, data_sets.Rel, placeholder, FLAGS)
    print('Build Loss Function...', file = outfile)
    # loss = model.loss()
    kb_loss = model.kbc_loss()
    cbow_loss = model.cbow_loss()
    print('Build Encode Function...', file = outfile)
    if FLAGS.tuple_model == 'ave':
      embed_t1 = model.tuple_embedding(placeholder['t1_idx_placeholder'], placeholder['t1_msk_placeholder'], placeholder['t1_length_placeholder'], model.getWe())
      embed_t2 = model.tuple_embedding(placeholder['t2_idx_placeholder'], placeholder['t2_msk_placeholder'], placeholder['t2_length_placeholder'], model.getWe())
      # embed_nt1 = model.tuple_embedding(placeholder['nt1_idx_placeholder'], placeholder['nt1_msk_placeholder'], placeholder['nt1_length_placeholder'], model.getWe())
      # embed_nt2 = model.tuple_embedding(placeholder['nt2_idx_placeholder'], placeholder['nt2_msk_placeholder'], placeholder['nt2_length_placeholder'], model.getWe())

    elif FLAGS.tuple_model == 'lstm':
      with tf.variable_scope('term_embed', reuse = True):
        embed_t1 = model.tuple_lstm_embedding(placeholder['t1_idx_placeholder'], placeholder['t1_msk_placeholder'], placeholder['t1_length_placeholder'], model.getWe())
      with tf.variable_scope('term_embed', reuse = True): 
        embed_t2 = model.tuple_lstm_embedding(placeholder['t2_idx_placeholder'], placeholder['t2_msk_placeholder'], placeholder['t2_length_placeholder'], model.getWe())
      # with tf.variable_scope('term_embed', reuse = True): 
        # embed_nt1 = model.tuple_lstm_embedding(placeholder['nt1_idx_placeholder'], placeholder['nt1_msk_placeholder'], placeholder['nt1_length_placeholder'], model.getWe())
      # with tf.variable_scope('term_embed', reuse = True): 
        # embed_nt2 = model.tuple_lstm_embedding(placeholder['nt2_idx_placeholder'], placeholder['nt2_msk_placeholder'], placeholder['nt2_length_placeholder'], model.getWe())
    else:
      print('Sorry, currently only support lstm terms and average terms')

    embed_r = model.rel_embedding(model.getRel(), placeholder['rel_placeholder'], placeholder['rel_msk_placeholder'])
    # embed_nr = model.rel_embedding(model.getRel(), placeholder['nrel_placeholder'], placeholder['nrel_msk_placeholder'])
   
    print('Build Hierarchical Error Function...', file = outfile)
    h_error = model.hierarchical_error(embed_t1, embed_r, embed_t2, FLAGS.eps, FLAGS.batch_size, FLAGS.embed_dim)
    nh_error = model.neg_hier_error()
    # nh_error = model.neg_hier_error(embed_nt1, embed_nr, embed_nt2, FLAGS.eps, FLAGS.margin, FLAGS.batch_size, FLAGS.embed_dim)
    # test_h_error = model.test_hierarchical_error(embed_t1, embed_r, embed_t2, FLAGS.eps, FLAGS.batch_size, FLAGS.embed_dim)
    print('Build Training Function...', file = outfile)
    # train_op = model.training(loss, FLAGS.learning_rate)
    kb_train_op = model.training(kb_loss, FLAGS.epsilon, FLAGS.learning_rate)
    cbow_train_op = model.training(cbow_loss, FLAGS.epsilon, FLAGS.cbow_learning_rate)
  
    data_shared = np.asarray(data_sets.word2vec_data[:FLAGS.data_shard_rows*FLAGS.data_shard_cols]).reshape((FLAGS.data_shard_rows, FLAGS.data_shard_cols))
    

    model_we = model.getWe()
    model_rel = model.getRel()  
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    partial_feed_dict = partial_word2vec_fill_feed_dict(placeholder, data_shared)
    sess.run(model.assign_data_shard_var, feed_dict = partial_feed_dict)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)

    # profile 
    # run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)  
    # run_metadata = tf.RunMetadata()
    # profile
    perm = np.arange(FLAGS.data_shard_rows)
    word2vec_idx = 0
    for step in range(FLAGS.max_steps):
      start_time = time.time()
      if (word2vec_idx + FLAGS.word2vec_batch_size) > FLAGS.data_shard_rows:
        random.shuffle(perm)
        word2vec_idx = 0
        # row_indices = perm[:FLAGS.word2vec_batch_size]
      row_indices = perm[word2vec_idx:word2vec_idx+FLAGS.word2vec_batch_size]
      word2vec_idx += FLAGS.word2vec_batch_size
      feed_dict = fill_feed_dict(train_data, placeholder, row_indices, data_sets.rel)
      # print('feed_dict', time.time()-start_time)
      if(FLAGS.kb_only):
        t1 = time.time()
        # _, kb_loss_value = sess.run([kb_train_op, kb_loss], feed_dict=feed_dict, options=run_opts,run_metadata=run_metadata)
        _, kb_loss_value = sess.run([kb_train_op, kb_loss], feed_dict=feed_dict)
        # print('kb_train_op, ',time.time()-t1)
      elif(step<FLAGS.cbow_step):
        t1 = time.time()
        # _, cbow_loss_value = sess.run([cbow_train_op,cbow_loss], feed_dict=feed_dict, options=run_opts,run_metadata=run_metadata)
        _, cbow_loss_value = sess.run([cbow_train_op,cbow_loss], feed_dict=feed_dict)
        # _, kb_loss_value = sess.run([kb_train_op, kb_loss], feed_dict=feed_dict)
        # print('cbow_train_op, ',time.time()-t1)
      else:
        t1 = time.time()
          # _,loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        # _, kb_loss_value = sess.run([kb_train_op, kb_loss], feed_dict=feed_dict, options=run_opts,run_metadata=run_metadata)
        # _, cbow_loss_value = sess.run([cbow_train_op,cbow_loss], feed_dict=feed_dict, options=run_opts,run_metadata=run_metadata)
        _, kb_loss_value = sess.run([kb_train_op, kb_loss], feed_dict=feed_dict)
        # print('kb_train_op', time.time()-t1)
        t2 = time.time()
        _, cbow_loss_value = sess.run([cbow_train_op,cbow_loss], feed_dict=feed_dict)
        # print('cbow_train_op, ', time.time()-t2)

      he_error = sess.run(h_error, feed_dict=feed_dict)

      duration = time.time() - start_time
      # if (train_data.index_in_epoch + FLAGS.batch_size) > train_data.num_examples:
      # if (FLAGS.save):
        # saver.save(sess, FLAGS.train_dir, global_step=step)
      if (step%(FLAGS.print_every) == 0):
        embed = sess.run(model_we,feed_dict=feed_dict)
        print(step, file = outfile)
        print('*'*80, file = outfile)
        if(FLAGS.kb_only):
          print('Epoch %d: kb_loss = %.2f (%.3f sec)' % (train_data.epochs_completed, kb_loss_value, duration), file = outfile)
        elif(step<FLAGS.cbow_step): 
          print('Epoch %d: cbow_loss = %.2f (%.3f sec)' % (train_data.epochs_completed, cbow_loss_value, duration), file = outfile)
          print('Epoch %d: cbow_loss = %.2f (%.3f sec)' % (train_data.epochs_completed, cbow_loss_value, duration))
          # print('Epoch %d: kb_loss = %.2f (%.3f sec)' % (train_data.epochs_completed, kb_loss_value, duration), file = outfile)
        else:
          print('Epoch %d: kb_loss = %.2f (%.3f sec)' % (train_data.epochs_completed, kb_loss_value, duration), file = outfile)
          print('Epoch %d: cbow_loss = %.2f (%.3f sec)' % (train_data.epochs_completed, cbow_loss_value, duration), file = outfile)
      
        if FLAGS.eval == 'map':
                print('MAP Evaluation......', file = outfile)
                train_map = map_eval.do_train_eval(sess, h_error, placeholder, data_sets.train_test, train_data.epochs_completed, data_sets.train_neg, FLAGS, outfile)
                train_accu_list.append(train_map)
                print('Training MAP:%.5f' %train_map, file = outfile)
                dev_map = map_eval.do_train_eval(sess, h_error, placeholder, data_sets.dev, train_data.epochs_completed, data_sets.train_neg, FLAGS, outfile)
                print('Dev MAP:%.5f' %dev_map, file = outfile)
                accuracy = map_eval.do_train_eval(sess, h_error, placeholder, data_sets.devtest, train_data.epochs_completed, data_sets.train_neg, FLAGS, outfile)
                accu_list.append(accuracy)
                print('Devtest MAP:%.5f' %accuracy, file = outfile)
                print('', file = outfile)
        

        if FLAGS.eval == 'acc':
                train_acc = eval_model.do_train_eval(sess, h_error, nh_error, placeholder, train_data, train_data.epochs_completed, data_sets.train_neg, curr_best, FLAGS, error_file_name, outfile, data_sets.rel, data_sets.words)
                train_accu_list.append(train_acc)

                dev2_acc, test_acc, wrong_indices, wrong_preds = eval_model.do_eval(sess, h_error, placeholder, data_sets.dev, data_sets.devtest, data_sets.test, train_data.epochs_completed,curr_best, FLAGS, error_file_name, outfile, data_sets.rel, data_sets.words)
                accu_list.append(dev2_acc)
                test_accu_list.append(test_acc)

                eval_model.knn(data_sets.nn_data, data_sets.words, embed, FLAGS.knn, outfile)
                # print("Accuracy for Devtest: %.5f" % dev2_acc)
                # print("Accuracy for Test: %.5f" %test_acc)
                # print ('')
                print("Accuracy for Devtest: %.5f" % dev2_acc, file = outfile)
                print("Accuracy for Test: %.5f" %test_acc, file = outfile)
                print ('', file = outfile)
        if FLAGS.save and dev2_acc > curr_best:
                print('saving model')
                f = open(fname,'wb')
                save_model = {}
                save_model['embeddings'] = sess.run(model_we, feed_dict=feed_dict)
                save_model['rel'] = sess.run(model_rel, feed_dict = feed_dict)
                pickle.dump(save_model, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()
    # tl = timeline.Timeline(run_metadata.step_stats)
    # ctf = tl.generate_chrome_trace_format(show_memory=True)
    # # ctf = tl.generate_chrome_trace_format()
    # with open('timeline_cache.json', 'w') as f:
    #   f.write(ctf)

    print('Average of Top 10 Training Score', np.mean(sorted(train_accu_list, reverse = True)[:10]), file = outfile)
    opt_idx = np.argmax(np.asarray(accu_list))
    print('Epoch', opt_idx, file = outfile)
    print('Best Dev2 Score: %.5f' %accu_list[opt_idx], file = outfile)
    print('Best Test Score: %.5f' %test_accu_list[opt_idx], file = outfile)
def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
