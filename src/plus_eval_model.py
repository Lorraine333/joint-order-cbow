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

"""Evaluates the network."""
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import plus_input_data
import numpy as np
from plus_model import tf_model
from collections import defaultdict
from scipy.spatial import distance

def fill_eval_feed_dict(data_set, placeholder, FLAGS, rel):
  r_idx, t1_idx, t2_idx, labels = data_set.eval_batch()

  t1x, t1mask, t1length= plus_input_data.prepare_data(t1_idx)
  t2x, t2mask, t2length = plus_input_data.prepare_data(t2_idx)
  relmsk = plus_input_data.rel_msk(r_idx, rel)

  # define identity matrix
  # t = np.zeros((r_idx.shape[0],FLAGS.embed_dim,FLAGS.embed_dim))
  # t = np.asarray([1.0 if i == j else 0.0 for k in range(t.shape[0]) for i in range(t.shape[1]) for j in range(t.shape[2])], np.float32)
  # t = t.reshape(r_idx.shape[0],FLAGS.embed_dim,FLAGS.embed_dim)

  feed_dict = {
      placeholder['t1_idx_placeholder']: t1x,
      placeholder['t1_msk_placeholder']: t1mask, 
      placeholder['t1_length_placeholder']: t1length,
      placeholder['t2_idx_placeholder']: t2x,
      placeholder['t2_msk_placeholder']: t2mask,
      placeholder['t2_length_placeholder']: t2length,
      placeholder['rel_placeholder']: r_idx,
      placeholder['label_placeholder']: labels,
      placeholder['rel_msk_placeholder']: relmsk,
  }

  return feed_dict



def fill_feed_dict(data_set, placeholder, FLAGS, rel):

  r_idx, t1_idx, t2_idx, labels = data_set.eval_batch()
  t1x, t1mask, t1length= plus_input_data.prepare_data(t1_idx)
  t2x, t2mask, t2length = plus_input_data.prepare_data(t2_idx)
  # print('r_idx', r_idx.shape)
  relmsk = plus_input_data.rel_msk(r_idx, rel)
  
  #random find negative examples from the same batch
  # nr_idx, nt1_idx, nt2_idx, nlabels = plus_input_data.find_neg(r_idx, t1_idx, t2_idx, labels)
  # nt1x, nt1mask, nt1length= plus_input_data.prepare_data(nt1_idx)
  # nt2x, nt2mask, nt2length = plus_input_data.prepare_data(nt2_idx)
  # nrelmsk = plus_input_data.rel_msk(nr_idx, rel)


  # define identity matrix
  #t = np.zeros((r_idx.shape[0],FLAGS.embed_dim,FLAGS.embed_dim))
  #t = np.asarray([1.0 if i == j else 0.0 for k in range(t.shape[0]) for i in range(t.shape[1]) for j in range(t.shape[2])], np.float32)
  #t = t.reshape(r_idx.shape[0],FLAGS.embed_dim,FLAGS.embed_dim)
  # iden = tf.Variable(t)

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
  }

  return feed_dict

def best_threshold(errs, target, outfile):
  indices = np.argsort(errs)
  sortedErrors = errs[indices]
  sortedTarget = target[indices]
  tp = np.cumsum(sortedTarget)
  invSortedTarget = (sortedTarget == 0).astype('float32')
  Nneg = invSortedTarget.sum()
  fp = np.cumsum(invSortedTarget)
  tn = fp * -1 + Nneg
  accuracies = (tp + tn) / sortedTarget.shape[0]
  i = accuracies.argmax()
  # print('errors', sortedErrors[:])
  # print('target', invSortedTarget[:])
  print("Accuracy for Dev:", accuracies[i], file = outfile)
  # calculate recall precision and F1
  Npos = sortedTarget.sum()
  fn = tp * -1 + Npos
  # print('tp',tp)
  # print('fp',fp)
  precision = tp/(tp + fp)
  recall = tp/(tp + fn)
  # print(precision[i])
  # print(recall[i])
  # print(tp[i])
  # print(fp[i])
  # print(tp[i]+tn[i])
  f1 = (2*precision[i]*recall[i])/(precision[i]+recall[i])
  # print("Precision, Recall and F1 are %.5f %.5f %.5f" % (precision[i], recall[i], f1), file = outfile)
  print("Precision, Recall and F1 are %.5f %.5f %.5f" % (precision[i], recall[i], f1))

  # print("Number of positives, negatives, tp, tn: %f %f %f %f" % (target.sum(), Nneg, tp[i], tn[i]))
  return sortedErrors[i], accuracies[i]


def wordnet_train_eval(sess,h_error, placeholder,data_set, num, FLAGS, rel):

  feed_dict = fill_eval_feed_dict(data_set, placeholder, FLAGS, rel)
  true_label = feed_dict[placeholder['label_placeholder']]
  he_error = sess.run(h_error, feed_dict = feed_dict)
  _, acc = best_threshold(he_error, true_label)

  return acc


def do_eval(sess,h_error,placeholder,data_set, devtest,test, num, curr_best, FLAGS,error_file_name,outfile, rel, words):

  feed_dict = fill_eval_feed_dict(data_set, placeholder, FLAGS, rel)
  true_label = feed_dict[placeholder['label_placeholder']]
  he_error = sess.run(h_error, feed_dict = feed_dict)
  thresh, _ = best_threshold(he_error, true_label, outfile)

  #evaluat devtest
  feed_dict_devtest = fill_eval_feed_dict(devtest, placeholder, FLAGS, rel)
  true_label_devtest = feed_dict_devtest[placeholder['label_placeholder']]
  devtest_he_error = sess.run(h_error, feed_dict = feed_dict_devtest)
  pred = devtest_he_error <= thresh
  correct = (pred == true_label_devtest)
  accuracy = float(correct.astype('float32').mean())
  wrong_indices = np.logical_not(correct).nonzero()[0]
  wrong_preds = pred[wrong_indices]
   #evaluate test
  feed_dict_test = fill_eval_feed_dict(test, placeholder, FLAGS, rel)
  true_label_test = feed_dict_test[placeholder['label_placeholder']]
  test_he_error = sess.run(h_error, feed_dict = feed_dict_test)
  test_pred = test_he_error <= thresh
  test_correct = (test_pred == true_label_test)
  test_accuracy = float(test_correct.astype('float32').mean())
  test_wrong_indices = np.logical_not(test_correct).nonzero()[0]
  test_wrong_preds = test_pred[test_wrong_indices]

  if accuracy>curr_best:
  # #evaluat devtest
    error_file = open(error_file_name+"_test.txt",'wt')
    if FLAGS.rel_acc:
      rel_acc_checker(feed_dict_devtest, placeholder, correct, data_set, error_file, rel)

    if FLAGS.error_analysis:
      err_analysis(data_set, wrong_indices, feed_dict_devtest, placeholder, error_file, rel, words)
  return accuracy,test_accuracy, wrong_indices, wrong_preds


def do_train_eval(sess,h_error,nh_error, placeholder,data_set, num, neg_data, curr_best, FLAGS, error_file_name, outfile, rel, words):

  feed_dict = fill_feed_dict(data_set, placeholder, FLAGS, rel)
  # concatenate the true and false labels
  true_label = feed_dict[placeholder['label_placeholder']]
  false_label = np.zeros(true_label.shape)
  labels = np.concatenate((true_label, false_label), axis = 0)
  # print("type of true labels",type(true_label))

  he_error = sess.run(h_error, feed_dict = feed_dict)
  nhe_error = sess.run(nh_error, feed_dict = feed_dict)
  errors = np.concatenate((he_error, nhe_error), axis = 0)
  # print("type of errors",type(he_error))
  thresh, acc = best_threshold(errors, labels, outfile)


  if acc > curr_best:
    error_file = open(error_file_name+"_train.txt",'wt')
    pred = he_error <= thresh
    correct = (pred == true_label)
    accuracy = float(correct.astype('float32').mean())
    wrong_indices = np.logical_not(correct).nonzero()[0]
    wrong_preds = pred[wrong_indices]

    if FLAGS.rel_acc:
      rel_acc_checker(feed_dict, placeholder, correct, data_set, error_file, rel)

    if FLAGS.error_analysis:
      err_analysis(data_set, wrong_indices, feed_dict, placeholder, error_file, rel, words)

  return acc

def dist(v1, v2):
  # print(v1)
  # print(v2)
  return distance.euclidean(tuple(v1),tuple(v2))

def knn(nn_list, words, We, k, outfile):
  idx2word = {words[w]: w for w in words}
  # embed = sess.run(We, feed_dict = feed_dict)
  for w in nn_list:
    idx = words[w]
    temp = []
    for embed in We:
      temp.append(dist(We[idx], embed))
    top_idx = np.argpartition(np.asarray(temp),k)[:k]
    print('*'*50, file = outfile)
    print('target word:', w, file = outfile)
    # print(top_idx)
    for t in top_idx:
      print(idx2word[t], file = outfile)



def err_analysis(data_set, wrong_indices, feed_dict, placeholder, error_file, rel, words):
  temp,temp1, temp2 = {}, {}, {}
  for w in words:
    temp[words[w]] = w
  for w1 in rel:
    temp1[rel[w1]] = w1

  # print(wrong_indices)
  # outputfile = open('result/train_test'+str(num)+'.txt','wt') 
  for i in wrong_indices:
    wrong_t1 = feed_dict[placeholder['t1_idx_placeholder']][i]
    wrong_t2 = feed_dict[placeholder['t2_idx_placeholder']][i]
    wrong_rel = feed_dict[placeholder['rel_placeholder']][i]
    wrong_lab = feed_dict[placeholder['label_placeholder']][i]


    for t in wrong_t1:
      if "</s>" not in temp[t]:
        print(temp[t]+"|",end='', file = error_file),
        # print("\t"),
        # outputfile.write(temp[t]+"_")
        # outputfile.write("\t")
    for t2 in wrong_t2:
      if "</s>" not in temp[t2]:
        print(temp[t2]+"|",end='', file = error_file)
        # print("\t"),
        # outputfile.write(temp[t2]+"_")
        # outputfile.write("\t")
    print(temp1[wrong_rel]+'\t',end='', file = error_file)
    print(str(wrong_lab), file = error_file)
  #check different relation wrong numbers
    if wrong_rel in temp2:
      temp2[wrong_rel] += 1
    else:
      temp2[wrong_rel] = 1
    # outputfile.write(temp1[wrong_rel]+"\t")
    # outputfile.write(str(wrong_lab)+"\n")
  print('relation analysis', file = error_file)
  for key in temp2:
    print(str(temp1[key]) + ":" +str(temp2[key]), file = error_file)
    # outputfile.write(str(temp1[key]) + ":" +str(temp2[key])+"\n")



def rel_acc_checker(feed_dict_devtest, placeholder, correct, data_set, error_file, rel):
  print('Relation Accurancy','*'*50, file = error_file)
  #check the different relation accurancy
  test_rel_id = feed_dict_devtest[placeholder['rel_placeholder']]

  # count the relation 
  cnt = defaultdict(int)
  for t in test_rel_id:
    cnt[t] += 1
  print('Relation Count', '*'*50, file = error_file)
  for c in cnt:
    print(c, cnt[c], file = error_file)

  # count the correct prediction for each relation
  right = {}
  for i in range(len(correct)):
    if test_rel_id[i] in right and correct[i]:
      right[test_rel_id[i]] += 1
    elif test_rel_id[i] not in right and correct[i]:
      right[test_rel_id[i]] = 1
    elif test_rel_id[i] not in right and not correct[i]:
      right[test_rel_id[i]] = 0

  # calculate the accurancy for different relation
  result = defaultdict(int)
  for j in cnt:
    result[j] = float(right[j])/float(cnt[j])

  # print out the result
  rel_dict = {}
  for w1 in rel:
    rel_dict[rel[w1]] = w1
    # print(rel_dict)
  for rel in result:
    acc = result[rel]
  #  print(rel)
    print(rel_dict[rel],rel, acc, file = error_file)
