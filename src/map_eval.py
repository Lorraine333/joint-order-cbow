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
import plus_input_data as input_data
import numpy as np
from plus_model import tf_model
from collections import defaultdict
from sklearn.metrics import average_precision_score

def fill_eval_feed_dict(data_set, placeholder, FLAGS):
  r_idx, t1_idx, t2_idx, labels = data_set.eval_batch()

  t1x, t1mask, t1length= input_data.prepare_data(t1_idx)
  t2x, t2mask, t2length = input_data.prepare_data(t2_idx)
  relmsk = input_data.rel_msk(r_idx, data_set._rel)

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

def rel_map(idx):
  rel_map = defaultdict(list)
  for i in range(len(idx)):
    rel_map[idx[i]].append(i)
  return rel_map

def MAP(errs, target):
  # invTarget = (target == 0).astype('float32')
  score = np.zeros((errs.shape))
  for i in range(errs.shape[0]):
    score[i] = -errs[i]
  ap = average_precision_score(target, score)
  return ap

def do_train_eval(sess,h_error, placeholder,data_set, num, neg_data, FLAGS, outfile):

  feed_dict = fill_eval_feed_dict(data_set, placeholder, FLAGS) 
  labels = feed_dict[placeholder['label_placeholder']]
  # for l in labels:
  #   if l == 1.0:s
  #     print(l)
  errors = sess.run(h_error, feed_dict = feed_dict)
  # relation index
  rel_idx = feed_dict[placeholder['rel_placeholder']]
  # since the query is each relation
  # get the map which is 'relations idx': [data index list]
  map_idx = rel_map(rel_idx)

  # mean average precision list
  map_list = []

  # print(labels)
  # print(errors)

  # map from relation index to relation name
  itorel = {data_set._rel[k]: k for k in data_set._rel}
  for i in map_idx:
    # print(errors[map_idx[i]])
    # print(labels[map_idx[i]])
    AP = MAP(errors[map_idx[i]], labels[map_idx[i]])
    print('AP for ',itorel[i], 'is',AP, file = outfile)
    map_list.append(AP)

  # print('MAP:',sum(map_list)/len(map_list))
  return sum(map_list)/len(map_list)
