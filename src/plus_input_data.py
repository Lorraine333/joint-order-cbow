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

from __future__ import division
from __future__ import print_function
from random import shuffle
from random import randint
from random import choice
import numpy as np
import tensorflow as tf
import collections

class DataSet(object):

  def __init__(self, input_tuples):
    """Construct a DataSet"""
    self._num_examples = len(input_tuples)
    self._tuples = input_tuples
    # self._rel = rel
    # self._words = words
    # self._Rel = Rel
    # self._We = We,
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  @property
  def index_in_epoch(self):
    return self._index_in_epoch

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      # perm = np.arange(self._num_examples)
      # shuffle(perm)
      shuffle(self._tuples)
      # self._tuples = self._tuples[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    next_batch = self._tuples[start:end]
    # print('train next batch', next_batch)
    # batch_idx = [convertToIndex(i, self._words, self._We, self._rel, self._Rel) for i in next_batch]
    batch_idx = [i for i in next_batch]
    r_idx, t1_idx, t2_idx, s = sep_idx(batch_idx)
    l = np.ones(len(s))
    for i in range(len(s)):
      if s[i] <= 0:
        l[i] = 0.
    # print np.asarray(r_idx).shape
    return np.asarray(r_idx), t1_idx, t2_idx, l

  def eval_batch(self):
    """Return the next whole examples from this eval data set."""

    # perm = np.arange(self._num_examples)
    # shuffle(perm)
    # print(self._tuples)
    # print(perm)
    # self._tuples = self._tuples[perm]
    end = self._num_examples
    next_batch = self._tuples[0:end]
    # print('next_batch', next_batch)
    # batch_idx = [convertToIndex(i, self._words, self._We, self._rel, self._Rel) for i in next_batch]
    batch_idx = [i for i in next_batch]
    r_idx, t1_idx, t2_idx, s = sep_idx(batch_idx)
    l = np.ones(len(s))
    for i in range(len(s)):
      if s[i] <= 0:
        l[i] = 0.
    # print np.asarray(r_idx).shape
    return np.asarray(r_idx), t1_idx, t2_idx, l


def read_data_sets(FLAGS, outfile, dtype=tf.float32):
  train_dir = FLAGS.train_dir
  embed_dim = FLAGS.embed_dim
  lower = FLAGS.lower_scale
  higher = FLAGS.higher_scale
  class DataSets(object):
    pass
  data_sets = DataSets()

  trainfile = FLAGS.train
  testfile = FLAGS.test
  if FLAGS.eval == 'map':
    testfile = 'many_neg_'+testfile
  # 1207 exp
  TRAIN_FILE = train_dir+'/training/msr_omcs/'+trainfile+'.txt'
  # TRAIN_FILE = train_dir+'/training/msr_omcs/noisa.txt'
  #TRAIN_FILE = train_dir+'/training/msr_omcs/all_rel.txt'
  DEV_FILE = train_dir+'/eval/msr_omcs/'+testfile+'_dev1.txt'
  if FLAGS.overfit:
    DEVTEST_FILE = train_dir+'/eval/msr_omcs/'+testfile+'_dev1.txt'
  else:
    DEVTEST_FILE = train_dir+'/eval/msr_omcs/'+testfile+'_dev2.txt'
  TEST_FILE = train_dir+'/eval/msr_omcs/'+testfile+'_test.txt'
  DICT_FILE = train_dir + '/training/msr_omcs/dict.txt'
  words, We = get_word_net(DICT_FILE)
  REL_FILE = train_dir+'/training/rel.txt'
  TRAIN_TEST_FILE = train_dir + '/training/msr_omcs/many_neg_isa.txt'
  WORD2VEC_FILE = FLAGS.word2vec_train_data
  TRAIN_NEG_FILE = train_dir + '/training/extract_not/omcs100_neg.txt'
  NN_FILE = 'neighbors.txt'

  #wordnet
  #TRAIN_FILE = train_dir+'/training/wordnet_train.txt'
  #DEV_FILE = train_dir+'/eval/wordnet_valid.txt'
  #DEVTEST_FILE = train_dir+'/eval/wordnet_test.txt'
  #DICT_FILE = train_dir + '/training/wordnet_dict.txt'
  #words, We = get_word_net(DICT_FILE)
  
  #msr_concept_data
  # TRAIN_FILE = train_dir+'/training/all_train.txt'
  # # TRAIN_FILE = train_dir+'/training/wordnet_clean.txt'
  # DEV_FILE = train_dir+'/eval/wordnet_valid_clean.txt'
  # DEVTEST_FILE = train_dir+'/eval/wordnet_test_clean.txt'
  # #useless
  # TRAIN_NEG_FILE = train_dir + '/training/extract_not/omcs100_neg.txt'
  # DICT_FILE = train_dir + '/training/all_dict.txt'
  # words, We = get_word_net(DICT_FILE)
  # EMBED_FILE = train_dir+'/embedding/embeddings.skip.newtask.en.d'+str(embed_dim)+'.m1.w5.s0.it20.txt'
  # words, We = getWordmap(EMBED_FILE)

  #conceptnet
  # TRAIN_FILE = train_dir+'/training/extract_not/omcs100_pos.txt'
  # TRAIN_NEG_FILE = train_dir + '/training/extract_not/omcs100_neg.txt'
  # TRAIN_TEST_FILE = train_dir + '/training/train_test.txt'
  # DEV_FILE = train_dir+'/eval/new_omcs_dev1.txt'
  # DEVTEST_FILE = train_dir+'/eval/new_omcs_dev2.txt'
  # EMBED_FILE = train_dir+'/embedding/embeddings.skip.newtask.en.d'+str(embed_dim)+'.m1.w5.s0.it20.txt'
  # words, We = getWordmap(EMBED_FILE)

  #conceptnet_isa
  # TRAIN_FILE = train_dir+'/training/new_omcs100_withneg.txt'
  # DEV_FILE = train_dir+'/eval/omcs_dev1_isa.txt'
  # DEVTEST_FILE = train_dir+'/eval/omcs_dev2_isa.txt'
  # EMBED_FILE = train_dir+'/embedding/embeddings.skip.newtask.en.d'+str(embed_dim)+'.m1.w5.s0.it20.txt'
  # words, We = getWordmap(EMBED_FILE)

  

  # print(words)
  rel, Rel, spe_idx = getRelation(REL_FILE, embed_dim, lower, higher)

  train_data = getData(TRAIN_FILE, outfile, words, We, rel, Rel)
  train_neg_data = getData(TRAIN_NEG_FILE, outfile, words, We, rel, Rel)
  train_test = getData(TRAIN_TEST_FILE, outfile, words, We, rel, Rel)
  dev_data = getData(DEV_FILE, outfile, words, We, rel, Rel)
  devtest_data = getData(DEVTEST_FILE, outfile, words, We, rel, Rel)
  test_data = getData(TEST_FILE, outfile, words, We, rel, Rel)
  with open(NN_FILE) as inputfile:
    nn_data = inputfile.read().splitlines()

  
  data_sets.train = DataSet(train_data)
  data_sets.train_neg = DataSet(train_neg_data)
  data_sets.train_test = DataSet(train_test)
  data_sets.dev = DataSet(dev_data)
  data_sets.devtest = DataSet(devtest_data)
  data_sets.test = DataSet(test_data)
  data_sets.word2vec_data, data_sets.count, data_sets.words  = build_dataset(WORD2VEC_FILE ,FLAGS.vocab_size, words)
    
  data_sets.nn_data = nn_data
  data_sets.rel = rel
  # data_sets.spe_rel_idx = spe_idx
  #data_sets.words = words
  data_sets.Rel = Rel
  data_sets.We = We

  data_sets.input_size = embed_dim
  return data_sets

def getWordmap(filename):
    words={}
    We = []
    f = open(filename,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]=n
        We.append(v)
    return (words, np.matrix(We, dtype = np.float32))

def get_word_net(filename):
    words={}
    We = []
    f = open(filename,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
      #n is the line number-1 (start from 0), i is the actually word
      i=i.strip()
      v = np.zeros(50)
      words[i]=n
      We.append(v)
    return (words, np.matrix(We, dtype = np.float32))

def getRelation(filename, embed_size, lower, higher):
    rel = {}
    f = open(filename,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i = i.strip()
        rel[i] = n
    #define special relation
    # spe_rel = ['desires','capableof','hasproperty','isa','hasa','madeof']
    spe_rel = ['desires','capableof','hasproperty','hasa','madeof']
    spe_idx = []
    for j in rel:
      if j in spe_rel:
        spe_idx.append(rel[j])
    return rel, np.random.uniform(lower, higher, (len(lines),embed_size)).astype('float32'), spe_idx

def getData(filename, outfile, words, We, rel, Rel):
    data = open(filename,'r')
    lines = data.readlines()
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            e = (i[0], i[1], i[2], float(i[3]))
            e_idx = convertToIndex(e, words, We, rel, Rel)
            examples.append(e_idx)
            # examples.append(e)
            # print('idx', e_idx)
    # print('examples',examples)
    shuffle(examples)
    print('read into data from',filename,'of length',len(examples), file = outfile)
    # return np.asarray(examples)
    return examples

# def get_neg_Data(filename):
#     data = open(filename,'r')
#     lines = data.readlines()
#     pos_examples = []
#     neg_examples = []
#     for i in lines:
#         i=i.strip()
#         if(len(i) > 0):
#             i=i.split('\t')
#             e = (i[0], i[1], i[2], float(i[3]))
#             if float(i[3]) > 0:
#               pos_examples.append(e)
#             else:
#               neg_examples.append(e)
#     shuffle(pos_examples)
#     shuffle(neg_examples)
#     print 'read into positive data from',filename,'of length',len(pos_examples)
#     print 'read into negative data from',filename,'of length',len(neg_examples)
#     return [np.asarray(pos_examples), np.asarray(neg_examples)]

def convertToIndex(e ,words, We, rel, Rel):
  if len(e) > 1:
    (r,t1,t2,s) = e
    # print rel
    return (lookupRelIDX(rel, r),lookupwordID(words, t1), lookupwordID(words, t2), float(s))
  else:
    return (lookupwordID(words, e))

def lookupwordID(words,w):
  result = []
  array = w.split(' ')
  for i in range(len(array)):
    if(array[i] in words):
      result.append(words[array[i]])
    else:
      result.append(words['UUUNKKK'])
  return result

def lookupRelIDX(rel_dict,r):
  # print rel_dict
  r = r.lower()
  if r in rel_dict:
    return rel_dict[r]
  else:
    # print r
    return rel_dict['UUUNKKK']

def sep_idx(e_idx):
  g1=[];g2=[];R=[];labels = []
  for e in e_idx:
    # print(e)
    (r, t1, t2, s) =e
    g1.append(t1)
    g2.append(t2)
    R.append(r)
    labels.append(s)
  return R, g1, g2, labels


def prepare_data(list_of_seqs):
  lengths = [len(s) for s in list_of_seqs]
  n_samples = len(list_of_seqs)
  maxlen = np.max(lengths)
  x = np.zeros((n_samples, maxlen)).astype('int32')
  x_mask = np.zeros((n_samples, maxlen, 1)).astype('int32')
  x_len = np.zeros((n_samples,1)).astype('int32')
  for idx, s in enumerate(list_of_seqs):
    x[idx, :lengths[idx]] = s
    x[idx, lengths[idx]:] = 0.
    x_len[idx,:] = lengths[idx]
    x_mask[idx, :lengths[idx]] = 1.
  return x, x_mask, x_len

def rel_msk(rel_idx, rel):
  relmsk = np.ones((rel_idx.shape[0], 1)).astype('float32')
  for i in range(len(rel_idx)):
    if rel['isa']== int(rel_idx[i]):
      relmsk[i] = 0 
  return relmsk.reshape(relmsk.shape[0], 1)

def inverse_dict(words):
  result = {}
  for i in words:
    result[words[i]] = i
  return result

def convertToWord(e ,words, We, rel, Rel):
  # print e
  (r, t1, t2, s) = e
  inv_rel = inverse_dict(rel)
  inv_words = inverse_dict(words)
  # print inv_rel[r]
  wt1 = []
  wt2 = []
  for i in t1:
    wt1.append(inv_words[i])
  for j in t2:
    wt2.append(inv_words[j])
  # print wt1, wt2
  return (inv_rel[r], wt1, wt2, s)

def read_data(filename):
  lines = open(filename,'r').read().splitlines()
  words = []
  for line in lines:
    parts = line.split()
    for p in parts:
      words.append(p)
  return words

def build_dataset(filename, vocabulary_size, vocab):
  words = read_data(filename)
  count = collections.Counter(words)
  for w in dict(count):
    if count[w] <= 2:
      del count[w]
  for word in count:
    if(word not in vocab):
       vocab[word] = len(vocab)
  data = list()
  unk_count = 0
  for word in words:
    if word in vocab:
      index = vocab[word]
    else:
      index = vocab['UUUNKKK']  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  del words
  print(len(vocab))
  return data, count, vocab


def find_neg(r_idx, t1_idx, t2_idx, labels):
  #if is't one of the specific relation, find neg example in their not version
  # if special_neg_sample:
    # r_idx, neg_idx1, neg_idx2, neg_labels = spe_neg(r_idx, t1_idx, t2_idx, labels)
  # else:
  r_idx, neg_idx1, neg_idx2, neg_labels = normal_neg(r_idx, t1_idx, t2_idx, labels)
  return r_idx, neg_idx1, neg_idx2, neg_labels

def normal_neg(r_idx, t1_idx, t2_idx, labels):
  neg_idx1 = []
  neg_idx2 = []
  neg_labels = []
  for i in range(len(r_idx)):
    #only randoms change term1 or term2
    pointer = randint(0,1)
    wpick1 = ''
    if pointer == 1:
      wpick = ''
      #change term1, term2 remain unchanged
      neg_idx2.append(t2_idx[i])         
      while(wpick==''):
        index=randint(0,len(t1_idx)-1)
        if(index!=i):
          wpick = t1_idx[index]
          neg_idx1.append(wpick)    
    else:
      #change term2, term1 remain unchanged
      neg_idx1.append(t1_idx[i])         
      while(wpick1==''):
        index1=randint(0,len(t2_idx)-1)
        if(index1!=i):
          wpick1 = t2_idx[index1]
          neg_idx2.append(wpick1)  
    neg_labels.append(int(0.)) 
  # print 'here',len(r_idx), len(neg_idx1), len(neg_idx2), len(neg_labels)
  return r_idx, neg_idx1, neg_idx2, neg_labels

def spe_neg(r_idx, t1_idx, t2_idx, labels, neg_data, spe_idx):
  neg_idx1 = []
  neg_idx2 = []
  neg_labels = []
  for i in range(len(r_idx)):
    if r_idx[i] in spe_idx:
      cur_rel = r_idx[i]
      idx = [convertToIndex(i, neg_data._words, neg_data._We, neg_data._rel, neg_data._Rel) for i in neg_data._tuples]
      neg_tuple = choice(idx)
      neg_rel = neg_tuple[0]
      while neg_rel != cur_rel:
        neg_tuple = choice(idx)
        neg_rel = neg_tuple[0]
      # print convertToWord(neg_tuple, neg_data._words, neg_data._We, neg_data._rel, neg_data._Rel)
      neg_idx1.append(neg_tuple[1])
      neg_idx2.append(neg_tuple[2])
      neg_labels.append(int(0.))
    else:
      #only randoms change term1 or term2
      pointer = randint(0,1)
      wpick1 = ''
      if pointer == 1:
        wpick = ''
        #change term1, term2 remain unchanged
        neg_idx2.append(t2_idx[i])         
        while(wpick==''):
          index=randint(0,len(t1_idx)-1)
          if(index!=i):
            wpick = t1_idx[index]
            neg_idx1.append(wpick)    
      else:
        #change term2, term1 remain unchanged
        neg_idx1.append(t1_idx[i])         
        while(wpick1==''):
          index1=randint(0,len(t2_idx)-1)
          if(index1!=i):
            wpick1 = t2_idx[index1]
            neg_idx2.append(wpick1)  
      neg_labels.append(int(0.)) 
  # print 'here',len(r_idx), len(neg_idx1), len(neg_idx2), len(neg_labels)
  return r_idx, neg_idx1, neg_idx2, neg_labels





