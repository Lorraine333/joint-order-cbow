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

# coding: utf-8

# #Read In Variables

# In[1]:
from __future__ import division
from __future__ import print_function
import pickle
import sys
import numpy as np
import tensorflow as tf
from scipy.spatial import distance



# In[2]:

def get_words(filename):
    words={}
    f = open(filename,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.strip()
        words[i]=n
    return words


# In[3]:

def get_rel(filename):
    rel = {}
    f = open(filename,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i = i.strip()
        rel[i] = n
    return rel


# #Get Embedings for Tuples

# In[7]:

def dist(v1, v2):
    return distance.euclidean(tuple(v1),tuple(v2))


# In[8]:

def get_term_embedding(lines, words, We, term1 = False):
    t2 = []
    for line in lines:
        if term1:
            term = line.split('\t')[1]
        else:
            term = line.split('\t')[2]
        array = term.split()
        if array[0] in words:
            vec = We[words[array[0]],:]
        else:
            vec = We[words['UUUNKKK'],:]
#             print('can not find corresponding vector:',array[0].lower())
        for i in range(len(array)-1):
            if array[i+1] in words:
                vec = vec + We[words[array[i+1]],:]
            else:
#                 print('can not find corresponding vector:',array[i+1].lower())
                vec = vec + We[words['UUUNKKK'],:]
        vec = vec/len(array)
        t2.append(vec)
    return t2


# In[9]:

def read_file(filename):
    lines = open(filename, 'r').read().splitlines()
    return lines


# In[10]:

def get_rel_embedding(lines, rel, Rel):
    result = []
    for line in lines:
        single_rel = line.split('\t')[0].lower()
        result.append(Rel[int(rel[single_rel])])
    return result
        

# #NN Calculation

# In[84]:

def topk_nn(term1, rel, cand_term2, k):
    result = []
    for i in range(len(term1)):
        t = term1[i]
        r = rel[i]
        temp = []
        ct_list = []
        for j in range(len(cand_term2)):
            ct2 = cand_term2[j]
            if np.sum(np.abs(ct2-t)) >1e-3:
                ct_list.append(j)
                temp.append(dist(t+r, ct2))
        # get the top k ranking for term1 and rel
        # idx is the index of the candidate term2 from the training set + test set
        idx = np.argpartition(np.asarray(temp),k)[:k]
#         print(idx)
        
        result.append(np.asarray(ct_list)[idx])
    return np.asarray(result)


# In[85]:

def get_term(query_tuples, cand_tuples, idx, n):
    result = []
    term1 = []
    rel = []
    cand_term2 = []
    for q_l in query_tuples:
        term1.append(q_l.split('\t')[1])
        rel.append(q_l.split('\t')[0])
    for c_l in cand_tuples:
        cand_term2.append(c_l.split('\t')[2])
    for i in idx:
        cand_lines = np.asarray(cand_term2)[i]
        result.append(cand_lines)

    return term1[:n], rel[:n], np.asarray(result)


if __name__ == "__main__":

    my_model = sys.argv[1] # params_isa/reg1dim50steps10000
    train = sys.argv[2] # rel_test/all_rel
    test = sys.argv[3] #isa_dev1
    outfile = open(my_model+'_'+train+'_'+test+'.txt','wt')
    k = 10
    num = 100

    train_dir = '../data'
    DICT_FILE = train_dir + '/training/msr_omcs/dict.txt'
    REL_FILE = train_dir+'/training/rel.txt'

    # Get words and relation dictionaries
    words = get_words(DICT_FILE)
    rel = get_rel(REL_FILE)

    # Reload the model and get embeddings and relation matrix
    model = pickle.load(open(my_model, "rb"))
    Rel = model['rel']
    We = model['embeddings']

    # the test set we want to evaluation on, and the candidate term2 should be generated from which file
    TRAIN = train_dir+'/training/msr_omcs/'+train+'.txt'
    DEV_FILE = train_dir+'/eval/msr_omcs/'+test+'_pos.txt'
    train_lines = read_file(TRAIN)
    dev_lines = read_file(DEV_FILE)

    # Get embeddings for them 
    dev_t2_cand_embed = get_term_embedding(train_lines+dev_lines, words, We, False)
    dev_t1_embed = get_term_embedding(dev_lines, words, We, True)
    dev_rel_embed = get_rel_embedding(dev_lines, rel, Rel)
    print('*'*50)
    print('Test Term2 Candidate Length:', len(dev_t2_cand_embed))
    print('Test Term1 Length:', len(dev_t1_embed))
    print('Test Relation Length:', len(dev_t1_embed))
    print('*'*50)
    
    # keep top k nearest neighbors 
    idx = topk_nn(dev_t1_embed[:num], dev_rel_embed[:num], dev_t2_cand_embed, k)

    t1,r,t2=get_term(dev_lines, train_lines+dev_lines, idx, num)
    
    for j in range(num):
        print(t1[j],r[j],t2[j],'\n', file = outfile)
        #for n in range(len(t2[j])):
        #    print(t2[j][n]+' ', file = outfile)
        #print('\n', file = outfil





