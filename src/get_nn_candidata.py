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
from collections import Counter
# from future import print_function


cnt = Counter()
tuplelines = open('new_isa.txt','r').read().splitlines()
for line in tuplelines:
	parts = line.split('\t')
	words1 = parts[1].split(' ')
	words2 = parts[2].split(' ')
	for w1 in words1:
		cnt[w1] += 1
	for w2 in words2:
		cnt[w2] += 1
total_token = 0
for c in cnt.values():
	total_token+=c
print('total', total_token)
tuple_words = Counter(cnt).most_common(20)
for w in tuple_words:
	print(w[1]/total_token)
# print(tuple_words)

print('*'*50)


cnt1 = Counter()
textlines = open('text8','r').read().splitlines()
for w in textlines[0].split(' '):
	cnt1[w] += 1

tuple_words1 = Counter(cnt1).most_common(50)

total_cn = 0
for c in cnt1.values():
	# print c
	total_cn+=c
print(total_cn)

for tw in tuple_words1:
	print(tw[0])
	print(cnt1[tw[0]]/total_cn)
	print(cnt[tw[0]]/total_token)


# print(Counter(cnt1).most_common(20))

# with open('neighbors.txt', 'wt') as outfile:
# 	for tw in tuple_words:
# 		outfile.write(tw[0]+'\n')