#! /usr/bin/Python
# -*- coding: utf8 -*- 
import sys
import mxnet as mx
import numpy as np
import random
import bisect

# set up logging
import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

from lstm import lstm_unroll, lstm_inference_symbol
from chn_bucket_io import BucketSentenceIter
from rnn_model import LSTMInferenceModel

#---------
# helper strcuture for prediction
def MakeRevertVocab(vocab):
    dic = {}
    for k, v in vocab.items():
        dic[v] = k
    return dic

#----------
# make input from char
def MakeInput(char, vocab, arr):
    idx = vocab[char]
    tmp = np.zeros((1,))
    tmp[0] = idx
    arr[:] = tmp

#----------
# helper function for random sample 
def _cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def _choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = _cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]

# we can use random output or fixed output by choosing largest probability
def MakeOutput(prob, vocab, sample=False, temperature=1.):
    if sample == False:
        idx = np.argmax(prob, axis=1)[0]
    else:
        fix_dict = [""] + [vocab[i] for i in range(1, len(vocab) + 1)]
        scale_prob = np.clip(prob, 1e-6, 1 - 1e-6)
        rescale = np.exp(np.log(scale_prob) / temperature)
        rescale[:] /= rescale.sum()
        return _choice(fix_dict, rescale[0, :])
    try:
        char = vocab[idx]
    except:
        char = ''
    return char
#---------
# Read from doc
def read_content(path):
    with open(path) as ins:
        content = ins.read()
        return content
def seg_read_content(path):
	content = []
	fp = open(path,'r')
	for line in fp:
		s = []
		tokens = line.split(' ')
		if len(tokens) == 0:
			s.append('\n')
			content.append(s)
			continue
		for t in tokens:
			if t != '-':
				s.append(t)
		if tokens[:-1] != '\n':
			s.append('\n')
		content.append(s)
	fp.close()
	return content

# Build a vocabulary of what char we have in the content
def build_vocab(path):
    content = read_content(path)
    content = list(content)
    idx = 1 # 0 is left for zero-padding
    the_vocab = {}
    for word in content:
        if len(word) == 0:
            continue
        if not word in the_vocab:
            the_vocab[word] = idx
            idx += 1
    return the_vocab
def seg_build_vocab(path):
    #content = read_content(path)
    #content = list(content)
    content = seg_read_content(path)
    idx = 1 # 0 is left for zero-padding
    the_vocab = {}
    for s in content:
        for word in s:
            if len(word) == 0:
                continue
            if not word in the_vocab:
                the_vocab[word] = idx
                idx += 1
    return the_vocab
#----------
# number of lstm layer
num_lstm_layer = 3
# hidden unit in LSTM cell
num_hidden = 512
# embedding dimension, which is, map a char to a 256 dim vector
num_embed = 256
#vocab = build_vocab("./data/demo/wangfeng-rnn/input.txt")
vocab = seg_build_vocab("./data/demo/wangfeng-rnn/input.txt.seg")
#----------
# load from check-point
#_, arg_params, __ = mx.model.load_checkpoint("./data/demo/obama", 75)
_, arg_params, __ = mx.model.load_checkpoint("./data/demo/wangfeng-rnn/wangfeng-rnn", int(sys.argv[1]))

# build an inference model
model = LSTMInferenceModel(num_lstm_layer, len(vocab) + 1,
                           num_hidden=num_hidden, num_embed=num_embed,
                           num_label=len(vocab) + 1, arg_params=arg_params, ctx=mx.cpu(), dropout=0.2)
# generate a sequence of 1200 chars

seq_length = int(sys.argv[2])
input_ndarray = mx.nd.zeros((1,))
revert_vocab = MakeRevertVocab(vocab)
# Feel free to change the starter sentence
output =['我']
random_sample = True
new_sentence = True

ignore_length = len(output)

for i in range(seq_length):
    if i <= ignore_length - 1:
        MakeInput(output[i], vocab, input_ndarray)
    else:
        MakeInput(output[-1], vocab, input_ndarray)
    prob = model.forward(input_ndarray, new_sentence)
    new_sentence = False
    next_char = MakeOutput(prob, revert_vocab, random_sample)
    #print i,output[-1],next_char
    if next_char == '':
        new_sentence = True
    if next_char != '' and i >= ignore_length - 1:
        #output += next_char
        output.append(next_char)

# Let's see what we can learned from char in Obama's speech.
print ''.join(output)
