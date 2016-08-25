#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
import re
import mxnet as mx
import numpy as np
import time
import math
import data_helpers
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # get a logger to accuracies are printed

logs = sys.stderr

def predict(model, X_list):
	m = model
	#print >> logs, 'X_list:%s' % (X_list)
        # predict
        """
	batch_size = 1
        for begin in range(0, X_list.shape[0], batch_size):
            batchX = X_list[begin:begin+batch_size]

            if batchX.shape[0] != batch_size:
                continue

            m.data[:] = batchX
            m.cnn_exec.forward(is_train=False)

            batchY = np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1)
	    print >> logs, 'ID[%d - %d] Label[%s]' % (begin, begin+batch_size-1, batchY)
            #num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            #num_total += len(batchY)
        """
	"""
        for i in range(0, len(X_list)):
            print >> logs, 'X:%s' % (np.array(X_list[i]))
            prediction = model.predict(np.array(X_list[i]))
            #prediction = model.predict(X_list[i])
            print >> logs, 'ID[%d] Label[%s]' % (i, prediction)
	"""
	batch_size = 50
        for begin in range(0, X_list.shape[0], batch_size):
            batchX = X_list[begin:begin+batch_size]

            if batchX.shape[0] != batch_size:
                continue
	    print >> logs, 'X:%s' % (batchX)
	    prediction = model.predict(batchX)
	    print "prediction shape:",np.array(prediction).shape
	    print >> logs, 'ID[%d - %d] Label[%s]' % (begin, begin+batch_size-1, np.argmax(np.array(prediction), axis=1))

def load_model(prefix, load_epoch):
	# Load the pre-trained model
	tmp_model = mx.model.FeedForward.load(prefix, load_epoch, ctx=mx.cpu())
	model = mx.model.FeedForward(symbol=tmp_model.symbol, ctx=mx.cpu(), numpy_batch_size=50, arg_params=tmp_model.arg_params, aux_params=tmp_model.aux_params)
        print >> logs, 'Load model done!'
	return model

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data():
    # Load data from files
    test_examples = list(open("./example/cnn_text_classification/data/rt-polaritydata/rt-polarity.pos").readlines())
    test_examples = [s.strip() for s in test_examples]
    # Split by words
    x_text = test_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    return x_text

def load_vocab():
    """
    Load vocab for predict process
    """
    vocabulary = {}
    vocabulary_inv=[]
    vocab_file = "./example/cnn_text_classification/data/vocab"
    vocabinv_file = "./example/cnn_text_classification/data/vocab-inv"
    #load mapping from index to word
    fp_vinv = open(vocabinv_file,'r')
    for line in fp_vinv:
        tokens = line.strip().split("\t")
        if len(tokens) != 2:
            continue
        index = int(tokens[0])
        vocab = tokens[1]
        vocabulary_inv.append(vocab)
    fp_vinv.close()
    #load mapping from word to index
    fp_v = open(vocab_file, 'r')
    for line in fp_v:
        tokens = line.strip().split("\t")
        if len(tokens) != 2:
            continue
        index = int(tokens[1])
        vocab = tokens[0]
        vocabulary[vocab] = index
    fp_v.close()
    print "vocabulary size %s" % len(vocabulary)
    return [vocabulary, vocabulary_inv]

def pad_sentences(sentences, vocabulary, padding_word="</s>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = 56
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        use_sentence = [w for w in sentence if w in vocabulary]
        num_padding = sequence_length - len(use_sentence)
        new_sentence = use_sentence
        if num_padding <=0:
            new_sentence = use_sentence[:sequence_length]
        else:
            new_sentence = use_sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences
def build_input_data(sentences, vocabulary):
    """
    Maps sentencs to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence if word in vocabulary] for sentence in sentences])
    return x

if __name__ == '__main__':
    #load pre-trained model
    model = load_model(prefix="./example/cnn_text_classification/checkpoint/cnn", load_epoch=19)
    #load saved vocabulary
    vocabulary, vocabulary_inv = load_vocab()
    #load input test data
    sentences = load_data()
    sentences_padded = pad_sentences(sentences, vocabulary)
    #build input vector
    x = build_input_data(sentences_padded, vocabulary)
    print "x_train.shape: ",x.shape
    #predict
    predict(model, x) 
