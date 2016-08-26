import mxnet as mx
import numpy as np
import random
import bisect

# set up logging
import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

from lstm import lstm_unroll, lstm_inference_symbol
#from bucket_io import BucketSentenceIter
from chn_bucket_io import BucketSentenceIter
from rnn_model import LSTMInferenceModel

#--------
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
			#s.append('\n')
			#content.append(s)
			continue
		for t in tokens:
			if t == '-':
				continue
			s.append(t)
		if tokens[-1] != '\n':
			s.append('\n')
		content.append(s)
	fp.close()
	return content

# Build a vocabulary of what char we have in the content
def build_vocab(path):
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

# We will assign each char with a special numerical id
def text2id(sentence, the_vocab):
    #words = list(sentence)
    #words = [the_vocab[w] for w in words if len(w) > 0]
    words = [the_vocab[w] for w in sentence if len(w) > 0]
    return words
#--------
# Evaluation
def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)
#--------

# The batch size for training
batch_size = 32
# We can support various length input
# For this problem, we cut each input sentence to length of 129
# So we only need fix length bucket
buckets = [129]
# hidden unit in LSTM cell
num_hidden = 512
# embedding dimension, which is, map a char to a 256 dim vector
num_embed = 256
# number of lstm layer
num_lstm_layer = 3

#----------
# we will show a quick demo in 2 epoch
# and we will see result by training 75 epoch
num_epoch = 75
# learning rate
learning_rate = 0.001
# we will use pure sgd without momentum
momentum = 0.9
#---------
# we can select multi-gpu for training
# for this demo we only use one
#devs = [mx.context.gpu(i) for i in range(1)]
devs = [mx.context.cpu(i) for i in range(1)]
#--------
# build char vocabluary from input
vocab = build_vocab("./data/demo/wangfeng-rnn/input.txt.seg")
#--------
# generate symbol for a length
def sym_gen(seq_len):
    return lstm_unroll(num_lstm_layer, seq_len, len(vocab) + 1,
                       num_hidden=num_hidden, num_embed=num_embed,
                       num_label=len(vocab) + 1, dropout=0.2)
#--------
# initalize states for LSTM
init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h
#---------
# we can build an iterator for text
data_train = BucketSentenceIter("./data/demo/wangfeng-rnn/input.txt.seg", vocab, buckets, batch_size,
                                init_states, seperate_char='\n',
                                text2id=text2id, read_content=seg_read_content)
#---------
# the network symbol
symbol = sym_gen(buckets[0])
#---------
# Train a LSTM network as simple as feedforward network
model = mx.model.FeedForward(ctx=devs,
                             symbol=symbol,
                             num_epoch=num_epoch,
                             learning_rate=learning_rate,
                             momentum=momentum,
                             wd=0.0001,
                             initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
#---------
# Fit it
model.fit(X=data_train,
          eval_metric = mx.metric.np(Perplexity),
          batch_end_callback=mx.callback.Speedometer(batch_size, 50),
          epoch_end_callback=mx.callback.do_checkpoint("./data/demo/wangfeng-rnn/wangfeng-rnn"))
#---------
