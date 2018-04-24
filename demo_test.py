from data import Vocab
from batcher import Batcher

import config

import os

FLAGS = config.FLAGS
vocab_path = os.path.join(FLAGS.data_path, "vocab.in")
vocab = Vocab(vocab_path, FLAGS.vocab_size)

vocab_path = os.path.join(FLAGS.data_path, "vocab.out")
vocab_out = Vocab(vocab_path, FLAGS.vocab_size)

subwords_embedding = vocab_out.compute_predicate_indices_split(vocab)

batcher_train = Batcher(FLAGS.data_path, vocab,vocab_out, FLAGS, data_file='train.txt')
epoch = 0
while True:
    print(epoch)
    while batcher_train.c_epoch==epoch:
        batch = batcher_train.next_batch()
    epoch += 1
    print("done")