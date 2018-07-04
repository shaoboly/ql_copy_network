from data import Vocab
from batcher import Batcher

import config
import data
import os

FLAGS = config.FLAGS

vocab_in, vocab_out = data.load_dict_data(FLAGS)

batcher_train = Batcher(FLAGS.data_path, vocab_in,vocab_out, FLAGS, data_file='train.txt.tags')
epoch = 0
while True:
    print(epoch)
    while batcher_train.c_epoch==epoch:
        batch = batcher_train.next_batch()
    epoch += 1
    print("done")