#! /usr/bin/env python

import tensorflow as tf
import inspect
import os, sys
sys.path.append("..")
import os.path
from singleton import Singleton
import codecs
from cnn_multilabel_classification.config import BaseConfig
from sys import argv
import heapq
from log_helper import LogHelper
import numpy as np
import time


class Classify_CN(BaseConfig, metaclass=Singleton):

    wv = None
    lh = None

    def __init__(self, module_dir, user='user'):
        BaseConfig.__init__(self)
        print("module dir: " + module_dir)

        self.test_config = BaseConfig()
        self.test_config.batch_size = 1
        self.wv = self.test_config.wv

        # tf.flags.DEFINE_integer("embedding_dim_cn", 300, "Dimensionality of character embedding (default: 128)")
        # tf.flags.DEFINE_integer("batch_size_classify", 1, "Batch Size (default: 64)")
        #
        # self.FLAGS = tf.flags.FLAGS
        # self.FLAGS._parse_flags()
        # print("\nParameters:")
        # for attr, value in sorted(self.FLAGS.__dict__['__flags'].items()):
        #     print("{}={}".format(attr.upper(), value))
        # print("")

        checkpoint_dir = os.path.join(module_dir, "cn", "checkpoints")
        classes_file = codecs.open(os.path.join(module_dir, "cn", "classes"), "r", "utf-8")
        self.classes = list(line.strip() for line in classes_file.readlines())
        classes_file.close()
        
        print("\nEvaluating...\n")
        
        # Evaluation
        # ==================================================
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/cpu:0"):
                session_conf = tf.ConfigProto(
                  allow_soft_placement=self.test_config.allow_soft_placement,
                  log_device_placement=self.test_config.log_device_placement)
                session_conf.gpu_options.allow_growth = True
                self.sess = tf.Session(config=session_conf)
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
                
                # Get the placeholders from the graph by name
                self.embedded_chars = graph.get_operation_by_name("embedded_chars").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                
                # Tensors we want to evaluate
                self.scores = graph.get_operation_by_name("output/scores").outputs[0]
                self.probabilities = graph.get_operation_by_name("output/probabilities").outputs[0]

        this_file = inspect.getfile(inspect.currentframe())
        dir_name = os.path.abspath(os.path.dirname(this_file))
        self.chat_log_path = os.path.join(dir_name, '..', 'log/module/cnn_classify')
        if not os.path.exists(os.path.join(self.chat_log_path, user)):
            if not os.path.exists(self.chat_log_path):
                os.makedirs(self.chat_log_path)
            f = open(self.chat_log_path+'/' + user, 'w', encoding='utf-8')
            f.close()

        if not self.lh:
            self.lh = LogHelper(user, self.chat_log_path)

    def __enter__(self):
        print('Classify_CN enter')

    def __exit__(self):
        print('Classify_CN exit')
        self.sess.close()

    def getCategory(self, _sentence, top_number=5, reverse=False):

        sentence_raw = _sentence
        if reverse:
            sen_list = _sentence.strip().split(' ')
            sen_list.reverse()
            _sentence = ' '.join(sen_list)
        else:
            pass

        x_test = [_sentence]
 
        # # Generate batches for one epoch
        # batches = list(data_helpers.batch_iter(list(x_test), self.test_config.batch_size, 1, shuffle=False))
        #
        # # Collect the predictions here
        # x_test_batch = batches[0]
        # test_batch = list(x_test_batch)
        sentence_embedded_chars = self.wv.embedding_lookup(
            len(x_test), self.test_config.sentence_words_num, self.test_config.embedding_dim_cn_train, x_test)

        # 是否增加词性特征
        if self.embedded_chars.shape[2] == sentence_embedded_chars.shape[2]:
            expend_input = sentence_embedded_chars
        else:
            from tools.word_cut import WordCutHelper
            wh = WordCutHelper(1)
            tag = wh.getTagAndWord(sentence_raw.replace(' ', ''))['tag']
            if reverse:
                tag.reverse()
            else:
                pass

            tag_x = np.zeros((self.sentence_words_num, len(self.tags_table)))
            idx_list = map(lambda x: self.tags_table.index(x), tag)

            for i, idx in enumerate(idx_list):
                tag_x[i, idx] = 1

            tag_x = np.expand_dims(tag_x, axis=0)

            expend_input = np.concatenate((sentence_embedded_chars, tag_x), axis=2)

        feed_dict = {
          self.embedded_chars: expend_input,
          self.dropout_keep_prob: 1.0
        }

        batch_scores = self.sess.run(self.scores, feed_dict)

        batch_probabilities = self.sess.run(self.probabilities, feed_dict)

        top_index = heapq.nlargest(top_number, range(len(batch_probabilities[0])), batch_probabilities[0].take)

        num_classes = len(self.classes)

        if num_classes > 5:
            top_number = 5
        else:
            top_number = num_classes

        top_classes = []
        top_scores = []
        top_probabilities = []
        for i in range(top_number):
            top_scores.append(float(batch_scores[0][top_index[i]]))
            top_classes.append(self.classes[top_index[i]])
            top_probabilities.append(float(batch_probabilities[0][top_index[i]]))

        _result = {'sentence': sentence_raw,
                   'score': float(top_scores[0]),
                   'probability': float(top_probabilities[0]),
                   'value': top_classes[0],
                   'top5_value': top_classes,
                   'top5_score': top_scores,
                   'top5_probability': top_probabilities
                   }

        return _result


if __name__ == '__main__':
    try:
        script, module_path = argv
    except ValueError:
        module_path = '../work_space/banktest/module/cnn_multi'

    print("model path: {}".format(module_path))

    from tools.word_cut import WordCutHelper
    wc = WordCutHelper(1)

    classify = Classify_CN(module_path)
    while 1:
        sentence = input('sentence: ')

        # value = wh.getWords(sentence)
        # sentence = ' '.join(value)

        r = wc.getTagAndWord(sentence)
        print(r)
        words = r['word']
        sentence = ' '.join(words)

        print('sentence after cut: {}'.format(sentence))

        if not sentence:
            break

        tic = time.time()
        result = classify.getCategory(sentence, top_number=5)
        toc = time.time()

        print(result)
        print('Time cost: {}'.format((toc-tic)))
