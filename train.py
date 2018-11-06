#! /usr/bin/env python
# coding:utf-8

import sys
sys.path.append('..')
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import cnn_multilabel_classification.data_helper as data_helpers
from cnn_multilabel_classification.cnn_attention_model import TextCNN_PreTrained
import codecs
from sys import argv
import math
from cnn_multilabel_classification.config import TrainConfig as Config
import re

try:
    script, scene = argv
except ValueError:
    scene = 'banktest'

print('=====scene: {}====='.format(scene))

tic = int(time.time())

# scene = 'banktest'
config = Config(scene)
expend = config.append_tag
if expend:
    dim_input = config.embedding_dim_cn_train + len(config.tags_table)
else:
    dim_input = config.embedding_dim_cn_train

# Data Preparation
# ==================================================
print("Loading data...")
# train_data, dev_data, classes = data_helpers.load_data(
#     config.data_path, config.dev_sample_percentage,
#     sort_by_len=config.sort_by_len, enhance=config.data_enhance, reverse=config.data_reverse, data_limit=None)
train_data, dev_data, classes = data_helpers.load_data(config.train_data_path, config.dev_sample_percentage,
                                                       sort_by_len=config.sort_by_len, enhance=config.data_enhance,
                                                       reverse=config.data_reverse,
                                                       word_cut=False, data_limit=None)

train_set = list(zip(*train_data))
dev_set = list(zip(*dev_data))
# if expend:
#     x_train, y_train, tag_train = train_data
#     x_dev, y_dev, tag_dev = valid_data
# else:
#     x_train, y_train = train_data
#     x_dev, y_dev = valid_data


# 知乎看山杯评价指标
def evaluate(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]

    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)

    """
    right_label_num = 0  # 总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  # 在各个位置上总命中数量
    sample_num = 0  # 总问题数量
    all_marked_label_num = 0  # 总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:  # 命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, config.top_number), right_label_at_pos_num):
        precision += (right_num / float(sample_num)) / \
            math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num

    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = (precision * recall) / (precision + recall)

    return f1


# 从probabilities中取出前五 get label using probs
def get_label_using_probs(probability, top_number=1):
    index_list = np.argsort(probability)[-top_number:]
    index_list = index_list[::-1]
    return index_list


# Training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=config.allow_soft_placement,
        log_device_placement=config.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        # if config.use_attention:
        cnn = TextCNN_PreTrained(
            sequence_length=config.sentence_words_num,
            num_classes=len(classes),
            embedding_size=dim_input,
            filter_sizes=list(map(int, config.filter_sizes.split(","))),
            num_filters=config.num_filters,
            l2_reg_lambda=config.l2_reg_lambda,
            attention_dim=config.attention_dim,
            use_attention=config.use_attention)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        ckpt_path = os.path.join(config.out_dir, "cn")
        print("Writing to {}\n".format(ckpt_path))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge(
            [loss_summary, acc_summary, grad_summaries_merged])
        # train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(ckpt_path, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(ckpt_path, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(
            os.path.join(ckpt_path, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=config.num_checkpoints)

        # Write classify names
        classes_file = codecs.open(os.path.join(
            ckpt_path, "classes"), "w", "utf-8")
        for classify_name in classes:
            classes_file.write(classify_name)
            classes_file.write('\n')
        classes_file.close()

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(_x_batch, _y_batch, _tag=None):
            """
            A single training step
            """
            input_x = list(_x_batch)
            embedded_chars = config.wv.embedding_lookup(
                len(input_x), config.sentence_words_num, config.embedding_dim_cn_train, input_x, 0)

            # embedded_chars = np.array(embedded_chars)

            if _tag:
                _tag = np.array(_tag)
                input_expand = np.concatenate((embedded_chars, _tag), axis=2)
            else:
                input_expand = embedded_chars

            feed_dict = {
                cnn.input_y: _y_batch,
                cnn.embedded_chars: input_expand,
                cnn.dropout_keep_prob: config.dropout_keep_prob
            }
            _, step, summaries, loss, probabilities, accuracy = sess.run(
                [train_op, global_step, train_summary_op,
                    cnn.loss, cnn.probabilities, cnn.accuracy],
                feed_dict)
            # _, step, loss, probabilities, accuracy = sess.run(
            #     [train_op, global_step, cnn.loss, cnn.probabilities, cnn.accuracy],
            #     feed_dict)
            time_str = datetime.datetime.now().isoformat()

            # predict_label_and_marked_label_list = []
            # for i in range(len(x_batch)):
            #     predict_label = get_label_using_probs(probabilities[i], top_number=config.top_number)
            #     predict_label_list = list(predict_label)
            #     marked_label = np.where(y_batch[i] == 1)[0]
            #     marked_label_list = list(marked_label)
            #     predict_label_and_marked_label_list.append((predict_label_list, marked_label_list))

            # f1 = evaluate(predict_label_and_marked_label_list)
            # print("{}: step {}, loss {:g}, F1 {:g}".format(time_str, step, loss, f1))
            if train_summary_writer:
                train_summary_writer.add_summary(summaries, step)
                train_summary_writer.flush()
            print("{} : step {}, train loss {:g}, accuracy {:g}".format(
                time_str, step, loss, accuracy))

        def dev_step(_y_batch, embedded_chars, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_y: _y_batch,
                cnn.embedded_chars: embedded_chars,
                cnn.dropout_keep_prob: 1.0
            }

            step, summaries, loss, probabilities, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss,
                    cnn.probabilities, cnn.accuracy],
                feed_dict)

            # step, loss, probabilities = sess.run(
            #     [global_step, cnn.loss, cnn.probabilities],
            #     feed_dict)
            time_str = datetime.datetime.now().isoformat()

            # predict_label_and_marked_label_list = []
            # for i in range(len(x_batch)):
            #     predict_label = get_label_using_probs(probabilities[i], top_number=config.top_number)
            #     predict_label_list = list(predict_label)
            #     marked_label = np.where(y_batch[i] == 1)[0]
            #     marked_label_list = list(marked_label)
            #     predict_label_and_marked_label_list.append((predict_label_list, marked_label_list))
            #
            # f1 = evaluate(predict_label_and_marked_label_list)

            # print("{}: step {}, loss {:g}, F1 {:g}".format(time_str, step, loss, f1))

            if writer:
                writer.add_summary(summaries, step)
                writer.flush()

            print("{} : step {}, dev loss {:g}, accuracy {:g}".format(
                time_str, step, loss, accuracy))

        # Generate batches
        # if expend:
        #     batches = data_helpers.batch_iter(
        #         list(zip(x_train, y_train, tag_train)), config.batch_size_train, config.num_epoch)
        # else:
        #     batches = data_helpers.batch_iter(
        #         list(zip(x_trin, y_train)), config.batch_size_train, config.num_epoch)

        expend_input_dev = None
        if len(dev_set) > 0:
            x_dev = list(dev_set[0])
            y_dev = list(dev_set[1])
            embedded_chars_dev = config.wv.embedding_lookup(
                len(x_dev), config.sentence_words_num, config.embedding_dim_cn_train, x_dev, 0)

            # embedded_chars_dev = np.array(embedded_chars_dev)

            if expend:
                tag_dev = list(dev_set[2])
                tag_dev = np.array(tag_dev)
                expend_input_dev = np.concatenate(
                    (embedded_chars_dev, tag_dev), axis=2)
            else:
                expend_input_dev = embedded_chars_dev

        # Training loop. For each batch...
        print('Generate batches')
        batches = data_helpers.batch_iter(
            train_data, config.batch_size_train, config.num_epoch)
        try:
            for batch in batches:
                if expend:
                    x_batch, y_batch, _, tag_batch = zip(*batch)
                    train_step(x_batch, y_batch, tag_batch)
                else:
                    x_batch, y_batch, _ = zip(*batch)
                    train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % config.evaluate_every == 0 and (expend_input_dev is not None):
                    print("\nEvaluation:")
                    dev_step(y_dev, expend_input_dev,
                             writer=dev_summary_writer)
                    print("")
                if current_step % config.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix,
                                      global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
        except KeyboardInterrupt:
            pass


def replace_model_path(checkpoint_path):
    pat = re.compile(r'\"(/.+/)model-\d+\"$')
    model = ''.join(pat.findall(checkpoint_path))
    text = re.sub(model, '', checkpoint_path)
    return text


def remove_old_file(dir_name):
    lists = os.listdir(dir_name)  # 列出目录的下所有文件和文件夹保存到lists
    # print(lists)
    lists.sort(key=lambda fn: os.path.getmtime(dir_name + "/" + fn))  # 按时间排序
    file_new = os.path.join(dir_name, lists[-1])  # 获取最新的文件保存到file_new
    # print(file_new)

    paths = [os.path.join(dir_name, file_name) for file_name in lists]

    for p in paths:
        if p == file_new:
            pass
        else:
            os.remove(p)


# 修改checkpoint文件中的model路径
lines = []
with open(os.path.join(checkpoint_dir, "checkpoint"), "r") as f:
    f_lines = list(f.readlines())
    for i in range(len(f_lines)):
        if i in [0, len(f_lines)-1]:
            line_deal = replace_model_path(f_lines[i])
            lines.append(line_deal)

with open(os.path.join(checkpoint_dir, "checkpoint"), "w") as f:
    for line in lines:
        f.write(line)

# 删除旧的summary文件
remove_old_file(train_summary_dir)
remove_old_file(dev_summary_dir)

print("the train is finished")
toc = int(time.time())
print("training takes {} seconds already\n".format(toc-tic))
print("program end!")
