import numpy as np
import random
import json
from cnn_multilabel_classification.config import BaseConfig
import copy

config = BaseConfig()
append_tag = config.append_tag


# load chinese data from json file
def load_cn_data_from_json_file(_data_path, word_cut, data_limit):

    q_label_file = _data_path.get('data_path', None)
    labels_file = _data_path.get('labels_path', None)

    # 保存模型的输出对应类别
    _classes = list()
    with open(labels_file, "r") as f:
        for line in f.readlines():
            line_tmp = line.strip()
            _classes.append(line_tmp)
    num_classes = len(_classes)

    with open(q_label_file, "r") as f_json:
        json_data = json.load(f_json)
        if data_limit:
            json_data = dict(data for i, data in enumerate(json_data.items()) if i < data_limit)

    sentences_cut = list()
    outputs = list()
    sources = list()

    if word_cut:
        from tools.word_cut import WordCutHelper
        wh = WordCutHelper(1)
        print('cut sentences')

        sentences_tag = list()
        for sentence, values in json_data.items():
            tag_word = wh.getTagAndWord(sentence)
            if append_tag:
                # process tag
                tag = tag_word['tag'][:config.sentence_words_num]
                tag_tmp = np.zeros((config.sentence_words_num, len(config.tags_table)))
                idx_list = map(lambda x: config.tags_table.index(x), tag)

                for i, idx in enumerate(idx_list):
                    tag_tmp[i, idx] = 1

                sentences_tag.append(tag_tmp)
            else:
                pass

            # process sentence
            value = tag_word['word'][:config.sentence_words_num]
            sentences_cut.append(' '.join(value))

            # process label
            indices = map(lambda x: _classes.index(x), values['label'])
            labels_tmp = np.zeros(num_classes)

            # process source
            sources.append(values.get('source', None))

            for idx in indices:
                labels_tmp[idx] = 1

            outputs.append(labels_tmp)

        if len(sentences_cut) == len(sentences_tag):
            return list(zip(sentences_cut, outputs, sources, sentences_tag)), _classes
        else:
            return list(zip(sentences_cut, outputs, sources)), _classes
    else:
        for sentence, values in json_data.items():

            # process sentence
            sentences_cut.append(sentence.strip())

            # process label
            indices = map(lambda x: _classes.index(x), values['label'])
            labels_tmp = np.zeros(num_classes)

            # process source
            sources.append(values.get('source', None))

            for idx in indices:
                labels_tmp[idx] = 1

            outputs.append(labels_tmp)

        return list(zip(sentences_cut, outputs, sources)), _classes


def load_data(_data_path, valid_portion,
              sort_by_len=False, enhance=True, reverse=False,
              word_cut=False, data_limit=None):

    data_set, _classes = load_cn_data_from_json_file(_data_path, word_cut, data_limit)

    # 数据集扩增(打乱词顺序,增加新样本)
    if enhance:
        # shuffle
        enhanced_data = list()
        for i, data in enumerate(data_set):
            data_new = list(copy.deepcopy(data))
            sentence_cut = data_new[0].split(' ')
            random.shuffle(sentence_cut)
            data_new[0] = ' '.join(sentence_cut)
            enhanced_data.append(tuple(data_new))

        data_set.extend(enhanced_data)

    else:
        pass

    if reverse:
        for data in data_set:
            data[0].reverse()
    else:
        pass

    random.shuffle(data_set)

    n_samples = len(data_set)
    n_train = int(np.round(n_samples * (1.0 - valid_portion)))

    print("Train/Dev split: {:d}/{:d}".format(n_train, (n_samples - n_train)))
    train_set = [data for data in data_set[: n_train]]
    dev_set = [data for data in data_set[n_train:]]

    if sort_by_len:
        sorted_indices = len_argsort(dev_set[0])
        dev_set = [dev_set[i] for i in sorted_indices]

        sorted_indices = len_argsort(train_set[0])
        train_set = [train_set[i] for i in sorted_indices]

    return train_set, dev_set, _classes


def len_argsort(seq):
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a data set.
    """
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = list(range(data_size))
            random.shuffle(shuffle_indices)
            shuffled_data = []
            for shuffle_indice in shuffle_indices:
                shuffled_data.append(data[shuffle_indice])
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    data_path = {'train_data_path': '../work_space/cmrc2018/dataset/out_ann.txt2',
                 'labels_path': '../work_space/cmrc2018/dataset/labels'}
    # data, classes = load_cn_data_from_json_file(data_path, word_cut=True, data_limit=None)
    load_data(data_path, valid_portion=0.2,
              sort_by_len=True, enhance=True, reverse=True,
              word_cut=False, data_limit=100)
    print('====Done====')
