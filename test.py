# coding:utf-8
import sys
sys.path.append("..")
from cnn_multilabel_classification.cnnclassify import Classify_CN
# import os
# import pandas as pd
# from pprint import pprint
from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
from cnn_multilabel_classification import data_helper
import numpy as np
import math
from sklearn import metrics


def eval(predict_label_and_marked_label_list):
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
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += (right_num / float(sample_num)) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num

    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = (precision * recall) / (precision + recall)

    return f1


def eval_cmrc(module_path, test_data_path):
    classify = Classify_CN(module_path)

    # top_number = 5
    data_set, classes = data_helper.load_cn_data_from_json_file(test_data_path, word_cut=False, data_limit=None)

    y_true = list()
    y_pred = list()

    for i, data in enumerate(data_set):
        sentence_cut = data[0]
        output_true = data[1]
        source = data[2]

        result = classify.getCategory(sentence_cut, top_number=5)
        output_pred = np.zeros(len(classes))
        j = 0
        for j, prob in enumerate(result['top5_probability']):
            if prob < 0.5:
                break

        label = result['top5_value'][:j]
        indices = map(lambda x: classes.index(x), label)
        for index in indices:
            output_pred[index] = 1

        if np.all(output_pred == output_true):
            pass
        else:
            print('=================')
            print(result)
            print(source)
            print('predicted label: {}; true label: {}'.
                  format(label, [classes[idx] for idx, l in enumerate(output_true) if l == 1]))
            print('=================')

        y_true.append(output_true)
        y_pred.append(output_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(accuracy_score(y_true, y_pred))

    precision = metrics.precision_score(y_true, y_pred, average='micro')
    recall = metrics.recall_score(y_true, y_pred, average='micro')
    F1 = metrics.f1_score(y_true, y_pred, average='micro')

    print('precision: {}; recall: {}; F1: {}'.format(precision, recall, F1))


def main(scene_id):
    module_path = '../work_space/{}/module/cnn_multi'.format(scene_id)
    test_data_path = {'data_path': '../work_space/{}/dataset/questions_test.json'.format(scene_id),
                      'labels_path': '../work_space/{}/dataset/labels'.format(scene_id)}
    eval_cmrc(module_path, test_data_path)


if __name__ == '__main__':
    main('banktest')
    print('Done')
