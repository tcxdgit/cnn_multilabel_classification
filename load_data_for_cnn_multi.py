import sys
sys.path.append('..')
from utils.mongodb_client import Mongo
import os
import nlp_property
import copy
import json
from tools.word_cut import WordCutHelper
wh = WordCutHelper(1)
punctuations = ['～',  '！',  '（',  '）',  '{',  '}',  '【',  '】',  '、',  '；',  '：',  '“',  '”',  '‘',  '’',
                '《',  '》',  '，',  '。',  '？',  '~',  '!',  '(',  ')',  '{',  '}',  '[',  ']',  ';',  ':',
                '"',  '"',  '<',  '>',  ',',  '.',  '?', "'"]


def gather_data(am_data, cut_sentence, expand_punct=True):
    labels = set()
    data_train = dict()
    data_test = dict()
    n_train_sample = 0
    n_test_sample = 0
    for data in am_data:
        questions = data['questions']
        # questions_test = data['questions_test']
        questions_test = data.get('questions_test', None)
        intent = data['intent']

        if intent:
            pass
        else:
            intent = 'null'

        entities = data['entities']

        if not entities:
            label = intent
        elif isinstance(entities, list):
            label = intent + '|' + '#'.join(entities)
        else:
            label = intent + '|' + entities

        labels.add(label)

        for question in questions:
            # line = question + '##' + intent
            if cut_sentence:
                value = wh.getTagAndWord(question)['word']
                question = ' '.join(value)
            data_train[question] = {'label': [label]}
            n_train_sample += 1

            if expand_punct:
                expanded_question = copy.deepcopy(question)
                flag = False
                for char in question:
                    if char in punctuations:
                        flag = True
                        expanded_question = expanded_question.replace(char, '')
                if flag:
                    if cut_sentence:
                        value = wh.getTagAndWord(expanded_question)['word']
                        expanded_question = ' '.join(value)
                    data_train[expanded_question] = {'label': [label]}
                    n_train_sample += 1

        if questions_test:
            for question in questions_test:

                if cut_sentence:
                    value = wh.getTagAndWord(question)['word']
                    question = ' '.join(value)

                data_test[question] = {'label': [label]}
                n_test_sample += 1
        else:
            pass

    return data_train, data_test, labels, n_train_sample, n_test_sample


def load_scene_data_cnn(scene_id, cut_sentence=True, load_qa=False, expand_punct=True):
    """

    :param scene_id:场景名称
    :param cut_sentence:预先分词
    :param load_qa:
    :param expand_punct: 通过去标点符号来扩充语料
    :return:
    """
    print('downloading data {} from mongodb...'.format(scene_id))
    mongodb = Mongo(ip=nlp_property.NLP_FRAMEWORK_DATA_IP, db_name=scene_id)
    # mongodb_auto = Mongo(ip=nlp_property.NLP_FRAMEWORK_IP, db_name='automata')
    # automata
    # shangyongjieshao: query={"category": {"$nin":["&", "&1","&2", "&3","&4", "&&"]}}
    query = {}
    if scene_id == 'shangyongjieshao' or scene_id == 'jieshaoshangyong':
        query = {"category": {"$nin": ["&", "&1", "&2", "&3", "&4", "&&"]}}
    am_data = mongodb.search(collection='automata', query=query, field={
        "intent": 1, "entities": 1, "questions": 1, "questions_test": 1})
    frame_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # directory = context.train_data_path_prefix.format(scene_id)
    data_save_dir = os.path.join(frame_dir, 'work_space', scene_id, 'dataset')
    if os.path.exists(data_save_dir):
        pass
    else:
        os.makedirs(data_save_dir)

    # load data
    if cut_sentence:
        print('cut sentences')
    data_train, data_test, labels, n_train, n_test = gather_data(am_data, cut_sentence, expand_punct=expand_punct)

    if load_qa:
        q_data = mongodb.search(collection='qa', field={"equal_questions": 1})
        if len(q_data) > 0:
            labels.add('qa')
            # write_qa(data_save_dir + "/questions_train", q_data)
            for data in q_data:
                questions = data['equal_questions']
                for question in questions:

                    if cut_sentence:
                        value = wh.getTagAndWord(question)['word']
                        question = ' '.join(value)

                    data_train[question] = {'label': ['qa']}
                    n_train += 1

    with open(os.path.join(data_save_dir, 'questions_train.json'), 'w') as f_train:
        # for data in data_train:
        #     f_train.write(json.dumps(data, ensure_ascii=False))
        json.dump(data_train, f_train, ensure_ascii=False)

    with open(os.path.join(data_save_dir, 'questions_test.json'), 'w') as f_test:
        json.dump(data_test, f_test, ensure_ascii=False)

    with open(os.path.join(data_save_dir, 'labels'), 'w') as f:
        for label in labels:
            f.write(label + '\n')

    print('download completed data {} from mongodb...'.format(scene_id))
    print('number of training/test data is: {}/ {} '.format(n_train, n_test))


def load_shared_data(cut_sentence=True, load_qa=False, expand_punct=True):
    print('downloading data of all shared scene from mongodb')
    db_automata = Mongo(ip=nlp_property.NLP_FRAMEWORK_IP, db_name='automata')
    docs = db_automata.search(collection='machines', query={'shared': 'True'})
    frame_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_save_dir = os.path.join(frame_dir, 'work_space/common/dataset')
    if os.path.exists(data_save_dir):
        pass
    else:
        os.makedirs(data_save_dir)

    train_set = dict()
    test_set = dict()
    labels = set()
    n_train = 0
    n_test = 0

    if cut_sentence:
        print('cut sentence')
    for doc in docs:
        scene_id = doc['scene_id']
        mongodb = Mongo(ip=nlp_property.NLP_FRAMEWORK_IP, db_name=scene_id)
        am_data = mongodb.search(collection='automata',
                                 field={"intent": 1, "entities": 1, "questions": 1})
        data_train, data_test, labels_doc, num_train, num_test = gather_data(am_data, cut_sentence, expand_punct=expand_punct)
        train_set.update(data_train)
        test_set.update(data_test)
        labels.update(labels_doc)
        n_train += num_train
        n_test += num_test

        if load_qa:
            q_data = mongodb.search(collection='qa', field={"equal_questions": 1})
            if len(q_data) > 0:
                labels.add('qa')
                # write_qa(data_save_dir + "/questions_train", q_data)
                for data in q_data:
                    questions = data['equal_questions']
                    for question in questions:

                        if cut_sentence:
                            value = wh.getTagAndWord(question)['word']
                            question = ' '.join(value)

                        train_set[question] = {'label': ['qa']}
                        n_train += 1

    with open(os.path.join(data_save_dir, 'questions_train.json'), 'w') as f_train:
        json.dump(train_set, f_train, ensure_ascii=False)

    with open(os.path.join(data_save_dir, 'questions_test.json'), 'w') as f_test:
        json.dump(test_set, f_test, ensure_ascii=False)

    with open(os.path.join(data_save_dir, 'labels'), 'w') as f:
        for label in labels:
            f.write(label + '\n')
    print('Download data of all shared scene from mongodb')
    print('number of training/test data is: {}/ {} '.format(n_train, n_test))


if __name__ == "__main__":
    # 1 下载某一场景的数据
    # bookstore/suzhou_gov_centre
    scene = 'banktest'
    load_scene_data_cnn(scene, cut_sentence=True, load_qa=False, expand_punct=True)

    # 2 下载共享场景数据（所有shared为True场景的数据），对应于common场景
    # load_shared_data(cut_sentence=True)

    print('Done')
