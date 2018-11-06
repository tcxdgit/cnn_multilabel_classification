import sys
sys.path.append("..")
from sys import argv
import json
# from IMQ import IMessageQueue
from cnn_multilabel_classification.cnnclassify import Classify_CN
from mq_server import MQServer


class ClassifyCNNMQ(MQServer):

    def __init__(self, model_path, publish_key, receive_key):
        # Classify_CN.__init__(self, model_path, user=receive_key.split('.')[-1])
        # self.mq = IMessageQueue(receive_key, publish_key, receive_key, receive_key,  self.getCategory_mq)

        MQServer.__init__(self)

        self.classify = Classify_CN(model_path, user=receive_key.split('.')[-1])
        self.create_connection(receive_key, publish_key, receive_key, receive_key, self.getcategory_mq)

    def getcategory_mq(self, key, sentence, publish_func=''):
        # r = Classify_CN.getCategory(self, sentence)
        # if publish_func and r:
        #     publish_func(json.dumps(r), key.replace('request', 'reply'))
        # else:
        #     print('[ClassifyCNN] no publish')

        r = self.classify.getCategory(sentence)
        if publish_func and r:
            publish_func(json.dumps(r), key.replace('request', 'reply'))
        else:
            print('[ClassifyCNN] no publish')

if __name__ == '__main__':
    script, model_name, model_dir = argv
    bc = ClassifyCNNMQ(model_dir, '', 'nlp.classify.cnn.'+model_name+'.request.#')
