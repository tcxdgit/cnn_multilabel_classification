import sys
sys.path.append("..")
from sys import argv
# from IMQ import IMessageQueue
import json
import tools.random_helper as random_helper
from utils.MQ_tools import RabbitMQTool
from mq_client import MQClient
from tools.word_cut import WordCutHelper


class Classify_CN(MQClient):

    # def __init__(self, port_num):
    #     print('init cnn classify')
    #     self.url = "http://localhost:"+str(port_num)+"/"

    # def getCategory(self, sentence):
    #    url = self.url+"s/"+quote(sentence.replace(' ','|'))
    #    result = urequest.request_url(url)
    #    if result:
    #        #print(result)
    #        return result
    #    return {}

    def __init__(self, field):
        MQClient.__init__(self)
        self.field = field
        prefix = 'nlp.classify.cnn.'+field+'.request.'
        publish_key = prefix + random_helper.random_string()

        mqTool = RabbitMQTool()
        queue = prefix + '#'
        status = mqTool.get_queue_status(queue)
        if status:
            receive_key = publish_key.replace('request', 'reply')
            # self.mq = IMessageQueue(receive_key, publish_key, receive_key, receive_key, '')
            self.create_connection(
                receive_key, publish_key, receive_key, receive_key, '')
        else:
            print('{} is not found'.format(queue))

    def getCategory(self, sentence):
        if not sentence or not self.mq:
            return {}
        result = self.mq.request_synchronize(sentence)
        if result:
            return json.loads(result)
        return {}


if __name__ == '__main__':
    # script, field = argv
    field = 'banktest'
    cc = Classify_CN(field)
    wh = WordCutHelper(1)
    while 1:
        s = input('input: ')

        # 分词
        value = wh.getTagAndWord(s)['word']
        sentence = ' '.join(value)
        print(sentence)

        result = cc.getCategory(sentence)
        print(result)
