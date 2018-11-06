import os
# 切换词向量请求方法，改embedding.embedding_helper   本地/服务器
from embedding.embedding_helper import EmbeddingHelper


class BaseConfig:
    def __init__(self):

        self.batch_size = None
        # Number of filters per filter size (default: 128)
        self.num_filters = 32
        self.sentence_words_num = 25
        self.embedding_dim_cn_train = 300  # Dimensionality of character embedding

        # Misc Parameters
        self.allow_soft_placement = True  # Allow device soft device placement
        self.log_device_placement = False  # Log placement of ops on devices

        # 句子逆序输入
        self.data_reverse = False

        # 切换词向量     本地/服务器
        try:
            # 请求服务器词向量
            self.wv = EmbeddingHelper('vector_hlt')
        except FileNotFoundError:
            # 请求本地词向量
            print("Ask for local word2vec")
            self.wv = EmbeddingHelper(
                '../work_space/vector/vector_hlt/dic_18_hlt.bin')

        # 扩展特征维数，加词性
        self.append_tag = False
        if self.append_tag:
            self.tags_table = ['AD', 'AS', 'BA', 'CC', 'CD', 'CS', 'DEC', 'DEG', 'DER', 'DEV', 'DT', 'ETC', 'FW', 'IJ',
                               'JJ', 'LB', 'LC', 'M', 'MSP', 'NN', 'NR', 'NT', 'OD', 'ON', 'P', 'PN', 'PU', 'SB', 'SP',
                               'VA', 'VC', 'VE', 'VV']

        # attention_cnn
        self.use_attention = False
        self.attention_dim = 100


# config for train
class TrainConfig(BaseConfig):
    def __init__(self, field):
        BaseConfig.__init__(self)

        self.frame_path = os.path.dirname(os.path.dirname(__file__))
        self.dataset_dir = os.path.join(
            self.frame_path, 'work_space', field, 'dataset')
        data_path = os.path.join(self.dataset_dir, 'questions_train.json')
        # data_path = '/opt/qa_reader/data/questype/train'
        labels_path = os.path.join(self.dataset_dir, 'labels')
        # labels_path = '/opt/qa_reader/data/questype/label'
        self.train_data_path = {'data_path': data_path,
                                'labels_path': labels_path}

        self.out_dir = os.path.join(
            self.frame_path, 'work_space', field, 'module/cnn_multi')

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Model Hyper-parameters
        self.dropout_keep_prob = 0.5
        self.l2_reg_lambda = 0.0001
        self.batch_size_train = 8
        self.num_epoch = 300  # num of epoch
        # Evaluate model on dev set after this many steps (default: 100)
        self.evaluate_every = 100
        # Save model after this many steps (default: 100)
        self.checkpoint_every = 100
        self.num_checkpoints = 5  # Number of checkpoints to store (default: 5)
        # Percentage of the training data to use for validation
        self.dev_sample_percentage = 0.05
        # Comma-separated filter sizes (default: '2,3,4,5')
        self.filter_sizes = "2,3,4,5"
        self.top_number = 5   # Number of the classes to predict

        # process data
        self.data_enhance = False
        self.sort_by_len = False
