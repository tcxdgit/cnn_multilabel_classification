import tensorflow as tf


class TextCNN_PreTrained:
    """
    A CNN for text classification.
    Uses an embedding layer,
    followed by an attention, convolution, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0, attention_dim=100, use_attention=True):

        print('TextCNN_PreTrained init\n')
        # Placeholders for input, output and dropout
        self.embedded_chars = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="embedded_chars")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #     self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        if use_attention:

            self.attention_hidden_dim = attention_dim
            # Wa = [attention_W  attention_U]
            self.attention_W = tf.Variable(
                tf.random_uniform([self.embedding_size, self.attention_hidden_dim], 0.0, 1.0),
                name="attention_W")
            self.attention_U = tf.Variable(
                tf.random_uniform([self.embedding_size, self.attention_hidden_dim], 0.0, 1.0),
                name="attention_U")
            self.attention_V = tf.Variable(tf.random_uniform([self.attention_hidden_dim, 1], 0.0, 1.0),
                                           name="attention_V")
            # attention layer before convolution
            self.output_att = list()
            with tf.name_scope("attention"):
                input_att = tf.split(self.embedded_chars, self.sequence_length, axis=1)
                for index, x_i in enumerate(input_att):
                    x_i = tf.reshape(x_i, [-1, self.embedding_size])
                    c_i = self.attention(x_i, input_att, index)
                    inp = tf.concat([x_i, c_i], axis=1)
                    self.output_att.append(inp)

                input_conv = tf.reshape(tf.concat(self.output_att, axis=1),
                                        [-1, self.sequence_length, self.embedding_size*2],
                                        name="input_convolution")
            self.input_conv_expanded = tf.expand_dims(input_conv, -1)
        else:
            self.input_conv_expanded = tf.expand_dims(self.embedded_chars, -1)

        dim_input_conv = self.input_conv_expanded.shape[-2].value

        # Create a convolution + max pool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, dim_input_conv, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.input_conv_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="convolution")
                # Apply non linearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="RelU")
                # Max pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # self.features = tf.reshape(self.h_pool_flat, [len(filter_sizes), num_filters], name="features")

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # self.probability = tf.nn.softmax(self.scores, name="probability")
            self.probabilities = tf.nn.sigmoid(self.scores, name="probabilities")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def attention(self, x_i, x, index):
        """
        Attention model for Neural Machine Translation
        :param x_i: the embedded input at time i
        :param x: the embedded input of all times(x_j of attentions)
        :param index: step of time
        """

        e_i = []
        c_i = []
        for output in x:
            output = tf.reshape(output, [-1, self.embedding_size])
            atten_hidden = tf.tanh(tf.add(tf.matmul(x_i, self.attention_W), tf.matmul(output, self.attention_U)))
            e_i_j = tf.matmul(atten_hidden, self.attention_V)
            e_i.append(e_i_j)
        e_i = tf.concat(e_i, axis=1)
        # e_i = tf.exp(e_i)
        alpha_i = tf.nn.softmax(e_i)
        alpha_i = tf.split(alpha_i, self.sequence_length, 1)

        # i!=j
        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = tf.reshape(output, [-1, self.embedding_size])
                c_i_j = tf.multiply(alpha_i_j, output)
                c_i.append(c_i_j)
        c_i = tf.reshape(tf.concat(c_i, axis=1), [-1, self.sequence_length-1, self.embedding_size])
        c_i = tf.reduce_sum(c_i, 1)
        return c_i


if __name__ == "__main__":
    cnn_model = TextCNN_PreTrained(
      sequence_length=25, num_classes=6,
      embedding_size=300, filter_sizes=[2, 3, 4, 5], num_filters=300, attention_dim=100, l2_reg_lambda=0.0)
    print(cnn_model.input_conv_expanded)
