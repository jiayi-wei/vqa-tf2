import tensorflowr as tf


class att_Module(tf.keras.Model):
    def __init__(self, units, attention_dim):
        super(att_Module, self).__init__()
        self.W_I = tf.keras.layers.Dense(attention_dim, use_bias=False)
        self.W_Q = tf.keras.layers.Dense(attention_dim)
        self.W_p = tf.keras.layers.Dense(1)

    def call(self, v_I, v_Q):
        # v_I shape (B, dim*dim, units)
        # v_Q shape (B, units)
        v_I_att = self.W_I(v_I)
        v_Q_att = self.W_Q(v_Q)

        v_Q_att = tf.expand_dims(v_Q_att, axis=1)
        # expand v_Q_att to shape (B, 1, units)
        h_A = tf.nn.tanh(v_I_att + v_Q_att)
        # h_A shape (B, dim*dim, units)

        p_I = self.W_p(h_A)
        # p_I shape (B, dim*dim, 1)
        p_I = tf.reshape(p_I, (p_I.shape[0], -1))
        # p_I shape (B, dim*dim)
        p_I_output = tf.nn.softmax(p_I)
        # p_I_output shape (B, dim*dim),
        # the reduce_sum on axis=1 is 1.0 (probability)
        p_I = tf.expand_dims(p_I_output, axis=-1)
        # p_I shape (B, dim*dim, 1)

        v_att = p_I * v_I
        # v_att is the weighted v_I
        # v_att shape (B, dim*dim, units)
        v_att = tf.reduce_sum(v_att, axis=1)
        # v_att is summed on dim*dim column
        # v_att shape (B, units)

        u = v_att + v_Q
        # u is the new feature for next level
        # u shape (B, units)

        return p_I_output, u


class SAN_LSTM(tf.keras.Model):
    def __init__(self, embedding_dim,
                 units,
                 vocab_size,
                 num_answer,
                 dim_att):
        # units is the hidden_units#
        # vocab_size
        # num_answer
        # dim_att is the dimension of attention module weight
        super(SAN_LSTM, self).__init__()
        self.img_fc = tf.keras.layers.Dense(units)

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(units,
                                         recurrent_initializer='glorot_uniform')

        self.att1 = att_Module(units, dim_att)
        self.att2 = att_Module(units, dim_att)

        self.ans_fc = tf.keras.layers.Dense(num_answer)

    def imgModel(self, img):
        # img feature encode for attention
        img = tf.nn.tanh(self.img_fc(img))
        # img shape (B, dim*dim, model_output_dim)
        # -> (B, dim*dim, units)
        # because the weighted img feature will
        # be added with que feature
        return img

    def queModel(self, q):
        # que feature encode for attention
        ques_enc = self.embedding(q)
        # q shape (B, sentence_len)
        # ques_enc shape (B, embedding_dim)
        hidden = self.lstm(ques_enc)
        # hidden shape (B, units)
        return hidden

    def attModel(self, v_img, v_que):
        img_weights_1, u1 = self.att1(v_img, v_que)
        img_weights_2, u2 = self.att2(v_img, u1)
        # u1, u2 are combined feature from 2 levels
        # two weights
        return img_weights_1, img_weights_2, u2

    def call(self, q, img):
        v_img = self.imgModel(img)
        v_que = self.queModel(q)
        img_weights_1, img_weights_2, u = self.attModel(v_img, v_que)
        output = tf.nn.softmax(self.ans_fc(u))
        return output, img_weights_1, img_weights_2
