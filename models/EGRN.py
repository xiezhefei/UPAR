import tensorflow as tf
init=tf.glorot_uniform_initializer()
class EGRN(object):
    def __init__(self, H,eH, nNodes):
        self.H = H
        self.eH=eH
        self.nNodes = nNodes
    def egat_head(self, seq, out_sz, H, activation, in_drop=0.0, coef_drop=0.0, residual=False):
        with tf.name_scope('my_attn'):
            if in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - in_drop)
            seq_fts = tf.layers.conv1d(seq, out_sz, 1, kernel_initializer=init, )  # 实现节点的维度变换
            H1 = tf.layers.conv1d(H, self.nNodes, 1, kernel_initializer=init, use_bias=False)
            # H2= tf.layers.conv1d(H, out_sz, 1, kernel_initializer=init,use_bias=False)
            H2 = tf.layers.conv1d(H, out_sz, 1, kernel_initializer=init, use_bias=False)
            # simplest self-attention possible
            f_1 = tf.layers.conv1d(seq_fts, 1, 1, kernel_initializer=init)
            # tf.initializers.glorot_uniform()
            f_2 = tf.layers.conv1d(seq_fts, 1, 1, kernel_initializer=init)
            logits = tf.concat([f_1 + tf.transpose(f_2, [0, 2, 1]),H1],axis=2)
            #logits = f_1 + tf.transpose(f_2, [0, 2, 1]) + H1
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))

            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
            vals = tf.matmul(coefs, seq_fts) + tf.matmul(coefs, H2)

            vals=self.GRU(vals,seq_fts)
            # 矩阵乘法
            ret = tf.layers.conv1d(vals, 1, 1, kernel_initializer=init)

            # residual connection
            if residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
                else:
                    ret = ret + seq

            return activation(ret)  # activation


    def GRU(self, inputs, state):

        gate_inputs = tf.layers.conv1d(tf.concat([inputs, state], 2), batch_size, 1)
        # gate_inputs=tf.matmul(tf.concat([inputs,state],2),gate_kernel)
        # gate_inputs=tf.add(gate_inputs,gate_bias)
        value = tf.sigmoid(gate_inputs)
        r, u = tf.split(value=value, num_or_size_splits=2, axis=2)
        # candidate=tf.add(tf.matmul(inputs,candidate_input_kernel),candidate_input_bias)
        candidate = tf.layers.conv1d(inputs, 32, 1)
        candidate += r * tf.layers.conv1d(state, 32, 1)
        # candidate+=r*tf.add(tf.matmul(state,candidate_hidden_kernel),candidate_hidden_bias)
        candidate = tf.tanh(candidate)
        new_h = (1 - u) * candidate + u * state
        return new_h
    def egrn(self, dH):
        output_dim =5
        n_heads =4
        attns = []
        for _ in range(n_heads):
            ffd_drop = 0
            attn_drop = 0.5
            attns.append(self.egat_head(dH, output_dim, self.eH, tf.nn.leaky_relu, ffd_drop, attn_drop, False))
        # attns.append(self.egat_head(dH, output_dim[0], self.eH, tf.nn.elu, ffd_drop, attn_drop, False))
        h_1 = tf.concat(attns, axis=-1)
        logits = tf.layers.conv1d(h_1, 1, 1, kernel_initializer=init,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        logits = tf.math.minimum(logits, self.Vmax) + tf.math.maximum(logits, 0) - logits
        logits = tf.squeeze(logits)
        return logits