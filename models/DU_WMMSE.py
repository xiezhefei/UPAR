import tensorflow as tf
class DU_WMMSE(object):
    def __init__(self, H, Hsq, var, nNodes,layers,V):
        self.H = H
        self.Hsq = Hsq
        self.var = var
        self.nNodes = nNodes
        self.layers=layers
        self.V=V

    def U_block(self, V):
        # H_ii * v_i
        num = tf.math.multiply(tf.linalg.diag_part(self.H), V)

        # sigma^2 + sum_j( (H_ji)^2 * (v_j)^2 )
        den = tf.reshape(tf.matmul(tf.transpose(self.Hsq, perm=[0,2,1]), tf.reshape(tf.math.square(V), [-1, self.nNodes, 1])), [-1, self.nNodes]) + self.var

        # U = num/den
        return tf.math.divide(num, den)

    def W_block(self, U, V):
        # 1 - u_i * H_ii * v_i
        den = 1. - tf.math.multiply(tf.linalg.diag_part(self.H), tf.math.multiply(U, V))
        # w1 = tf.get_variable(name='w1',
        #                      initializer = tf.constant(1.0, dtype=tf.float64))
        # W = 1/den
        return tf.math.reciprocal(den)

    def V_block(self, U, W, mu):
        # H_ii * u_i * w_i
        num = tf.math.multiply(tf.linalg.diag_part(self.H), tf.math.multiply(U, W))

        # mu + sum_j( (H_ij)^2 * (u_j)^2 *w_j )
        den = tf.math.add(tf.reshape(tf.matmul(self.Hsq, tf.reshape(tf.math.multiply(tf.math.square(U), W), [-1, self.nNodes, 1])), [-1, self.nNodes]), mu)
        # w2 = tf.get_variable(name='w2', shape=(20, 20),
        #                      initializer=init, dtype=tf.float64)
        # V = num/den
        return tf.math.divide(num, den)
    def Predict(self,l):
        V=self.V
        for l in range(self.layers):
            U = self.U_block(V)
            W = self.W_block(U, V)
            mu = tf.get_variable(name='mu', initializer=tf.constant(0., shape=(), dtype=tf.float64))
            V = self.V_block(U, W, mu)
        return V