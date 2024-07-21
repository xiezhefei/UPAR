import pdb
import tensorflow as tf
init=tf.glorot_uniform_initializer()
from models.EGRN import EGRN
from models.DU_WMMSE import DU_WMMSE
class UPAR(object):
        # Initialize
        def __init__( self, Pmax=1., var=7e-10, feature_dim=3, batch_size=64, layers=4, learning_rate=1e-3, max_gradient_norm=5.0 ):
            self.Pmax              = tf.cast( Pmax, tf.float64 )
            self.var               = var
            self.feature_dim       = feature_dim
            self.batch_size        = batch_size
            self.layers            = layers
            self.learning_rate     = learning_rate
            self.max_gradient_norm = max_gradient_norm
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.build_model()

        # Build Model
        def build_model(self):
            self.init_placeholders()
            self.build_network()
            self.build_objective()
            
        def init_placeholders(self):
            # CSI [Batch_size X Nodes X Nodes]
            self.H = tf.compat.v1.placeholder(tf.float64, shape=[64,20,20], name="H")
        # Building network
        def build_network(self):
            # Squared H 
            self.Hsq = tf.math.square(self.H)
            # Diag H
            dH =  tf.linalg.diag_part( self.H ) 
            self.dH = tf.matrix_diag( dH )#对角线元素
            self.eH=self.H-self.dH
            # Retrieve number of nodes for initializing V
            self.nNodes = tf.shape( self.H )[-1]
            # Maximum V = sqrt(Pmax)
            Vmax = tf.math.sqrt(self.Pmax)
            GNN=EGRN(self.H,self.eH,self.nNodes)
            F=GNN.egrn(self.dH)
            self.pow_alloc = []
            model=DU_WMMSE(self.H, self.Hsq, self.var, self.nNodes,self.layers,F)
            V=model.Predict(self.layers)
            V = tf.math.minimum(V, Vmax) + tf.math.maximum(V, 0) - V
            # Final V
            self.pow_alloc = V
                                                                                
        def build_objective(self):
            # (H_ii)^2 * (v_i)^2
            num = tf.math.multiply( tf.matrix_diag_part(self.Hsq), tf.math.square( self.pow_alloc ) )
            
            # sigma^2 + sum_j j ~= i ( (H_ji)^2 * (v_j)^2 ) 
            den = tf.reshape( tf.matmul( tf.transpose( self.Hsq, perm=[0,2,1] ), tf.reshape( tf.math.square( self.pow_alloc ), [-1, self.nNodes, 1] ) ), [-1, self.nNodes] ) + self.var - num 
            
            # rate
            rate = tf.math.log( 1. + tf.math.divide( num, den ) ) / tf.cast( tf.math.log( 2.0 ), tf.float64 )
            
            # Sum Rate = sum_i ( log(1 + SINR) )
            self.utility = tf.reduce_sum( rate, axis=1 )
            # Weighted Sum Rate
            #rate = tf.math.multiply( self.alpha, rate )
            #self.utility = tf.reduce_sum( rate, axis=1 )
            # Minimization objective
            self.obj = -tf.reduce_mean( self.utility )
            self.init_optimizer()

        def init_optimizer(self):
            # Gradients and SGD update operation for training the model
            self.trainable_params = tf.compat.v1.trainable_variables()
            
            #Learning Rate Decay
            #starter_learning_rate = self.learning_rate
            #self.learning_rate_decayed = tf.train.exponential_decay(starter_learning_rate, global_step=self.global_step, decay_steps=5000, decay_rate=0.99, staircase=True)
            
            # SGD with Momentum
            #self.opt = tf.train.GradientDescentOptimizer( learning_rate=learning_rate )
            #self.opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_decayed, momentum=0.9, use_nesterov=True )

            # Adam Optimizer
            self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

            # Compute gradients of loss w.r.t. all trainable variables
            gradients = tf.gradients(self.obj, self.trainable_params)

            # Clip gradients by a given maximum_gradient_norm
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            
            # Update the model
            self.updates = self.opt.apply_gradients(
                zip(clip_gradients, self.trainable_params), global_step=self.global_step)
                
        def save(self, sess, path, var_list=None, global_step=None):
            saver = tf.compat.v1.train.Saver(var_list)
            save_path = saver.save(sess, save_path=path, global_step=global_step)

        def restore(self, sess, path, var_list=None):
            saver = tf.compat.v1.train.Saver(var_list)
            saver.restore(sess, save_path=tf.train.latest_checkpoint(path))

        def train(self, sess, inputs ):
            input_feed = dict()
            input_feed[self.H.name] = inputs
            #input_feed[self.x.name] = features
            #input_feed[self.alpha.name] = alpha
            
            # Training Phase
            #input_feed[self.phase.name] = True
 
            output_feed = [self.obj, self.utility, self.pow_alloc, self.updates]
                            
            outputs = sess.run(output_feed, input_feed)
            
            return outputs[0], outputs[1], outputs[2]


        def eval(self, sess, inputs ):
            input_feed = dict()
            input_feed[self.H.name] = inputs
            #input_feed[self.x.name] = features
            #input_feed[self.alpha.name] = alpha

            # Training Phase
            #input_feed[self.phase.name] = False

            output_feed = [self.obj,self.utility, self.pow_alloc] 
                           
            outputs = sess.run(output_feed, input_feed)
            
            return outputs[0], outputs[1], outputs[2]
