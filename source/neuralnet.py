import os
import numpy as np
import tensorflow as tf
import source.layers as lay

class CNN(object):

    def __init__(self, height, width, channel, num_class, ksize, learning_rate=1e-3, embedding_dim=128, ckpt_dir='./Checkpoint'):

        print("\nInitializing Short-ResNet...")
        self.height, self.width, self.channel, self.num_class = height, width, channel, num_class
        self.ksize, self.learning_rate = ksize, learning_rate
        self.embedding_dim = num_class
        self.ckpt_dir = ckpt_dir

        self.customlayers = lay.Layers()
        self.model(tf.zeros([1, self.height, self.width, self.channel]), verbose=True)

        self.optimizer = tf.optimizers.Adam(self.learning_rate)

        self.summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

    def step(self, x, y, iteration=0, train=False):

        with tf.GradientTape() as tape:
            if(train): logits = self.arcface(self.model(x, verbose=False), y)
            else: logits = self.arcface(self.model(x, verbose=False), np.ones_like(y))
            smce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.math.reduce_mean(smce)

        score = self.customlayers.softmax(logits)
        pred = tf.argmax(score, 1)
        correct_pred = tf.equal(pred, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        if(train):
            gradients = tape.gradient(loss, self.customlayers.params_trainable)
            self.optimizer.apply_gradients(zip(gradients, self.customlayers.params_trainable))

            with self.summary_writer.as_default():
                tf.summary.scalar('ArcFace/loss', loss, step=iteration)
                tf.summary.scalar('ArcFace/accuracy', accuracy, step=iteration)

        return loss, accuracy, score

    def save_params(self):

        vars_to_save = {}
        for idx, name in enumerate(self.customlayers.name_bank):
            vars_to_save[self.customlayers.name_bank[idx]] = self.customlayers.params_trainable[idx]
        vars_to_save["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=self.ckpt_dir, max_to_keep=3)
        ckptman.save()

    def load_params(self):

        vars_to_load = {}
        for idx, name in enumerate(self.customlayers.name_bank):
            vars_to_load[self.customlayers.name_bank[idx]] = self.customlayers.params_trainable[idx]
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

    def arcface(self, x, gt, m=0.5, s=64):

        """ Algorithm 1 The Pseudo-code of ArcFace on MxNet
        -------------------------------------------------------
        Input: Feature Scale s, Margin Parameter m in Eq. 3, Class Number n, Ground-Truth ID gt.
            1. x = mx.symbol.L2Normalization (x, mode = ’instance’)
            2. W = mx.symbol.L2Normalization (W, mode = ’instance’)
            3. fc7 = mx.sym.FullyConnected (data = x, weight = W, no bias = True, num hidden = n)
            4. original target logit = mx.sym.pick (fc7, gt, axis = 1)
            5. theta = mx.sym.arccos (original target logit)
            6. marginal target logit = mx.sym.cos (theta + m)
            7. one hot = mx.sym.one hot (gt, depth = n, on value = 1.0, off value = 0.0)
            8. fc7 = fc7 + mx.sym.broadcast mul (one hot, mx.sym.expand dims (marginal target logit - original target logit, 1))
            9. fc7=fc7*s
        Output: Class-wise affinity score fc7. """

        embedd_w = self.customlayers.get_weight(vshape=[self.embedding_dim, self.num_class], name="arc", bias=False)

        x = tf.nn.l2_normalize(x, axis=1) # line 1
        W = tf.nn.l2_normalize(embedd_w, axis=0) # line 2
        fc7 = tf.matmul(x, W) # line 3 (N, self.num_class)

        original_target_logit = tf.clip_by_value(tf.math.reduce_sum(fc7 * gt, axis=1), -1+(1e-12), 1-(1e-12)) # line 4 (N, )
        theta = tf.acos(original_target_logit) # line 5 (N, )
        marginal_target_logit = tf.cos(theta + m) # line 6 (N, )

        one_hot = gt # line 7 (N, self.num_class)
        fc7 = fc7 + tf.multiply(one_hot, tf.expand_dims(marginal_target_logit - original_target_logit, axis=1)) # line 8
        fc7 = fc7 * s # line 9

        return fc7

    def model(self, x, verbose=False):

        if(verbose): print("input", x.shape)

        conv1 = self.customlayers.conv2d(x, \
            self.customlayers.get_weight(vshape=[3, 3, self.channel, 16], name="%s" %("conv1")), \
            stride_size=1, padding='SAME')
        conv1_bn = self.customlayers.batch_norm(conv1, name="%s_bn" %("conv1"))
        conv1_act = self.customlayers.elu(conv1_bn)
        conv1_pool = self.customlayers.maxpool(conv1_act, pool_size=2, stride_size=2)

        conv2_1 = self.residual(conv1_pool, \
            ksize=self.ksize, inchannel=16, outchannel=32, name="conv2_1", verbose=verbose)
        conv2_2 = self.residual(conv2_1, \
            ksize=self.ksize, inchannel=32, outchannel=32, name="conv2_2", verbose=verbose)
        conv2_pool = self.customlayers.maxpool(conv2_2, pool_size=2, stride_size=2)

        conv3_1 = self.residual(conv2_pool, \
            ksize=self.ksize, inchannel=32, outchannel=64, name="conv3_1", verbose=verbose)
        conv3_2 = self.residual(conv3_1, \
            ksize=self.ksize, inchannel=64, outchannel=64, name="conv3_2", verbose=verbose)

        [n, h, w, c] = conv3_2.shape
        flat = tf.compat.v1.reshape(conv3_2, shape=[-1, h*w*c], name="flat")
        if(verbose):
            num_param_fe = self.customlayers.num_params
            print("flat", flat.shape)

        fc1 = self.customlayers.fullcon(flat, \
            self.customlayers.get_weight(vshape=[h*w*c, self.embedding_dim], name="fullcon1"))
        if(verbose):
            print("fullcon1", fc1.shape)
            print("\nNum Parameter")
            print("Feature Extractor : %d" %(num_param_fe))
            print("Classifier        : %d" %(self.customlayers.num_params - num_param_fe))
            print("Total             : %d" %(self.customlayers.num_params))

        return fc1

    def residual(self, input, ksize, inchannel, outchannel, name="", verbose=False):

        convtmp_1 = self.customlayers.conv2d(input, \
            self.customlayers.get_weight(vshape=[ksize, ksize, inchannel, outchannel], name="%s_1" %(name)), \
            stride_size=1, padding='SAME')
        convtmp_1bn = self.customlayers.batch_norm(convtmp_1, name="%s_1bn" %(name))
        convtmp_1act = self.customlayers.elu(convtmp_1bn)
        convtmp_2 = self.customlayers.conv2d(convtmp_1act, \
            self.customlayers.get_weight(vshape=[ksize, ksize, outchannel, outchannel], name="%s_2" %(name)), \
            stride_size=1, padding='SAME')
        convtmp_2bn = self.customlayers.batch_norm(convtmp_2, name="%s_2bn" %(name))
        convtmp_2act = self.customlayers.elu(convtmp_2bn)

        if(input.shape[-1] != convtmp_2act.shape[-1]):
            convtmp_sc = self.customlayers.conv2d(input, \
                self.customlayers.get_weight(vshape=[1, 1, inchannel, outchannel], name="%s_sc" %(name)), \
                stride_size=1, padding='SAME')
            convtmp_scbn = self.customlayers.batch_norm(convtmp_sc, name="%s_scbn" %(name))
            convtmp_scact = self.customlayers.elu(convtmp_scbn)
            input = convtmp_scact

        output = input + convtmp_2act

        if(verbose): print(name, output.shape)
        return output
