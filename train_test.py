#coding:utf-8
import tensorflow as tf
import sys,time
import numpy as np
import pickle as cPickle
import os
import random
import Config
from datetime import datetime
from SC_LSTM_Model import SC_LSTM
from SC_LSTM_Model import SC_MultiRNNCell
from SC_LSTM_Model import SC_DropoutWrapper
try:
    from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import sequence_loss_by_example
except:
    pass

total_step = 311 #get value from output of Preprocess.py file

class Graph(object):
    def __init__(self, is_training, word_embedding, config, filename):

        ####training config####
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.max_max_epoch,0.8,staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.batch_size = config.batch_size
     
        ####model config####
        self.num_steps = config.num_steps
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.key_words_voc_size = config.key_words_voc_size
        self.alpha = tf.constant(0.5)
        self.filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
        ####data parsing####
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(self.filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                'sentence': tf.FixedLenFeature([self.batch_size*self.num_steps],tf.int64),
                'sentence_no_start': tf.FixedLenFeature([self.batch_size*self.num_steps],tf.int64),
                'mask': tf.FixedLenFeature([self.batch_size*self.num_steps],tf.float32),
                'keywords': tf.FixedLenFeature([self.batch_size*self.key_words_voc_size],tf.float32),
            })

        ####placeholders####
        sentence = tf.cast(features['sentence'], tf.int32)
        self.sentence = tf.reshape(sentence, [self.batch_size, -1])
        sentence_no_start = tf.cast(features['sentence_no_start'], tf.int32)
        self.sentence_no_start = tf.reshape(sentence_no_start, [self.batch_size, -1])
        mask = tf.cast(features['mask'], tf.float32)
        self.mask = tf.reshape(mask, [self.batch_size, -1])
        self.keywords = tf.cast(features['keywords'], tf.float32)
        self.keywords = tf.reshape(self.keywords, [self.batch_size, -1])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable('word_embedding', [self.vocab_size, config.word_embedding_size], trainable=True, initializer=tf.constant_initializer(word_embedding))
            inputs = tf.nn.embedding_lookup(embedding, self.sentence)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)  
      
        ####call SC_LSTM cell####
        LSTM_cell = SC_LSTM(self.key_words_voc_size, self.hidden_size, forget_bias=0.0, state_is_tuple=False)
        if is_training and config.keep_prob < 1:
            LSTM_cell = SC_DropoutWrapper(
                LSTM_cell, output_keep_prob=config.keep_prob)
        cell = SC_MultiRNNCell([LSTM_cell] * config.num_layers, state_is_tuple=False)
        self._initial_state = cell.zero_state(self.batch_size, tf.float32)
        self._init_output = tf.zeros([self.batch_size, self.hidden_size*config.num_layers], tf.float32)
        sc_vec = self.keywords
        outputs = []
        output_state = self._init_output
        state = self._initial_state
        
        ####define model used in graph####
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                with tf.variable_scope("RNN_sentence"):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()

                    keyword_detect_wr = tf.get_variable('keyword_detect_wr',[config.word_embedding_size, self.key_words_voc_size])
                    keyword_detect = tf.matmul(inputs[:, time_step, :], keyword_detect_wr)
                    
                    keyphrase_detect = tf.zeros_like(keyword_detect, dtype = tf.float32)
                    for layer_id in range(config.num_layers):
                        keyphrase_detect_hr = tf.get_variable('keyphrase_detect_hr_%d'%layer_id,[self.hidden_size, self.key_words_voc_size])
                        keyphrase_detect += self.alpha * tf.matmul(tf.slice(output_state, [0, self.hidden_size*layer_id], [-1, self.hidden_size]), keyphrase_detect_hr)
                    r_t = tf.sigmoid(keyword_detect + keyphrase_detect)
                    self.keywords = r_t * self.keywords
                    
                    (cell_output, state, cell_outputs) = cell(inputs[:, time_step, :], state, self.keywords)
                    outputs.append(cell_outputs)
                    output_state = cell_outputs
            
            self._end_output = output_state
            
        try:
            output = tf.reshape(tf.concat(1, outputs), [-1, self.hidden_size*config.num_layers])
        except:
            output = tf.reshape(tf.concat(outputs, 1), [-1, self.hidden_size*config.num_layers])
        ## inference and error calculation ##
        softmax_w = tf.get_variable("softmax_w", [self.hidden_size*config.num_layers, self.vocab_size])
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        try:
            loss = tf.nn.seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(self.sentence_no_start, [-1])],
                [tf.reshape(self.mask, [-1])], average_across_timesteps=False)
        except:
            loss = sequence_loss_by_example(
                [logits],
                [tf.reshape(self.sentence_no_start, [-1])],
                [tf.reshape(self.mask, [-1])], average_across_timesteps=False)
            
        self.cost = tf.reduce_sum(loss) / self.batch_size
        self._final_state = state

        if not is_training:
            prob = tf.nn.softmax(logits)
            return
        trainable_vars = tf.trainable_variables()
    
        ## gradient descent(optimization) ##
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_vars),config.max_grad_norm)
        self._train_op = self.optimizer.apply_gradients(zip(grads, trainable_vars))
    
def main(_):
    ####set up model saving path####
    config = Config.Config()
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    logdir = os.path.join(config.logdir_root, date_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    model_path_save = logdir+config.model_path

    ####load files####
    kwd_voc = cPickle.load(open('keyword.pkl','rb'))
    config.key_words_voc_size = len(kwd_voc)
    word_vec = cPickle.load(open('parallel_vec.pkl', 'rb'))
    vocab = cPickle.load(open('parallel_word.pkl','rb'))
    config.vocab_size = len(vocab)
    
    ####define graph####
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        graph = Graph(is_training=True, word_embedding=word_vec, config=config, filename='sclstm_data')
    
    ####tf saver####
    model_saver = tf.train.Saver(tf.global_variables())

    ####sess config, tf_summary####
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    session = tf.Session(config=config_tf)
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(session.graph)
    cost = tf.summary.scalar(name='loss', tensor=graph.cost)
    summaries = tf.summary.merge([cost])
    with session.as_default():
        tf.global_variables_initializer().run()        
        tf.train.start_queue_runners(sess=session)
        current_step = 0
        for i in range(config.max_max_epoch): #80
            start_time = time.time()
            costs = 0.0
            iters = 0
            
            for step in range(total_step+1): #312
                current_step +=1
                summary,cost, _ = session.run([summaries,graph.cost, graph.train_op])
                writer.add_summary(summary, current_step)
                if np.isnan(cost):
                    print ('cost is nan!!!')
                    exit()
                costs += cost
                iters += graph.num_steps
                if step and step % (total_step // 5) == 0:
                    print("%d-step perplexity: %.3f cost-time: %.2f s" %(step, np.exp(costs / iters),time.time() - start_time))
            start_time = time.time()
            train_perplexity = np.exp(costs / iters)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            if (i+1) % config.save_freq == 0:
                print ('model saving ...')
                model_saver.save(session, model_path_save+'--%d'%(i+1))
                print ('Done!')      
if __name__ == "__main__":
    tf.app.run()
