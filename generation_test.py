#coding:utf-8
import tensorflow as tf
import sys,time
import numpy as np
import pickle as cPickle 
import os
import random
import Config
from SC_LSTM_Model import SC_LSTM
from SC_LSTM_Model import SC_MultiRNNCell
from SC_LSTM_Model import SC_DropoutWrapper
import argparse

test_word = [u'afternoon', u'usually_acceptable',u'shouting',u'YH_Basketball']

def get_arguments():
    parser = argparse.ArgumentParser(description='model_path')
    parser.add_argument('--mod_path',type=str)
    return parser.parse_args()

class Graph(object):
    def __init__(self, is_training, config):
        ####graph config####
        self.batch_size = config.generation_batch_size
        self.num_steps = config.generation_num_steps
        self.hidden_size = config.hidden_size
        self.alpha = tf.constant(0.5)
        self.vocab_size = config.vocab_size
        self.key_words_voc_size = config.key_words_voc_size
    
        ####graph input####
        self.sentence = tf.placeholder(tf.int32,[self.batch_size, self.num_steps],name = 'sentence')
        self.sentence_no_start = tf.placeholder(tf.int32, [self.batch_size, self.num_steps],name = 'sentence_no_start')
        self.keywords = tf.placeholder(tf.float32, [self.batch_size, self.key_words_voc_size],name = 'keywords')
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.num_steps],name = 'mask')
        
        ####call model####
        LSTM_cell = SC_LSTM(self.key_words_voc_size, self.hidden_size, forget_bias=0.0, state_is_tuple=False)
        if is_training and config.keep_prob < 1:
            LSTM_cell = SC_DropoutWrapper(
                LSTM_cell, output_keep_prob=config.keep_prob)
        cell = SC_MultiRNNCell([LSTM_cell] * config.num_layers, state_is_tuple=False)
        self._initial_state = cell.zero_state(self.batch_size, tf.float32)
        self._init_output = tf.zeros([self.batch_size, self.hidden_size*config.num_layers], tf.float32)
        with tf.device("/cpu:0"):
            embedding = tf.get_variable('word_embedding', [self.vocab_size, config.word_embedding_size], trainable=True)
            inputs = tf.nn.embedding_lookup(embedding, self.sentence)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        sc_vec = self.keywords
        outputs = []
        output_state = self._init_output
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                with tf.variable_scope("RNN_sentence"):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    sc_wr = tf.get_variable('sc_wr',[config.word_embedding_size, self.key_words_voc_size])
                    res_wr = tf.matmul(inputs[:, time_step, :], sc_wr)
                    res_hr = tf.zeros_like(res_wr, dtype = tf.float32)
                    for layer_id in range(config.num_layers):
                        sc_hr = tf.get_variable('sc_hr_%d'%layer_id,[self.hidden_size, self.key_words_voc_size])
                        res_hr += self.alpha * tf.matmul(tf.slice(output_state, [0, self.hidden_size*layer_id], [-1, self.hidden_size]), sc_hr)
                    r_t = tf.sigmoid(res_wr + res_hr)
                    sc_vec = r_t * sc_vec
                    (cell_output, state, cell_outputs) = cell(inputs[:, time_step, :], state, sc_vec)
                    outputs.append(cell_outputs)
                    output_state = cell_outputs
            self._sc_vec = sc_vec
            self._end_output = output_state
        try:
            output = tf.reshape(tf.concat(1, outputs), [-1, self.hidden_size*config.num_layers])
        except:
            output = tf.reshape(tf.concat(outputs, 1), [-1, self.hidden_size*config.num_layers])
        softmax_w = tf.get_variable("softmax_w", [self.hidden_size*config.num_layers, self.vocab_size])
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        self._final_state = state
        self._prob = tf.nn.softmax(logits)

        return

def forward_prop(session, graph, config, state=None, keywords=None, sentence=None,flag =1, last_output=None):
    ####forward propagation####
    sentence = sentence.reshape((1,1))
    if flag == 0:
        prob, _state, _last_output, _sc_vec = session.run([graph._prob, graph._final_state, graph._end_output, graph._sc_vec], feed_dict = {graph.keywords: keywords,graph.sentence:sentence})
    else:
        prob, _state, _last_output, _sc_vec = session.run([graph._prob, graph._final_state, graph._end_output, graph._sc_vec],{graph.keywords: keywords, graph.sentence:sentence,graph._initial_state: state,graph._init_output: last_output})
    return prob, _state, _last_output, _sc_vec
    
def main(_):    
    ####set up model resotring path and config####
    generated_model_path = get_arguments().mod_path 
    gen_config = Config.Config()
    beam_size = gen_config.BeamSize

    ####load files####
    keyword_file = cPickle.load(open('keyword.pkl','rb'))
    gen_config.key_words_voc_size = len(keyword_file)
    embedded_vec = cPickle.load(open('parallel_vec.pkl', 'rb'))
    embedded_word = cPickle.load(open('parallel_word.pkl','rb'))
    gen_config.vocab_size = len(embedded_word)
    
    word_to_idx = { ch:i for i,ch in enumerate(embedded_word) }
    idx_to_word = { i:ch for i,ch in enumerate(embedded_word) }
    keyword_to_idx = { ch:i for i,ch in enumerate(keyword_file) }
    
    ####set up session config####
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True

    ####restore session####
    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        
        ##initialize and restore##
        initializer = tf.random_uniform_initializer(-gen_config.init_scale,gen_config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            graph = Graph(is_training=False, config=gen_config)
        tf.initialize_all_variables().run()
        model_saver = tf.train.Saver(tf.all_variables())
        print ('model loading ...')
        model_saver.restore(session,generated_model_path)
        print ('Done!')

        ##input keyword and 'START'##
        start_word = (0.0, [idx_to_word[1]]) #probability, [word]
        keyword_count = np.zeros(gen_config.key_words_voc_size)
        for word in test_word:
            keyword_count[keyword_to_idx[word]] = 1.0
        keyword_input = np.array([keyword_count], dtype=np.float32)
        sentence = np.int32([1])  #'START'

        ##forward prop##
        prob, _state, _last_output, _sc_vec = forward_prop(session, graph, gen_config, keywords=keyword_input, sentence=sentence, flag=0)#prob shape (1,no of words)

        ##beam search##       
        prob_enlarged = np.log(1e-20 + prob.reshape(-1))
        top_indices = np.random.choice(gen_config.vocab_size, beam_size, replace=False, p=prob.reshape(-1))#select top (beam_size) words candidate
        candidates_final = []
        for i in range(beam_size): #candidate append seq: probability,word_seq,probable_word_idx, _state, _last_output, _sc_vec
            probable_word_idx = top_indices[i]
            probability = start_word[0] + prob_enlarged[probable_word_idx]
            word_seq = start_word[1] + [idx_to_word[probable_word_idx]]
            candidates_final.append((probability,word_seq,probable_word_idx, _state, _last_output, _sc_vec))
        candidates_final.sort(key = lambda x:x[0], reverse = True) # sort by probility in decreasing order
        for i in range(gen_config.len_of_generation-1):
            candidates_tmp = []
            for properties in candidates_final:
                sentence = np.int32(properties[2])
                prob, _state, _last_output, _sc_vec = forward_prop(session, graph,gen_config, keywords=properties[5],sentence = sentence,state = properties[3],last_output=properties[4],  flag=1)
                prob_enlarged = np.log(1e-20 + prob.reshape(-1))
                top_indices = np.random.choice(gen_config.vocab_size, beam_size, replace=False, p=prob.reshape(-1))
                for j in range(beam_size):
                    probable_word_idx = top_indices[j]
                    probability = properties[0] + prob_enlarged[probable_word_idx]
                    word_seq = properties[1] + [idx_to_word[probable_word_idx]]
                    candidates_tmp.append((probability,word_seq,probable_word_idx, _state, _last_output, _sc_vec))
            candidates_tmp.sort(key = lambda x:x[0], reverse = True) # sort by probility in decreasing order                
            candidates_final = candidates_tmp[:beam_size] # truncate to avoid exponential increase

        print (' '.join(candidates_final[0][1][1:]).encode('utf-8'))
            
if __name__ == "__main__":
    tf.app.run()
