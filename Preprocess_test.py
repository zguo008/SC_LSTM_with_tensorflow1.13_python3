#coding:utf-8
import numpy as np
import tensorflow as tf
import os, collections
import Config
import pickle as cPickle


class analyzer():
    def __init__(self):
        self.config = Config.Config()
        self.config.vocab_size += 4 #pad,start,end,unk

    def read_embedding(self):
        with open(self.config.vec_file, 'r') as fvec:
            words = []
            vecs =[]
            fvec.readline()
            words.append(u'PAD')
            vecs.append([0]*self.config.word_embedding_size)
            words.append(u'START')
            vecs.append([0]*self.config.word_embedding_size)
            words.append(u'END')
            vecs.append([0]*self.config.word_embedding_size)
            words.append(u'UNK')
            vecs.append([0]*self.config.word_embedding_size)
            for line in fvec:
                line = line.split()
                word = line[0]
                vec = [float(i) for i in line[1:]]
                assert len(vec) == self.config.word_embedding_size
                words.append(word)
                vecs.append(vec)
            assert len(words) == self.config.vocab_size
            word_vec = np.array(vecs, dtype=np.float32)    
            cPickle.dump(word_vec, open('parallel_vec.pkl','wb'), protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(words, open('parallel_word.pkl','wb'), protocol=cPickle.HIGHEST_PROTOCOL) 
        return words, word_vec #parallel word(parallel_word.pkl) vector (parallel_vec.pkl)

    def Create_Keyword(self):
        word_bag = []
        with open(os.path.join(self.config.data_dir, 'TrainingData_Keywords_NTU.txt'), 'r') as fr:
            for line in fr:
                kwd = line.split()
                word_bag += kwd
            collection = collections.Counter(word_bag)
            keyword = []
            for word in collection:
                if collection[word] >= self.config.keyword_min_count:
                    keyword.append(word)
            cPickle.dump(keyword, open('keyword.pkl','wb'), protocol=cPickle.HIGHEST_PROTOCOL) 
        return keyword

    def Read_Text(self, user_keywords):
        text_n_keywords = []
        with open(os.path.join(self.config.data_dir, 'TrainingData_text_NTU.txt'),'r') as ftext, open(os.path.join(self.config.data_dir, 'TrainingData_Keywords_NTU.txt'),'r') as fkwd:
            for text, keyword_line in zip(ftext, fkwd):
                text = text
                all_text_words = text.split()
                keyword_line = keyword_line
                keyword = keyword_line.split()
                keywords = [word for word in keyword if word in user_keywords
                text_n_keywords.append((all_text_words, keywords))
        return text_n_keywords

    def idx_to_word(self,embedded_words,keywords):
        word_to_idx = { ch:i for i,ch in enumerate(embedded_words) }
	#WORD TO IDX {'PAD': 0, 'START': 1, 'END': 2, 'UNK': 3, '</s>': 4, 'in': 5, 'to': 6, 'for': 7, 'of': 8, 'the': 9, 'on': 10, 'and': 11
        idx_to_word = { i:ch for i,ch in enumerate(embedded_words) }
        no_of_keyword = len(keywords)
        keyword_to_idx = { ch:i for i,ch in enumerate(keywords) }
        #{'leaks': 0, 'water': 1, 'plant': 2, 'Japan': 3, 'Radioactive': 4, 'crippled': 5,
        return word_to_idx,idx_to_word,keyword_to_idx  

    def tfrecord_preprocess(self,text_n_keywords,embedded_words,keywords,batch_size, num_steps):
        no_of_batch = len(text_n_keywords) // batch_size
        word_to_idx,idx_to_word,keyword_to_idx = self.idx_to_word(embedded_words,keywords)
        for i in range(no_of_batch):
            batch_data = text_n_keywords[i*batch_size:(i+1)*batch_size]
            training_texts = []
            training_keywords = []
            for pair in batch_data:
                training_texts.append(pair[0]) #pair[0]:all words from text
                keyword_count = np.zeros(len(keywords))
                for kwd in pair[1]: #pair[1] is a set of keywords in this sentence
                    keyword_count[keyword_to_idx[kwd]] = 1.0
                training_keywords.append(keyword_count)
            data = np.zeros((len(training_texts), num_steps+1), dtype=np.int64)
            for i,training_text in enumerate(training_texts):
                 word_count = [1] #'START': 1
                 for wd in training_text:
                     if wd in embedded_words:
                         word_count.append(word_to_idx[wd]) #append one word
                     else:
                         word_count.append(3) #'UNK': 3
                         print('{} is not recognized in the embedding'.format(wd))
                 word_count.append(2)        #'END': 2
                 word_count = np.array(word_count, dtype=np.int64)
                 _size = word_count.shape[0]
                 data[i][:_size] = word_count
	    #data : (no of all sentence, length of sentence )  #numerical representation of a sentence
            _keywords = np.array(training_keywords, dtype=np.float32)
            #keyword: (batch size, length of sentence) #which word is keyword
            sentence = data[:, 0:num_steps]
            sentence_no_start = data[:, 1:]
            mask = np.float32(sentence != 0)
            yield (sentence, sentence_no_start, mask,_keywords)

    def tfrecord_writer(self,text_n_keywords,keywords,embedded_words):      
        writer = tf.python_io.TFRecordWriter("sclstm_data")
        preprocessor = self.tfrecord_preprocess(text_n_keywords,embedded_words,keywords,self.config.batch_size, self.config.num_steps)
        counter = -1
        for (x, y, mask, key_words) in preprocessor:
            example = tf.train.Example(features=tf.train.Features(feature={
                            'sentence': tf.train.Feature(int64_list=tf.train.Int64List(value=x.reshape(-1).astype("int64"))),
                            'sentence_no_start': tf.train.Feature(int64_list=tf.train.Int64List(value=y.reshape(-1).astype("int64"))),
                            'mask': tf.train.Feature(float_list=tf.train.FloatList(value=mask.reshape(-1).astype("float"))),
			    'keywords': tf.train.Feature(float_list=tf.train.FloatList(value=key_words.reshape(-1).astype("float"))),       }))
            serialized = example.SerializeToString()
            writer.write(serialized)
            counter+=1
        return counter

if __name__=='__main__':
    analyzer = analyzer()
    embedded_words, _ = analyzer.read_embedding()   
    keywords = analyzer.Create_Keyword()
    text_n_keywords = analyzer.Read_Text(keywords)
    step = analyzer.tfrecord_writer(text_n_keywords,keywords,embedded_words)
    print(step)
    
