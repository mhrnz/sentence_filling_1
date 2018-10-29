
# coding: utf-8

# In[52]:


import numpy as np
import pickle
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers.core import Densez


# In[51]:


class sentence_filling_model():
    def __init__(self,batch_size,hidden_size,num_layers,keep_prob,vocab_size,embedding_dim):
        self.batch_size=batch_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.vocab_size=vocab_size
        self.keep_prob=keep_prob
        self.embedding_dim=embedding_dim
        self.encoder_left_inputs=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,None],name='encoder_left_inputs')
        self.encoder_right_inputs=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,None],name='encoder_right_inputs')
        self.encoder_left_seqlen=tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name='encoder_left_seqlen')
        self.encoder_right_seqlen=tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name='encoder_right_seqlen')
        self.decoder_inputs=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,None],name='decoder_inputs')
        self.decoder_seqlen=tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name='decoder_seqlen')
        self.glove_weights=tf.Variable(tf.constant(0.0,shape=[vocab_size,embedding_dim]),trainable=False,name='embedding_weights')
        self.embedding_placeholder=tf.placeholder(tf.float32,shape=[vocab_size,embedding_dim],name='embedding_placeholder')
        self.embedding_init=self.glove_weights.assign(self.embedding_placeholder)
        self.blank=tf.placeholder(tf.float32,shape=[1,embedding_dim],name='blank')
        self.embedding_weights=tf.concat([self.glove_weights,self.blank],axis=0)
        
        self.build_encoder()
        self.build_decoder()
        self.calculate_loss()
        self.optimize()
        
        
    def make_cell(self):
        cell=rnn.BasicLSTMCell(self.hidden_size)
        cell=rnn.DropoutWrapper(cell,output_keep_prob=self.keep_prob)
        return cell
    
    def build_encoder(self):
        input_left_embeds=tf.nn.embedding_lookup(self.embedding_weights,self.encoder_left_inputs)
        input_right_embeds=tf.nn.embedding_lookup(self.embedding_weights,self.encoder_right_inputs)
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            encoder_cell=rnn.MultiRNNCell(cells=[self.make_cell() for _ in range(self.num_layers)])
            self.encoder_left_outputs,encoder_left_laststate=tf.nn.dynamic_rnn(cell=encoder_cell,dtype=tf.float32,
                                                                               sequence_length=self.encoder_left_seqlen,
                                                                               inputs=input_left_embeds)
            self.encoder_right_outputs,self.encoder_right_laststate=tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                                      sequence_length=self.encoder_right_seqlen,
                                                                                      initial_state=encoder_left_laststate,
                                                                                      inputs=input_right_embeds)
            self.encoder_outputs=tf.concat([self.encoder_left_outputs,self.encoder_right_outputs],axis=1)
            self.input_seq_len=self.encoder_left_seqlen+self.encoder_right_seqlen
   
    def build_decoder(self):
        self.decoder_cell,self.decoder_init_state=self.build_decoder_cell()
        #last word in glove_weights is <start> token
        start_token_ind=self.glove_weights.shape[0]-1
        start_token=tf.ones(dtype=tf.int32,shape=[self.batch_size,1])*start_token_ind
        self.decoder_input=tf.concat([start_token,self.decoder_inputs],axis=1)
        self.decoder_inputs_embeds=tf.nn.embedding_lookup(self.embedding_weights,self.decoder_input)
        
        #helper
        helper=seq2seq.TrainingHelper(inputs=self.decoder_inputs_embeds,sequence_length=self.decoder_seqlen)
        #basicDecoder
        output_layer=Dense(self.vocab_size,name='output_layer')
        basic_decoder=seq2seq.BasicDecoder(cell=self.decoder_cell,helper=helper,
                                           initial_state=self.decoder_init_state,
                                           output_layer=output_layer)
                                           
        #dynamic_decoder
        self.max_decoder_len=tf.reduce_max(self.decoder_seqlen) # scalar
        self.decoder_outputs,self.decoder_final_state,_=seq2seq.dynamic_decode(decoder=basic_decoder,maximum_iterations=self.max_decoder_len)
        
        self.decoder_logits=tf.identity(self.decoder_outputs.rnn_output)
    def build_decoder_cell(self):
        decoder_cell_list=[self.make_cell() for _ in range(self.num_layers)]
        attention_mechanism=seq2seq.LuongAttention(memory=self.encoder_outputs,
                                                   num_units=self.hidden_size,
                                                   memory_sequence_length=self.input_seq_len)
        
        decoder_cell_list[-1]=seq2seq.AttentionWrapper(cell=decoder_cell_list[-1],
                                                       attention_mechanism=attention_mechanism,
                                                       initial_cell_state=self.encoder_right_laststate[-1],
                                                       attention_layer_size=self.hidden_size)
        
        decoder_cell=rnn.MultiRNNCell(cells=decoder_cell_list)
        init_state=[s for s in self.encoder_right_laststate]
        init_state[-1]=decoder_cell_list[-1].zero_state(dtype=tf.float32,batch_size=self.batch_size)
        decoder_initial_state=tuple(init_state)
        
        return decoder_cell,decoder_initial_state
    
    def calculate_loss(self):
        mask=tf.sequence_mask(dtype=tf.float32,lengths=self.decoder_seqlen,maxlen=self.max_decoder_len)
        self.loss=seq2seq.sequence_loss(logits=self.decoder_logits,targets=self.decoder_inputs,
                                        weights=mask,average_across_timesteps=True,average_across_batch=True)
        
    def optimize(self):
        self.new_lr=tf.placeholder(tf.float32,shape=[])
        self.learning_rate=tf.Variable(tf.constant(0.0,shape=[]),trainable=False,name='learning_rate')
        self.update_lr=self.learning_rate.assign(self.new_lr)
        self.optimizer=tf.train.AdamOptimizer(self.learning_rate,name='optimizer').minimize(self.loss)

# model=sentence_filling_model(batch_size=12,hidden_size=200,num_layers=2,keep_prob=0.8,vocab_size=40000,embedding_dim=200)


# In[ ]:


def get_batch(batch_size,window_size,int_sents):
    batch_len=len(int_sents)//batch_size
    print('batch_len:',batch_len)
    def get_max_length(sents):
        return max(len(sent) for sent in sents)
    int_sents=np.array(int_sents[:batch_size*batch_len])
    max_len=get_max_length(int_sents)
    print('max_len:',max_len)
    int_sents=np.reshape(int_sents,[batch_size,batch_len])
    
    
    half_window_size=window_size//2
    for i in range(batch_len-window_size):
#         encoder_left_sents=int_sents[:,i:i+half_window_size]
        encoder_left_words=np.zeros((batch_size,half_window_size*max_len),dtype=np.int32)
        encoder_left_seqlen=np.zeros((batch_size,1),dtype=np.int32)
#         encoder_right_sents=int_sents[:,i+half_window_size+1:i+window_size]
        encoder_right_words=np.zeros((batch_size,half_window_size*max_len),dtype=np.int32)
        encoder_right_seqlen=np.zeros((batch_size,1),dtype=np.int32)
#         decoder_sents=int_sents[:,i+half_window_size]
        decoder_words=np.zeros((batch_size,max_len),dtype=np.int32)
        decoder_seqlen=np.zeros((batch_size,1),dtype=np.int32)
        
        for j in range(batch_size):
            last_encoder_left_ind=0
            last_encoder_right_ind=0
            for k in range(half_window_size):
#                 print('i:',i,'j:',j,'k:',k)
#                 print(int_sents[j,i+k])
                encoder_left_words[j,last_encoder_left_ind:last_encoder_left_ind+len(int_sents[j,i+k])]=int_sents[j,i+k]
                last_encoder_left_ind+=len(int_sents[j,i+k])
                encoder_right_words[j,last_encoder_right_ind:last_encoder_right_ind+len(int_sents[j,i+half_window_size+1+k])]=int_sents[j,i+half_window_size+1+k]
                last_encoder_right_ind+=len(int_sents[j,i+half_window_size+1+k])
            encoder_left_seqlen[j,0]=last_encoder_left_ind
            encoder_right_seqlen[j,0]=last_encoder_right_ind
            decoder_words[j,:len(int_sents[j,i+half_window_size])]=int_sents[j,i+half_window_size]
            decoder_seqlen[j,0]=len(int_sents[j,i+half_window_size])
        
        yield encoder_left_words,encoder_left_seqlen,encoder_right_words,encoder_right_seqlen,decoder_words,decoder_seqlen
                
                
#             for k in range(window_size//2+1,window_size): # right sentences
            # x_lenght : 
    
#     return x,x_length,y,y_length
   
# with open(r'C:\Users\mehrnaz\Anaconda3\envs\environment1\Codes\work\sentence_filling\int_sents.pickle','rb') as handle:
#     int_sents=pickle.load(handle)
# get_batch(100,10,int_sents,100,1)

