
# coding: utf-8

# In[80]:


import numpy as np
import pickle
import tensorflow as tf
import import_ipynb
from sentence_filling_model import sentence_filling_model


# In[74]:


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


# In[75]:


def train(int_words_train,glove_vectors,batch_size=64,hidden_size=200,num_layers=1,keep_prob=0.8,window_size=21):

    vocab_size=glove_vectors.shape[0]
    embedding_dim=glove_vectors.shape[1]
    with tf.device('/gpu:0'):
        with tf.variable_scope('model',tf.AUTO_REUSE):
            model=sentence_filling_model.sentence_filling_model(batch_size=batch_size,hidden_size=hidden_size,num_layers=num_layers,keep_prob=keep_prob,vocab_size=vocab_size,embedding_dim=embedding_dim)
    
    blank_embedding=np.random.normal(size=[embedding_dim])
    with tf.name_scope('run_session'):
        sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run([model.embedding_init],feed_dict={model.embedding_placeholder:glove_vectors})
        for epoch in range(epochs):
            batch_generator=batch_generator=get_batch(batch_size=15,int_sents=int_sents_train,window_size=21)
            i=0
            for encoder_left_words,encoder_left_seqlen,encoder_right_words,encoder_right_seqlen,decoder_words,decoder_seqlen in batch_generator:
                feed_dict={model.encoder_left_words:encoder_left_words,
                           model.encoder_right_words:encoder_right_words,
                           model.encoder_left_seqlen:encoder_left_seqlen,
                           model.encoder_right_seqlen:encoder_right_seqlen,
                           model.decoder_words:decoder_words,
                           model.decoder_seqlen:decoder_seqlen,
                           model.blank:blank_embedding}
                _,loss=sess.run([model.optimizer,model.loss],feed_dict=feed_dict)
                print('epoch:',epoch,'i:',i,'loss:',loss)
                i+=1
            
    
    
if __name__=='__main__':
    with open('glove_vectors_with_unk.pickle','rb') as handle:
        glove_vectors=pickle.load(handle)
    with open('int_sents_train.pickle','rb') as handle:
        int_sents_train=pickle.load(handle)
        int_sents_train=int_sents_train[:len(int_sents_train)//15]
        print(len(int_sents_train))
    batch_generator=get_batch(batch_size=15,int_sents=int_sents_train,window_size=21)
#     for encoder_left_words,encoder_left_seqlen,encoder_right_words,encoder_right_seqlen,decoder_words,decoder_seqlen in batch_generator:
#         print(encoder_left_words)
#         print(encoder_left_seqlen)
#         print(encoder_right_words)
#         print(encoder_right_seqlen)
#         print(decoder_words)
#         print(decoder_seqlen)
    
    train(int_words_train,glove_vectors)


# In[ ]:




import tensorflow as tf
import tensorflow.contrib.rnn as rnn


# In[4]:


# bidirectional lstm model for word filling
# assumes center word as blank, predicts it and calculates loss for that
class biLstm_word_filling():
    def __init__(self,window_size,num_layers,vocab_size,embedding_dim,hidden_size,keep_prob,max_gradient_norm):
        
        with tf.device('/cpu:0'):
            self.num_layers=num_layers
            self.hidden_size=hidden_size
            self.max_gradient_norm=max_gradient_norm
            self.glove_weights = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),trainable=False, name="glove_weights")
            self.embedding_placeholder = tf.placeholder(tf.float32, [None, embedding_dim],name="embedding_placeholder")
            self.embedding_init = self.glove_weights.assign(self.embedding_placeholder)

            self.random_blanks=tf.placeholder(tf.float32,shape=[None,None],name='random_blanks')
                # adding random blank embeddings to the end of out embedding_init matrix
            self.embedding_with_randoms=tf.concat([self.glove_weights,self.random_blanks],axis=0)

            self.inputs=tf.placeholder(tf.int32,shape=[None,None],name='inputs')
            self.output=tf.placeholder(tf.int32,shape=[None],name='output') # center word of the window
            self.batch_size=tf.shape(self.inputs)[0]

    #         self.init_state=tf.placeholder(tf.float32,[self.num_layers,2,self.batch_size,self.hidden_size])
    #         state_per_layer_list = tf.unstack(self.init_state, axis=0)
    #         rnn_tuple_state = tuple([rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) for idx in range(num_layers)])

        with tf.name_scope("embedding_lookup"):
                # getting embedding vectors of input indices from embedding_init
            self.embeds=tf.nn.embedding_lookup(self.embedding_with_randoms,self.inputs)
    #             print(type(self.embeds))
    #             self.embeds[:,int(window_size//2),:].assign(tf.reduce_mean(self.embeds[:,int(window_size//2-5):int(window_size//2),:],axis=1)+tf.reduce_mean(self.embeds[:,int(window_size//2+1):int(window_size//2+6),:],axis=1))
        with tf.device('/gpu:0'):
            with tf.name_scope("bi_lstm"):
                def make_cell():
                    cell=rnn.BasicLSTMCell(self.hidden_size)
                    cell=rnn.DropoutWrapper(cell,output_keep_prob=keep_prob)
                    return cell

                # forward and backward cell
#                 fw_cell=rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
                fw_cell=make_cell()
                bw_cell=make_cell()
#                 bw_cell=rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
                lstm_outputs,self.last_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=self.embeds,dtype=tf.float32)

                fw_output=lstm_outputs[0][:,int(window_size//2),:]
                bw_output=lstm_outputs[1][:,int(window_size//2),:]
                # predicted_outputs before softmax, from hiddens
                predicted_output=tf.concat([fw_output,bw_output],axis=1)

                # logits : densed predicted_output to vocab_size so we can pass it to softmax cross entropy loss
#             with tf.device('/cpu:0'):
                self.logits = tf.layers.dense(predicted_output, vocab_size,name='logits')
                self.one_hot_outputs=tf.one_hot(indices=self.output,depth=vocab_size,axis=-1)

                with tf.name_scope("loss"):
                    self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.logits,
                    labels=self.one_hot_outputs,
                    dim=1,
                    name='loss')
                self.cost=tf.reduce_mean(self.loss,name='cost')
                trainable_params=tf.trainable_variables()
                clip_gradients=[tf.clip_by_norm(grads,self.max_gradient_norm) for grads in tf.gradients(self.loss,trainable_params)] 
   
                
                self.new_lr=tf.placeholder(tf.float32,shape=[])
                self.learning_rate=tf.Variable(0.0,trainable=False,name='learning_rate')
                self.update_lr=self.learning_rate.assign(self.new_lr)
                self.optimizer=tf.train.AdamOptimizer(self.learning_rate,name='optimizer')
                self.updates = self.optimizer.apply_gradients(zip(clip_gradients, trainable_params))


# In[ ]:



# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import pickle
import biLstm_word_filling


# In[2]:


# batch generator
# output : x is a tensor of size [batch_size,window_size] containing indices of words in those windows , 
# random_blanks : random embeddings for blanks
def get_batch(batch_size,window_size,int_words,embedding_dim,glove_vectors_size):
    batch_len=(len(int_words)//batch_size)
    print(batch_len)
    int_words=np.array(int_words[:batch_len*batch_size])
    int_words=np.reshape(int_words,[batch_size,batch_len])
        # we take middle word of the window (that is going to be predicted) as a blank word with a random embedding ,
        # we add these random embeddings to the end of the glove_vectors
    blank_indices=[glove_vectors_size+i for i in range(batch_size)]
    for step_start in range(batch_len-window_size+1):
        x=np.copy(int_words[:,step_start:step_start+window_size])
        y=np.copy(x[:,int(window_size//2)])
    #             stoi['blank{}'.format(i)]=glove_vectors.shape[0]+1
    #             blank_embedding=np.mean(x[:,window_size//2-5:window_size//2],axis=1)+np.mean(x[:,window_size//2+1:window_size//2+5],axis=1)

                # random embeddings for blanks
        random_blanks=np.random.normal(size=(batch_size,embedding_dim))
        x[:,int(window_size//2)]=blank_indices
        yield x,y,random_blanks


# In[3]:


def train(int_words, glove_vectors, window_size=101, num_layers=1,
          batch_size=64, hidden_size=200, keep_prob=0.8, epochs=5,initial_lr=0.02,lr_decay_rate=0.8,max_gradient_norm=5):
    
    vocab_size=glove_vectors.shape[0]
    embedding_dim=glove_vectors.shape[1]
        # building bidirectional lstm model with appropriate parameters
    model=biLstm_word_filling.biLstm_word_filling(window_size=window_size,embedding_dim=embedding_dim,hidden_size=hidden_size,keep_prob=keep_prob,num_layers=num_layers,vocab_size=vocab_size,max_gradient_norm=max_gradient_norm)

    with tf.name_scope('run_session'):
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run([model.embedding_init],feed_dict={model.embedding_placeholder:glove_vectors})
        
        min_loss=10000
        for epoch in range(epochs):
            lr=initial_lr*(np.power(lr_decay_rate,epoch))
            print('lr:',lr)
            sess.run([model.update_lr],feed_dict={model.new_lr:lr})
            learning_rate=sess.run([model.learning_rate])
            print('learning_rate:',learning_rate)
            batch_generator=get_batch(batch_size=batch_size,window_size=window_size,int_words=int_words,embedding_dim=embedding_dim,glove_vectors_size=vocab_size)
            i=0
            for x,y,random_blanks in batch_generator:
            
                feed_dict={model.inputs:x,model.output:y,model.random_blanks:random_blanks}
                _,loss=sess.run([model.updates,model.cost],feed_dict=feed_dict)
                min_loss=min(min_loss,loss)
                if i%50==0:
                    print('i:',i,' loss:',loss)
                i+=1
        print('min_loss:',min_loss)
#             saver.save(sess, './trained_word_filling.ckpt',global_step=1000)
        saver = tf.train.Saver()    
        saver.save(sess, './trained_word_filling.ckpt')
            
if __name__=='__main__':
    with open('glove_vectors.pickle','rb') as handle:
        glove_vectors=pickle.load(handle)
    with open(r'int_words_train.pickle','rb') as handle:
        int_words=pickle.load(handle)
        print(int_words[:3])
    train(int_words,glove_vectors)


# In[ ]:



# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import pickle
import biLstm_word_filling
import numpy.random as random


# In[ ]:


# def get_batch_test(batch_size,window_size,int_words,embedding_dim,glove_vectors_size):
#     with tf.device('/gpu:0'):
#         batch_len=2*window_size
#         int_words=np.array(int_words[:batch_size*batch_len])
#         int_words=np.reshape(int_words,[batch_size,batch_len])
#         # we take middle word of the window (that is going to be predicted) as a blank word with a random embedding ,
#         # we add these random embeddings to the end of the glove_vectors
#         blank_indices=[glove_vectors_size+i for i in range(batch_size)]
#         for step_start in range(batch_len-window_size+1):
#             x=np.copy(int_words[:,step_start:step_start+window_size])
#             y=np.copy(x[:,int(window_size//2)])
#             # random embeddings for blanks
#             random_blanks=np.random.normal(size=(batch_size,embedding_dim))
#             x[:,int(window_size//2)]=blank_indices
#             yield x,y,random_blanks


# In[ ]:


# def add_blanks_to_glove(glove_vectors,num_blanks,embedding_dim):
#     glove_vectors=np.concatenate((glove_vectors,np.random.normal(size=(num_blanks,embedding_dim))),axis=0)
#     return glove_vectors
    
def delete_some_words(int_words,glove_vectors_size,window_size,int_to_vocab):
    blank_indices=[]
    # a dictionary from blank index in int_words to its real index in glove_vectors
    real_words={}
    for i in range(window_size//2,len(int_words)-(window_size//2)+1):
        if random.uniform()<0.03:
#             blank_indices.append(i)
            int_to_vocab[glove_vectors_size]=int_to_vocab[int_words[i]]
            real_words[i]=int_words[i]
            int_words[i]=glove_vectors_size
            glove_vectors_size+=1
    return int_words,glove_vectors_size,real_words,int_to_vocab


# In[ ]:


def test(int_words,glove_vectors,int_to_vocab,modelpath,window_size=101,num_layers=2,hidden_size=200,keep_prob=1,batch_size=1,max_gradient_norm=5):
    

        
    vocab_size=glove_vectors.shape[0]
    embedding_dim=glove_vectors.shape[1]
    model=biLstm_word_filling.biLstm_word_filling(window_size=window_size,num_layers=num_layers,embedding_dim=embedding_dim,hidden_size=hidden_size,keep_prob=keep_prob,vocab_size=vocab_size)
    saver=tf.train.Saver()
    file=open('result_1.txt',"w")
    with tf.Session() as sess:
#             trained_model=tf.train.import_meta_graph('./trained_word_filling-1000.meta')
#             trained_model.restore(sess,tf.train.latest_checkpoint('./'))
#             graph=tf.get_default_graph()
#             inputs=graph.get_tensor_by_name('inputs:0')
#             output=graph.get_tensor_by_name('output:0')
#             random_blanks=graph.get_tensor_by_name('random_blanks:0')
            
#             loss=graph.get_tensor_by_name('loss:0')
#             logits=graph.get_tensor_by_name('logits:0')
        saver.restore(sess,modelpath)
        int_words,glove_vectors_size,real_words,int_to_vocab=delete_some_words(int_words=int_words,glove_vectors_size=vocab_size,window_size=window_size,int_to_vocab=int_to_vocab,max_gradient_norm=max_gradient_norm)
        num_blanks=glove_vectors_size-vocab_size
#             glove_vectors=add_blanks_to_glove(glove_vectors,num_blanks,embedding_dim)
#             sess.run(tf.global_variables_initializer())
        x=np.zeros((1,window_size),dtype=np.int32)
#             y=np.zeros((1,1),dtype=np.int32)
        randomblanks=np.random.normal(size=(num_blanks,embedding_dim))
#             print(randomblanks.shape)
        
#             print(inputs.shape)
#             print(output.shape)
        i=0
        min_loss=12000
        correct=0
        total=0
        for blank_index,real_int in real_words.items():
            x[0]=np.copy(int_words[blank_index-(window_size//2):blank_index-(window_size//2)+window_size])
            y=np.array([real_int])
#                 print(y.shape)
                
            feed_dict={model.inputs:x,model.output:y,model.random_blanks:randomblanks}
            loss,logits=sess.run([model.loss,model.logits],feed_dict=feed_dict)
            min_loss=min(loss,min_loss)
            i+=1
#                 input_words=np.zeros(x.shape)
#                 label_words=np.zeros(y.shape)
                #logits is of size [batch_size,vocab_size]
#                 print(logits.shape)
            predicted_index=tf.argmax(logits,axis=1).eval()
#                 print('predicted')
#             print(predicted_index[0])
#                 print(x[0])
            input_words=[int_to_vocab[i] for i in x[0]]
            input_words[window_size//2]='<blank>'
            file.write('i:'+str(i)+' loss:'+str(loss))
            file.write('\n')
            print('i:',i,' loss:',loss)
            file.write('input_words:'+str(input_words))
            file.write('\n')
            print('input_words:',input_words)
            file.write('label:'+str(int_to_vocab[real_int]))
            file.write('\n')
            print('label:',int_to_vocab[real_int])
            file.write('predicted_word:'+str(int_to_vocab[predicted_index[0]]))
            file.write('\n')
            print('predicted_word:',str(int_to_vocab[predicted_index[0]]))
            total+=1
            if real_int==predicted_index[0]:
                correct+=1
        file.write('correct_factor:'+str(correct/total))
        print('correct_factor:',str(correct/total))

        file.write('min_loss:'+str(min_loss))
        print('min_loss:',min_loss)
        file.close()
                
#                 predicted_words=np.zeros(predicted_indices.shape)
#                 for i in range(x.shape[0]):
#                     for j in range(x.shape[1]):
#                         input_words[i,j]=int_to_vocab[x[i,j]]
#                 for i in range(y.shape[0]):
#                     label_words[i,1]=int_to_vocab[y[i,1]]
#                 for i in range(predicted_indices.shape[0]):
#                     predicted_words[i,1]=int_to_vocab[predicted_indices[i,1]]
                
#                 print('inputs:',input_words)
#                 print('labels:',label_words)
#                 print('predicted_outputs:',predicted_words)
#                 i+=1
#                 if i%(batch_size-1)==0:
#                     i=0
                    
                    


# In[ ]:


if __name__=="__main__":
    with open('./int_words_test.pickle','rb') as handle:
        int_words=pickle.load(handle)
        int_words=int_words[:int(0.5*len(int_words))]
    with open('./glove_vectors.pickle','rb') as handle:
        glove_vectors=pickle.load(handle)
        print(glove_vectors.shape)
    with open('./int_to_vocab_test.pickle','rb') as handle:
        int_to_vocab=pickle.load(handle)
    print(int_to_vocab[0])
    test(int_words,glove_vectors,int_to_vocab,modelpath='./trained_word_filling.ckpt')

