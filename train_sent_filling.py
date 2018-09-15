
# coding: utf-8

# In[2]:


import numpy as np
import pickle


# In[11]:


def get_batch(batch_size,window_size,int_sents,max_length,epochs):
    batch_len=len(int_sents)//batch_size
    int_sents=np.array(int_sents[:batch_size*batch_len])
    int_sents=np.reshape(int_sents,[batch_size,batch_len])
    zeros=np.zeros((batch_size,window_size*max_length),dtype=np.float32)
    
    for _ in range(epochs):
        for i in range(batch_len-window_size):
            # x : left and right sents, size: [batch_size,window_size-1]
            for batch_i in range(batch_size):
                batch_i_left_words=[]
                for j in range(window_size//2): # left sentences
                    batch_i_left_words+=int_sents[batch_i,j]
                print(batch_i_left_words)
                
#             for k in range(window_size//2+1,window_size): # right sentences
            # x_lenght : 
    
#     return x,x_length,y,y_length
   
with open(r'C:\Users\mehrnaz\Anaconda3\envs\environment1\Codes\work\sentence_filling\int_sents.pickle','rb') as handle:
    int_sents=pickle.load(handle)
get_batch(100,10,int_sents,100,1)

