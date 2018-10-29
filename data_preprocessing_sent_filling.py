
# coding: utf-8

# In[11]:


import nltk
import numpy as np
import pickle


# In[12]:


nltk.download('gutenberg')
nltk.download('punkt')

def get_sentences(is_training):
    if is_training==True:
        sents=nltk.corpus.gutenberg.sents('austen-emma.txt')
    else:
        sents=nltk.corpus.gutenberg.sents('austen-persuasion.txt')
    return sents


# In[13]:


def save(int_sents_train,int_sents_test):
    with open('int_sents_train.pickle','wb') as handle:
        pickle.dump(int_sents_train,handle,protocol=pickle.HIGHEST_PROTOCOL)
    with open('int_sents_test.pickle','wb') as handle:
        pickle.dump(int_sents_test,handle,protocol=pickle.HIGHEST_PROTOCOL)


# In[33]:


def add_unk():
    with open('stoi.pickle','rb') as handle:
        stoi=pickle.load(handle)
        print(len(stoi))
    with open('glove_vectors.pickle','rb') as handle:
        glove_vectors=pickle.load(handle)
    
    glove_vectors=np.vstack([glove_vectors,np.random.normal(size=glove_vectors.shape[1])])
    stoi['<unk>']=glove_vectors.shape[0]-1
    with open('stoi_with_unk.pickle','wb') as handle:
        pickle.dump(stoi,handle,pickle.HIGHEST_PROTOCOL)
    with open('glove_vectors_with_unk.pickle','wb') as handle:
        pickle.dump(glove_vectors,handle,pickle.HIGHEST_PROTOCOL)


# In[37]:


# words of setences should be replaced by corresponding int, in sentence vector
def sent_to_int(sents,stoi):
    int_sents=[]
    for sentence in sents:
        int_sentence=[]
        for word in sentence:
            try:
                word=word.lower()
                int_sentence.append(stoi[word])
            except:
                int_sentence.append(stoi['<unk>'])
        int_sents.append(int_sentence)
    return int_sents

if __name__=='__main__':
    add_unk()
    with open('stoi_with_unk.pickle','rb') as handle:
        stoi=pickle.load(handle)
        print(len(stoi))
    #train:
    train_sents=get_sentences(True)
    print('len(train):',len(train_sents))
    int_sents_train=sent_to_int(train_sents,stoi)
    test_sents=get_sentences(False)
    print('len(test):',len(test_sents))
    int_sents_test=sent_to_int(test_sents,stoi)
    save(int_sents_train,int_sents_test)
    
    

