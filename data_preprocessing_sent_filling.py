
# coding: utf-8

# In[33]:


import nltk
import numpy as np
import pickle


# In[34]:


nltk.download('gutenberg')
nltk.download('punkt')

def get_sentences(is_training):
    if is_training==True:
        sents=nltk.corpus.gutenberg.sents('austen-emma.txt')
    else:
        sents=nltk.corpus.gutenberg.words('austen-persuasion.txt')
    return sents

def get_max_length(sents):
    return max(len(sent) for sent in sents)


# In[35]:


# biuling glove_vectors and stoi , stoi is a dictionary from words, to indices in the glove_vectors, glove_vectors is a 2d np array from indices to embeddings
def get_glove_vectors(path):
    with open(path,encoding='UTF-8') as f:
        glove_vectors=np.zeros((400000,50),dtype=np.float32)
        stoi={}
        i=0
        for line in f.readlines():
            parts=line.split()
            stoi[parts[0]]=i
            glove_vectors[i]=[float(num) for num in parts[1:]]
            i+=1
    return glove_vectors,stoi


# In[39]:


def save(glove_vectors,stoi,int_sents):
    with open('glove_vectors.pickle','wb') as handle:
        pickle.dump(glove_vectors,handle,protocol=pickle.HIGHEST_PROTOCOL)
    with open('stoi.pickle','wb') as handle:
        pickle.dump(stoi,handle,protocol=pickle.HIGHEST_PROTOCOL)
    with open('int_sents.pickle','wb') as handle:
        pickle.dump(int_sents,handle,protocol=pickle.HIGHEST_PROTOCOL)


# In[40]:


#input: istraining is a boolean
#output: a list of size equal to the number of all words in the corpus, containing indices in the glove_vector for each word
# by encountering a word which is not in the glove dataset, we add it to stoi and assign a random embedding to it
def words_to_int(istraining,glove_path):
    words=get_words(istraining)
    glove_vectors,stoi=get_glove_vectors(glove_path)
    int_words=[]
    for word in words:
        try:
            int_words=int_words+[stoi[word]]
        except:
            stoi[word]=glove_vectors.shape[0]+1
            glove_vectors=np.vstack([glove_vectors,np.random.normal(size=glove_vectors.shape[1])])
            int_words+=[stoi[word]]
    return glove_vectors,stoi,int_words


# In[42]:


# words of setences should be replaced by corresponding int, in sentence vector
def sent_to_int(istraining,glove_path):
    sents=get_sentences(istraining)
    glove_vectors,stoi=get_glove_vectors(glove_path)
    int_sents=[]
    for sentence in sents:
        int_sentence=[]
        for word in sentence:
            try:
                int_sentence+=[stoi[word]]
            except:
                stoi[word]=glove_vectors.shape[0]+1
                glove_vectors=np.vstack([glove_vectors,np.random.normal(size=glove_vectors.shape[1])])
                int_sentence+=[stoi[word]]
        int_sents+=[int_sentence]
    save(glove_vectors,stoi,int_sents)
    return int_sents

int_sents=sent_to_int(True,r'D:\uni\shenakht pajuh\work\glove_data\glove.6B.50d.txt')
print(int_sents[:3])

