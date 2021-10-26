# Required utilies for NER Project 

#imports 
import numpy as np
import pandas as pd
from tensorflow import keras
import nltk

# download nltk libraries
nltk.download('punkt')

def get_vocab(all_word_list,all_tag_list):
  '''
  funciton to build vocabulary and return word2id and id2word and tag2id and id2tag
  '''
  # word vocabulary
  word_vocab = set(all_word_list)

  # tag vocabulary
  tag_vocab = set(all_tag_list)

  word2id = {word:id for id,word in enumerate(word_vocab)}
  id2word ={id:word for id,word in enumerate(word_vocab)}

  # Add Aditional index for 'UNKOWN WORDs' and 'PADDING'
  word2id['UNK'] = len(word_vocab) # adding at the end
  word2id['<PAD>'] = len(word_vocab) +1 # adding at the end

  id2word[len(word_vocab)]='UNK'
  id2word[len(word_vocab)+1] ='<PAD>'



  tag2id ={tag:id for id,tag in enumerate(tag_vocab)}
  id2tag ={id:tag for id,tag in enumerate(tag_vocab)}

  return word2id,id2word,tag2id,id2tag

def get_index(words,word2id):
  '''
  funtion to convert tokenized words into list of indices in vocab
  also tags into list of indices in tag vocab
  input : words- Tokenized words
          word2id - Dictionary with {word,index in dictionary}
  output : list of indices
  '''
  # check if word is in dictionary otherwise return word2id['UNK']

  words_index =[word2id[word] if word in word2id
                else word2id['UNK']
                for word in words]

  return words_index

def pad_values(sentences,vocab,max_length,pad='<PAD>'):

  '''
  Function to pad the sequence 
  input : sentences - list of sequences 
        : vocab - dictionarary of {words,index in vocabulary}
        : max_length - padding length
        : pad - padd token represented in the vocabulary
  output : padded sequences
  '''
  from keras.preprocessing.sequence import pad_sequences

  padded_sentece = pad_sequences(sentences,maxlen= max_length,value=vocab[pad])
  return padded_sentece

def NER(sentence,model,word2id,id2tag,max_len):
  '''
  function to get Named Entity recoginition of text passed 
  Input : sentence - Input text in str
        : model - Model
        : word2id - dictionary with {word,index for word in vocab}
        : id2tag - dictionary with {id of tag, tag}
        : max_len  - maximum sentence length supported by model

  output : df - Dataframe with two columsn ['words','tokens'] ,
           tokens reprent the NER tags
  '''
  import nltk
  import pandas as pd
  from tensorflow import keras
  import numpy as np



  #1. tokenzie sentence 
  sent_token = nltk.word_tokenize(sentence)
  # check the sent_length is within acceptable range
  if len(sent_token)>max_len:
    sent_token = sent_token[:max_len-1]

  #2. convert to indices
  sent_indices = get_index(sent_token,word2id)
  #3. Pad 
  padded_sent = pad_values([sent_indices],word2id,max_len)
  #4. Prediction
  predicition = model.predict(padded_sent)
  # Prediction label
  prediction_label = np.argmax(predicition,axis=2)
  #prediction_label

  ## Map the entity codes 
  tag2tag = {id:tag[2:] for id,tag in id2tag.items()}

  df = pd.DataFrame()
  df['sent']=sent_token
  df['tokens']=np.array(pd.Series(prediction_label[0]).map(tag2tag)[-len(sent_token):])
  #print(df.to_string())
  return df

tag2color ={
    '':'#FFFFFF',    # ghostwhite    
    'art':'#FF6A6A', # indian red 
    'eve':'#FF7D40', # flesh - orangish
    'geo':'#7FFF00', # chartreuse1 - greenish
    'gpe':'#FFD700', # gold1
    'nat':'#C0FF3E', # olivedrab1
    'org':'#87CEFA', # lightskyblue
    'per':'#E066FF', # mediumorchid1
    'tim':'#CDC1C5'# lavenderblush3
}

tag2fullform={
    'org':'Organization',
    'geo':'Geographical Entity',
    'per':'Person',
    'tim':'Time Indicator',
    'art':'Artifact',
    'eve':'Event',
    'gpe':'Geopolitical Entity',
    'nat':'Natural Phenomenon'   
}

def get_annoted_text(df,tag2color):

  import pandas as pd
  df1= df.copy()
  df1['color'] = df1['tokens'].map(tag2color)

  # text in annotated text format
  #('word','entity name','color) #tuple(df1.iloc[i,:1])
  #anotated_text_list =[tuple(df1.iloc[i,:]) for i in range(len(df1))]
  anotated_text_list =[df1.iloc[i,0]+' ' if df1.iloc[i,2]=='#FFFFFF' 
                       else tuple(df1.iloc[i,:]) 
                       for i in range(len(df1))]
  
  
  return anotated_text_list

def load_voabularies():
  '''
  function to load all vocabularies for the project
  '''
  import pickle
  folder_path ='./trained vocab/'
  all_dictionaries = []
  vocab_names = ['word2id','id2word','tag2id','id2tag']
  for vocab_name in vocab_names:
    with open(folder_path + vocab_name+'.pickle','rb') as file:
      vocab=pickle.load(file)
      all_dictionaries.append(vocab)

  
  word2id,id2word,tag2id,id2tag = all_dictionaries[0],all_dictionaries[1],all_dictionaries[2],all_dictionaries[3]
  return word2id,id2word,tag2id,id2tag
