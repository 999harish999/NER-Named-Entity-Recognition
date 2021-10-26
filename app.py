# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:42:06 2021

@author: harish
"""

# Named Entity Recognition streamlit app

import streamlit as st
from annotated_text import annotated_text,annotation
from tensorflow import keras
import pickle
import utils

# load the models
model_file_path = './trained_models/model2d_15e.h5'
model = keras.models.load_model(model_file_path)
max_sentence_length =104

# load vocabulary dictionaries
word2id,id2word,tag2id,id2tag =utils.load_voabularies()



# set webpage in wide mode
st.set_page_config(layout="wide")


st.title('Named Entity Recognition')


# split into two containers 
main_section,info_section = st.columns([3,1])

def get_ner(text):
    #main_section.write('getting ner..')
    
    # preprocess input text and prdict 
    ner_df = utils.NER(text,model,word2id,id2tag,max_len = max_sentence_length)
    
    # get the anotated text
    anotated_input = utils.get_annoted_text(ner_df,utils.tag2color)
    return anotated_input

# function to draw the entities and their colors
def get_entities(tag2fullform=utils.tag2fullform,tag2color=utils.tag2color):
    from annotated_text import annotated_text
    anotated_entites=[]
    
    for tag,fullform in tag2fullform.items():
        
        anotated_format = (fullform,tag,tag2color[tag])
        
        annotated_text(anotated_format)
        st.write(' ')
    
    



# Main section
with main_section:
    st.write('### Input Text')
    
    # Input Text Area
    input_text = st.text_area('Text for Extraction...')
    # Output Text
    output_text =''

    # Button
    if st.button('Recognize Named Entities'):
      # check if input is not emty
      if input_text!='':
        output_text =get_ner(input_text)
      else:
        output_text ='Nothing entered'
    
    # Clear button
    if st.button('Clear output'):
      #clear output 
      output_text=''
    
    # output 
    annotated_text(*output_text)
    

    
    
with info_section:
    st.write('### Enities',)

    get_entities()


