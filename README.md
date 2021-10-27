# NER-Named-Entity-Recognition
This app recognizes the named entities in a text document. It is designed to detect names of entities like Organisation, Geographical entities, Person, Time Indicator, Artifact, Events, Geopolitical Entity,
Natural Phenomenon.

[![open streamlit app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/999harish999/ner-named-entity-recognition/app.py)

## Demo



## Problem Statement
To locate and classify named entities mentioned in unstructured text into predefined categories such as person names, Organization, locations, time indicator, events and natural phenomenon.

## Motivation
Most of the valuable information on the internet is in the form of unstructured text data. From a business perspective, data is generated every minute in form of news articles, social media platforms, blogs etc. To have an impact on improving business its very important to extract meaningful information from these text data. The extracted information can be used to solve business problems. Bellow are some use cases where Named Entity Recognition can be used.

 - Classifying News articles
 - Efficient Search Algorithms
 - Powering Content Recommendations
 - Customer support 

## Data
The Data used in this project is an extract from GMB(Groningen Meaning Bank) corpus. Every word in the dataset is tagged with the entity as mentioned below: 

- geo = Geographical Entity
- org = Organization
- per = Person
- gpe = Geopolitical Entity
- tim = Time indicator
- art = Artifact
- eve = Event
- nat = Natural Phenomenon

Total Words Count = 1354149  

The data can be downloaded the bellow link from kaggle website:  
[https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)

GMB(Groningen Meaning Bank) corpus link:  
[https://gmb.let.rug.nl/data.php](https://gmb.let.rug.nl/data.php)

## Deployment 

The Web app is built using Streanlit and deployed on https://share.streamlit.io/   
To launch the app click on bellow    


[![open streamlit app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/999harish999/ner-named-entity-recognition/app.py)

## Requirements 

App was developed and deployed using bellow configuration 

```
numpy==1.19.5
nltk==3.2.5
pandas==1.1.5
tensorflow==2.6.0
streamlit==0.89.0
st-annotated-text==2.0.0

```
