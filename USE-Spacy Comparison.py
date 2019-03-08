# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:01:56 2019

@author: Vishnu
"""

import tensorflow_hub as hub
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import spacy

nlp = spacy.load('en')

host = mongo host
db_client = MongoClient(host, port=port)
 
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/2"
embed = hub.Module(module_url)

def SimilarUSE(query, _id):
    data = list(db_client.chatbotplatform.intents.find({'chatBotId': str(_id)}, {'mappings':1, 'intentId': 1}))
    list_data = [list(i['mappings'].keys()) for i in data if 'mappings' in i and type(i['mappings']) == dict]
    flat_list = [item for sublist in list_data for item in sublist]
    list_sent = [query] + flat_list
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(list_sent))
    sim = cosine_similarity(message_embeddings[0:1], message_embeddings)
    print (sim)
    sim = sim[0].tolist()
    sim.pop(0)
    index = [sim.index(max(sim)) for i in sim if i > .8]
    try:
        similar = flat_list[index[0]]
        result = []
        for j in data:
            if 'mappings' in j and type(j['mappings']) == dict and similar in j['mappings']:
                result_dict = {}
                result_dict['response'] = similar
                result_dict['intentId'] = j['intentId']
                try:
                    result_dict['entityName'] = j['mappings'][similar]['entitiesData'][0]['entityName']
                    result_dict['name'] = j['mappings'][similar]['entitiesData'][0]['name']
                    result_dict['id'] = j['mappings'][similar]['entitiesData'][0]['id']
                except:
                    result_dict['entityName'] = ''
                    result_dict['name'] = ''
                    result_dict['id'] = ''
                result.append(result_dict)
        return result
    except:
        result = [{'response': 'No Match Found', 'intentId': ''}]
        return result

def SimilarSpacy(query, _id):
    data = list(db_client.chatbotplatform.intents.find({'chatBotId': str(_id)}, {'mappings':1, 'intentId': 1}))
    list_data = [list(i['mappings'].keys()) for i in data if 'mappings' in i and type(i['mappings']) == dict]
    flat_list = [item for sublist in list_data for item in sublist]
    list_indx = []
    for indx, i in enumerate(flat_list):
        dict_indx = {}
        dict_indx['index'] = indx
        dict_indx['similarity'] = nlp(query).similarity(nlp(i.lower()))
        if dict_indx['similarity'] > .72:
            list_indx.append(dict_indx)
    try:
        refined = max(range(len(list_indx)), key=lambda index: list_indx[index]['similarity'])
        ind = list_indx[refined]['index']
        similar = flat_list[ind]
        result = []
        for j in data:
            if 'mappings' in j and type(j['mappings']) == dict and similar in j['mappings']:
                result_dict = {}
                result_dict['response'] = similar
                result_dict['intentId'] = j['intentId']
                result_dict['similarity'] = list_indx[0]['similarity']
                result.append(result_dict)
        return result
    except:
        result = [{'response': 'No Match Found', 'intentId': ''}]
        return result
    