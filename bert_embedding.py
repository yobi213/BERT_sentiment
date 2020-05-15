import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize

import logging

logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bertmodel = BertModel.from_pretrained('bert-base-uncased')
bertmodel.eval()
bertmodel.to('cuda')


def transform_bert(text):
    return '[CLS] ' + text + ' [SEP]'

def make_zeros(length):
    return np.zeros(length).astype(int).tolist()

def preprocess_bert(text):
    sentences = sent_tokenize(text)
    # 문장 중 토큰수 500개 넘는거 글자수 줄이기
    for i in range(len(sentences)):
        if len(word_tokenize(sentences[i])) > 440:
            sentences[i] = sentences[i][:2000]
    transformed = list(map(transform_bert,sentences))
    tokenized_text = list(map(tokenizer.tokenize,transformed))
    indexed_tokens = list(map(tokenizer.convert_tokens_to_ids,tokenized_text))
    segments_ids = list(map(make_zeros,list(map(len,indexed_tokens))))
    return indexed_tokens, segments_ids

def get_embedding_token(indexed_tokens, segments_ids):
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    with torch.no_grad():
        encoded_layers, _ = bertmodel(tokens_tensor, segments_tensors)
    return encoded_layers[-4][:][0] + encoded_layers[-3][:][0] + encoded_layers[-2][:][0] + encoded_layers[-1][:][0]

def get_i_s(doc):
    i,s = preprocess_bert(doc)
    return i,s

def get_doc_embedding_100(i,s):
    outputs = []
    for num_sent in range(len(i)):
        vecs = get_embedding_token(i[num_sent],s[num_sent])
        outputs.append(vecs)
    result = torch.cat(outputs, dim=0).to('cuda')
    if len(result) < 100:
        pad_len = 100 - len(result)
        padding = torch.zeros(pad_len,768).to('cuda')
        outputs.append(padding)
        result = torch.cat(outputs, dim=0).to('cuda')
    return result[:100].to('cuda')


def get_doc_embedding(doc):
    i,s = get_i_s(doc)
    embedding = get_doc_embedding_100(i,s)
    return embedding

def bert_embedding(docs):
    a = list(map(get_doc_embedding,docs))
    b = torch.cat(a)
    return b.view(-1,100,768)
