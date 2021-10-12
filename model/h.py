#!/usr/bin/env python3
"""
:file: h.py
:author: Brant Yukan
:date: 10/3/21
:brief: This header file contains function definitions for models.
"""

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


def calculate_sentence_embedding(text):
    """
    This function computes a dense vector representation of the input text using pytorch transformers library.  Refer to: http://localhost:8888/notebooks/movie-rec/model/similarity.ipynb
    :param: s is the string 
    :return: numpy vector of length 768
    """
    model_name = 'sentence-transformers/stsb-distilbert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokens = tokenizer.encode_plus(text, max_length=128,
                               truncation=True, padding='max_length',
                               return_tensors='pt')
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    return mean_pooled.detach().numpy()


def compute_similarity(main_index, sentences):
    """
    This function computes the ranked list of similarity scores from a list of sentences.
    https://github.com/jamescalam/transformers/blob/main/course/similarity/03_calculating_similarity.ipynb
    :param: main_index is the index that will be compared against all others
    :param: sentences a list of text descriptions
    :return: ranked list of cosine similarity scores
    """
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

# initialize dictionary that will contain tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sentences:
        # tokenize sentence and append to dictionary lists
        new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True,
                                           padding='max_length', return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']
    
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    mean_pooled = mean_pooled.detach().numpy()
    
    similarity_scores = cosine_similarity(
    [mean_pooled[main_index]],
    mean_pooled)
    
    ranked_list = sorted([(i, x) for i, x in enumerate(similarity_scores[0])], key=lambda x: x[1], reverse=True)
    return ranked_list[:10]