# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
from model_adhominem import AdHominem
import numpy as np
import pickle
import os
from somajo import SoMaJo
from tqdm import tqdm
from dataset import GermEval2021


def preprocess_doc(doc, tokenizer):
    sentences = tokenizer.tokenize_text([doc])
    doc_new = ''
    for sentence in sentences:
        for token in sentence:
            doc_new += token.text + ' '
    doc_new = doc_new.strip()
    return doc_new


def make_inference(doc, adhominem, sess):
    # return writing style embeddings
    emb = adhominem.inference([doc], sess)
    return emb


def get_writig_style_embeddings(corpus_train, corpus_dev):

    # load tokenizer
    tokenizer = SoMaJo(language='en_PTB', split_sentences=True)  # "de_CMC"

    # load Tensorflow models
    tf.reset_default_graph()

    with open(os.path.join("adhominem"), 'rb') as f:
        parameters = pickle.load(f)

    with tf.variable_scope('AdHominem'):
        adhominem = AdHominem(hyper_parameters=parameters['hyper_parameters'],
                              theta_init=parameters['theta'],
                              theta_E_init=parameters['theta_E'],
                              )

    # launch Tensorflow session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # load docs
    docs_tr, docs_dev = [], []
    for doc in corpus_train:
        docs_tr.append(doc.to_plain_string())
    for doc in corpus_dev:
        docs_dev.append(doc.to_plain_string())

    # preprocess
    for i in tqdm(range(len(docs_tr)), desc="preprocessing (train)"):
        docs_tr[i] = preprocess_doc(docs_tr[i], tokenizer)

    for i in tqdm(range(len(docs_dev)), desc="preprocessing (dev)"):
        docs_dev[i] = preprocess_doc(docs_dev[i], tokenizer)

    # get embeddings
    emb_tr = np.zeros((len(docs_tr), 100))
    emb_dev = np.zeros((len(docs_dev), 100))

    for i in tqdm(range(len(docs_tr)), desc="inference (train)"):
        emb = make_inference(docs_tr[i], adhominem, sess)
        emb_tr[i, :] = emb

    for i in tqdm(range(len(docs_dev)), desc="inference (dev)"):
        emb = make_inference(docs_dev[i], adhominem, sess)
        emb_dev[i, :] = emb

    return emb_tr, emb_dev


if __name__ == '__main__':
    for fold in range(4):
        # load data
        corpus = GermEval2021("shuffeled_df.csv", fold=fold)
        # get LEVs
        emb_tr, emb_dev = get_writig_style_embeddings(corpus_train=corpus.train, corpus_dev=corpus.dev)
        # store
        file = os.path.join("adhominem", "emb_" + str(fold))
        with open(file, 'wb') as f:
            pickle.dump((emb_tr, emb_dev), f)

