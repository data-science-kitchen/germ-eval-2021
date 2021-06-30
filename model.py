from features import Feature
import joblib
import numpy as np
import optuna
import os
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from typing import Dict, List, Optional, Union

from sklearn.linear_model import LogisticRegression


class FeatureSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, mask: np.array) -> None:
        self.mask = mask

    def transform(self, X):
        return X[..., self.mask]

    def fit(self, X, y=None):
        return self


class EnsembleVotingClassifier(ClassifierMixin):
    def __init__(self, base_classifiers: List[ClassifierMixin]) -> None:
        self.base_classifiers = base_classifiers

    def predict(self, features: np.array) -> np.array:
        num_samples, num_classifiers = features.shape[0], len(self.base_classifiers)
        predictions = np.zeros((num_samples, num_classifiers, 3))

        for idx, classifier in enumerate(self.base_classifiers):
            predictions[:, idx, :] = classifier.predict(features)

        return (predictions.sum(axis=1) >= num_classifiers / 2.).astype(np.int64)


class GermEvalModel(ClassifierMixin):
    TASK_NAMES = ['Toxic', 'Engaging', 'FactClaiming']

    def __init__(self,
                 feature_funcs: List[Feature]) -> None:
        self.feature_funcs = feature_funcs
        self.model = None

        feature_types = self._get_feature_type()
        self.num_numerical_features = np.asarray([x == 'numerical' for x in feature_types], dtype=bool).sum()
        self.num_embedding_features = np.asarray([x == 'embedding' for x in feature_types], dtype=bool).sum()

    def fit(self,
            features_train: np.array,
            labels_train: np.array,
            features_valid: Optional[np.array] = None,
            labels_valid: Optional[np.array] = None,
            num_trials: int = 100,
            save_file: Optional[Union[str, Path]] = None) -> None:
        if save_file is not None and os.path.isfile(save_file):
            model = joblib.load(save_file)
        else:
            study = optuna.create_study(directions=['maximize'])
            study.optimize(lambda x: self._tuning_objective(x, features_train, labels_train, features_valid, labels_valid),
                           n_trials=num_trials,
                           gc_after_trial=True)

            model = self._get_model(
                svd_num_components=study.best_params['svd_num_components'], lr_penalty=study.best_params['lr_penalty']
            )
            model.fit(features_train, labels_train)

            joblib.dump(model, save_file)

        self.model = model

    def predict(self, features: np.array) -> np.array:
        if self.model is not None:
            return self.model.predict(features)
        else:
            raise ValueError('No trained model found. Please run .fit() first.')

    def get_feature_importance(self,
                               top_k: int = 10) -> Dict:
        feature_names = self._get_feature_names()
        feature_type = self._get_feature_type()

        numerical_features = [feature_names[idx] for idx, x in enumerate(feature_type) if x == 'numerical']
        num_numerical_features = len(numerical_features)

        estimators = self.model.steps[-1][1].estimators_

        output = {}

        for estimator, task in zip(estimators, self.TASK_NAMES):
            feature_importance = np.abs(estimator.coef_.squeeze())

            embedding_dim = feature_importance.shape[-1] - num_numerical_features
            reduced_feature_names = numerical_features + embedding_dim * ['EmbeddingFeature']

            sort_idx = feature_importance.argsort()
            feature_importance = feature_importance[sort_idx]
            reduced_feature_names = [reduced_feature_names[x] for x in sort_idx]
            feature_importance, reduced_feature_names = feature_importance[-top_k:], reduced_feature_names[-top_k:]

            output[task] = [feature_importance, reduced_feature_names]

        return output

    def _get_feature_names(self) -> List[str]:
        feature_names = []

        for feature_func in self.feature_funcs:
            feature_names += feature_func.dim * [feature_func.__class__.__name__]

        return feature_names

    def _get_feature_type(self) -> List[str]:
        feature_type = []

        for feature_func in self.feature_funcs:
            feature_type += feature_func.dim * [feature_func.type]

        return feature_type

    def _get_model(self,
                   svd_num_components: Optional[int] = None,
                   lr_penalty: Optional[float] = 1.0):
        feature_types = self._get_feature_type()
        numerical_features = np.asarray([x == 'numerical' for x in feature_types], dtype=bool)
        embedding_features = np.asarray([x == 'embedding' for x in feature_types], dtype=bool)

        if svd_num_components is None:
            svd_num_components = embedding_features.sum()

        feature_pipeline = []

        if numerical_features.sum() > 0:
            feature_pipeline += [('numerical_split', FeatureSplitter(numerical_features))]

        if embedding_features.sum() > 0:
            feature_pipeline += [('embedding_pipeline', Pipeline([
                ('embedding_split', FeatureSplitter(embedding_features)),
                ('embedding_scaler', StandardScaler()),
                ('embeddings_svd', TruncatedSVD(n_components=svd_num_components))
            ]))]

        model = Pipeline([
            ('features', FeatureUnion(feature_pipeline)),
            ('feature_scaler', StandardScaler()),
            ('classifier', MultiOutputClassifier(
                LogisticRegression(C=lr_penalty, penalty='l1', max_iter=300, solver='liblinear', tol=0.1), n_jobs=-1
            ))
        ])

        return model

    def _tuning_objective(self,
                          trial: optuna.Trial,
                          features_train: np.array,
                          labels_train: np.array,
                          features_valid: Optional[np.array] = None,
                          labels_valid: Optional[np.array] = None) -> float:
        model = self._get_model(
            svd_num_components=trial.suggest_int('svd_num_components', 1, max(self.num_embedding_features - 1, 1)),
            lr_penalty=trial.suggest_loguniform('lr_penalty', 0.1, 1e4)
        )
        model.fit(features_train, labels_train)

        if (features_valid is not None) and (labels_valid is not None):
            predictions_valid = model.predict(features_valid)
            score = f1_score(labels_valid, predictions_valid, average='macro')
        else:
            predictions_train = model.predict(features_train)
            score = f1_score(labels_train, predictions_train, average='macro')

        return score


class AdHominem():
    """
        AdHominem describes a Siamese network topology for (binary) authorship verification, also known as pairwise
        (1:1) forensic text comparison. This implementation was developed for the The PAN 2020 authorship verification
        challenge. It represents a hierarchical fusion of two well-known approaches into a single end-to-end learning
        procedure: A deep metric learning framework at the bottom aims to learn a pseudo-metric that maps a document of
        variable length onto a fixed-sized feature vector. At the top, we incorporate a probabilistic layer to perform
        Bayes factor scoring in the learned metric space. As with most deep-learning approaches, the success of the
        proposed architecture depends heavily on the availability of a large collection of text samples with many
        examples of representative variations in writing style. The size of the train set (especially for the small
        dataset) can be increased synthetically by dissembling all predefined document pairs and re-sampling new
        same-author and different-author pairs in each epoch (see helper_functions).

        References:
        [1] Benedikt Boenninghoff, Robert M. Nickel, Steffen Zeiler, Dorothea Kolossa, 'Similarity Learning for
            Authorship Verification in Social Media', IEEE ICASSP 2019.
        [2] Benedikt Boenninghoff, Steffen Hessler, Dorothea Kolossa, Robert M. Nickel 'Explainable Authorship
            Verification in Social Media via Attention-based Similarity Learning', IEEE BigData 2019.
        [3] Benedikt Boenninghoff, Julian Rupp, Dorothea Kolossa, Robert M. Nickel 'Deep Bayes Factor Scoring For
            Authorship Verification', PAN Workshop Notebook at CLEF 2020.

    """

    def __init__(self, hyper_parameters, theta_init, theta_E_init):

        # hyper-parameters
        self.hyper_parameters = hyper_parameters

        # placeholders for input variables
        self.placeholders = self.initialize_placeholders(theta_E_init)

        # batch size
        self.B = 1  # tf.shape(self.placeholders['e_w'])[0]

        # trainable parameters
        self.theta = self.initialize_parameters(theta_init)

        ##########################################
        # document embeddings (feature extraction)
        ##########################################
        with tf.variable_scope('feature_extraction_doc2vec'):
            e_c = self.placeholders['e_c']
            e_w = self.placeholders['e_w']
            N_w = self.placeholders['N_w']
            N_s = self.placeholders['N_s']
            # doc2vec
            e_d = self.feature_extraction(e_c, e_w, N_w, N_s)

        #################
        # metric learning
        #################
        with tf.variable_scope('metric_learning'):
            self.emb = self.metric_layer(e_d, self.theta['metric'])

    ###############################
    # MLP layer for metric learning
    ###############################
    def metric_layer(self, e_d, theta):

        # fully-connected layer
        y = tf.nn.xw_plus_b(e_d,
                            theta['W'],
                            theta['b'],
                            )
        # nonlinear output
        y = tf.nn.tanh(y)

        return y

    ################################################
    # initialize all placeholders and look-up tables
    ################################################
    def initialize_placeholders(self, theta_E):

        T_c = self.hyper_parameters['T_c']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']
        D_c = self.hyper_parameters['D_c']
        D_w = self.hyper_parameters['D_w']

        # input character placeholder
        x_c = tf.placeholder(dtype=tf.int32,
                             shape=[None, T_s, T_w, T_c],
                             name='x_c',
                             )

        # initialize embedding matrix for characters
        with tf.variable_scope('character_embedding_matrix'):
            # <PAD> embedding
            E_c_0 = tf.zeros(shape=[1, D_c], dtype=tf.float32)
            # trainable embeddings
            E_c_1 = tf.Variable(theta_E['E_c_1'],
                                name='E_c_1',
                                dtype=tf.float32,
                                trainable=False,
                                )
            # concatenate zero-padding embedding + trained character embeddings
            E_c = tf.concat([E_c_0, E_c_1], axis=0)

        # character embeddings, shape=[B, T_s, T_w, T_c, D_c]
        e_c = tf.nn.embedding_lookup(E_c, x_c)

        # word-based placeholder for two documents
        x_w = tf.placeholder(dtype=tf.int32, shape=[None, T_s, T_w], name='x_w')

        # true sentence / document lengths
        N_w = tf.placeholder(dtype=tf.int32, shape=[None, T_s], name='N_w')
        N_s = tf.placeholder(dtype=tf.int32, shape=[None], name='N_s')

        # matrix for word embeddings, shape=[len(V_w), D_w]
        with tf.variable_scope('word_embedding_matrix'):
            # <PAD> embedding
            E_w_0 = tf.zeros(shape=[1, D_w], dtype=tf.float32)
            # <UNK> embedding
            E_w_1 = tf.Variable(theta_E['E_w_1'],
                                name='E_w_1',
                                trainable=False,
                                dtype=tf.float32,
                                )
            # pre-trained word embedding
            E_w_2 = tf.Variable(theta_E['E_w_2'],
                                name='E_w_2',
                                trainable=False,
                                dtype=tf.float32,
                                )
            # concatenate special-token embeddings + regular-token embeddings
            E_w = tf.concat([E_w_0, E_w_1, E_w_2], axis=0)

        # word embeddings, shape=[B, T_s, T_w, D_w]
        e_w = tf.nn.embedding_lookup(E_w, x_w)

        #############
        # make tuples
        #############
        placeholders = {'x_c': x_c,
                        'e_c': e_c,
                        #
                        'x_w': x_w,
                        'e_w': e_w,
                        #
                        'N_w': N_w,
                        'N_s': N_s,
                        }

        return placeholders

    ################################################
    # feature extraction: words-to-document encoding
    ################################################
    def feature_extraction(self, e_c, e_w, N_w, N_s):

        with tf.variable_scope('characters_to_word_encoding'):
            r_c = self.cnn_layer_cw(e_c)
        with tf.variable_scope('words_to_sentence_encoding'):
            e_cw = tf.concat([e_w, r_c], axis=3)
            h_w = self.bilstm_layer_ws(e_cw, N_w)
            e_s = self.att_layer_ws(h_w, N_w)
        with tf.variable_scope('sentences_to_document_encoding'):
            h_s = self.bilstm_layer_sd(e_s, N_s)
            e_d = self.att_layer_sd(h_s, N_s)

        return e_d

    ########################################
    # 1D-CNN for characters-to-word encoding
    ########################################
    def cnn_layer_cw(self, e_c):

        T_s = self.hyper_parameters['T_s']
        T_w = self.hyper_parameters['T_w']
        T_c = self.hyper_parameters['T_c']
        h = self.hyper_parameters['w']
        D_c = self.hyper_parameters['D_c']
        D_r = self.hyper_parameters['D_r']

        # zero-padding
        # reshape: [B, T_s, T_w, T_c, D_c] --> [B * T_s * T_w, T_c, D_c]
        e_c = tf.reshape(e_c, shape=[self.B * T_s * T_w, T_c, D_c])

        # zero-padding, shape = [B * T_s * T_w, T_c + 2 * (h-1), D_c]
        e_c = tf.pad(e_c,
                     tf.constant([[0, 0], [h - 1, h - 1], [0, 0]]),
                     mode='CONSTANT',
                     )

        # 1D convolution
        # shape = [B * T_s * T_w, T_c + 2 * (h-1) - h + 1, D_r] = [B * T_s * T_w, T_c + h - 1, D_r]
        r_c = tf.nn.conv1d(e_c,
                           self.theta['cnn']['W'],
                           stride=1,
                           padding='VALID',
                           name='chraracter_1D_cnn',
                           )
        # apply bias term
        r_c = tf.nn.bias_add(r_c, self.theta['cnn']['b'])
        # apply nonlinear function
        r_c = tf.nn.tanh(r_c)

        # max-over-time pooling
        # shape = [B * T_s * T_w, T_c + h - 1, D_r, 1]
        r_c = tf.expand_dims(r_c, 3)
        # max-over-time-pooling, shape = [B * T_s * T_w, 1, D_r, 1]
        r_c = tf.nn.max_pool(r_c,
                             ksize=[1, T_c + h - 1, 1, 1],
                             strides=[1, 1, 1, 1],
                             padding='VALID',
                             )
        # shape = [B * T_s * T_w, D_r]
        r_c = tf.squeeze(r_c)
        #  shape = [B, T_s, T_w, D_r]
        r_c = tf.reshape(r_c, [self.B, T_s, T_w, D_r])

        return r_c

    #############################################
    # BiLSTM layer for words-to-sentence encoding
    #############################################
    def bilstm_layer_ws(self, e_w_f, N_w):

        D_w = self.hyper_parameters['D_w']
        D_r = self.hyper_parameters['D_r']
        D_s = self.hyper_parameters['D_s']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']

        # reshape N_w, shape = [B * T_s]
        N_w = tf.reshape(N_w, shape=[self.B * T_s])
        # reshape input word embeddings, shape = [B * T_s, T_w, D_w + D_r]
        e_w_f = tf.reshape(e_w_f, shape=[self.B * T_s, T_w, D_w + D_r])
        # reverse input sentences
        e_w_b = tf.reverse_sequence(e_w_f, seq_lengths=N_w, seq_axis=1)

        h_0_f = tf.zeros(shape=[self.B * T_s, D_s], dtype=tf.float32)
        h_0_b = tf.zeros(shape=[self.B * T_s, D_s], dtype=tf.float32)
        c_0_f = tf.zeros(shape=[self.B * T_s, D_s], dtype=tf.float32)
        c_0_b = tf.zeros(shape=[self.B * T_s, D_s], dtype=tf.float32)

        t_0 = tf.constant(0, dtype=tf.int32)

        N_w_t = tf.tile(tf.expand_dims(N_w, axis=1), tf.constant([1, T_w], tf.int32))

        states = tf.scan(self.bilstm_cell_ws,
                         [tf.transpose(e_w_f, perm=[1, 0, 2]),
                          tf.transpose(e_w_b, perm=[1, 0, 2]),
                          tf.transpose(N_w_t, perm=[1, 0])],
                         initializer=[h_0_f, h_0_b, c_0_f, c_0_b, t_0],
                         name='lstm_ws_layer',
                         )

        h_f = states[0]
        h_b = states[1]
        h_f = tf.transpose(h_f, perm=[1, 0, 2])
        h_b = tf.transpose(h_b, perm=[1, 0, 2])
        # reverse again backward state
        h_b = tf.reverse_sequence(h_b, seq_lengths=N_w, seq_axis=1)

        # concatenate hidden states, shape=[2 * B * T_s, T_w, 2 * D_s]
        h = tf.concat([h_f, h_b], axis=2)
        # reshape input word embeddings, shape = [2 * B, T_s, T_w, 2 * D_s]
        h = tf.reshape(h, shape=[self.B, T_s, T_w, 2 * D_s])

        return h

    ############################################
    # BiLSTM cell for words-to-sentence encoding
    ############################################
    def bilstm_cell_ws(self, prev, input):

        # input parameters
        h_prev_f = prev[0]
        h_prev_b = prev[1]
        c_prev_f = prev[2]
        c_prev_b = prev[3]
        t = prev[4]

        e_w_f = input[0]
        e_w_b = input[1]
        N_w = input[2]

        # compute next forward states
        h_next_f, c_next_f = self.lstm_cell(e_w_f, h_prev_f, c_prev_f,
                                            self.theta['lstm_ws_forward'],
                                            )
        # compute next backward states
        h_next_b, c_next_b = self.lstm_cell(e_w_b, h_prev_b, c_prev_b,
                                            self.theta['lstm_ws_backward'],
                                            )

        # t < T
        condition = tf.less(t, N_w)

        # copy-through states if t > T
        h_next_f = tf.where(condition, h_next_f, h_prev_f)
        c_next_f = tf.where(condition, c_next_f, c_prev_f)
        h_next_b = tf.where(condition, h_next_b, h_prev_b)
        c_next_b = tf.where(condition, c_next_b, c_prev_b)

        return [h_next_f, h_next_b, c_next_f, c_next_b, tf.add(t, 1)]

    #################################################
    # BiLSTM layer for sentences-to-document encoding
    #################################################
    def bilstm_layer_sd(self, e_s_f, N_s):

        D_d = self.hyper_parameters['D_d']
        T_s = self.hyper_parameters['T_s']

        # reverse input sentences
        e_s_b = tf.reverse_sequence(e_s_f, seq_lengths=N_s, seq_axis=1)

        h_0_f = tf.zeros(shape=[self.B, D_d], dtype=tf.float32)
        h_0_b = tf.zeros(shape=[self.B, D_d], dtype=tf.float32)
        c_0_f = tf.zeros(shape=[self.B, D_d], dtype=tf.float32)
        c_0_b = tf.zeros(shape=[self.B, D_d], dtype=tf.float32)

        t_0 = tf.constant(0, dtype=tf.int32)

        N_s_t = tf.tile(tf.expand_dims(N_s, axis=1), tf.constant([1, T_s], tf.int32))

        states = tf.scan(self.bilstm_cell_sd,
                         [tf.transpose(e_s_f, perm=[1, 0, 2]),
                          tf.transpose(e_s_b, perm=[1, 0, 2]),
                          tf.transpose(N_s_t, perm=[1, 0])],
                         initializer=[h_0_f, h_0_b, c_0_f, c_0_b, t_0],
                         name='lstm_sd_layer',
                         )

        h_f = states[0]
        h_b = states[1]
        h_f = tf.transpose(h_f, perm=[1, 0, 2])
        h_b = tf.transpose(h_b, perm=[1, 0, 2])
        # reverse again backward state
        h_b = tf.reverse_sequence(h_b, seq_lengths=N_s, seq_axis=1)

        # concatenate hidden states, shape=[2 * B, T_s, 2 * D_d]
        h = tf.concat([h_f, h_b], axis=2)

        return h

    ################################################
    # BiLSTM cell for sentences-to-document encoding
    ################################################
    def bilstm_cell_sd(self, prev, input):

        # input parameters
        h_prev_f = prev[0]
        h_prev_b = prev[1]
        c_prev_f = prev[2]
        c_prev_b = prev[3]
        t = prev[4]

        e_s_f = input[0]
        e_s_b = input[1]
        N_s = input[2]

        # compute next forward states
        h_next_f, c_next_f = self.lstm_cell(e_s_f, h_prev_f, c_prev_f,
                                            self.theta['lstm_sd_forward'],
                                            )
        # compute next backward states
        h_next_b, c_next_b = self.lstm_cell(e_s_b, h_prev_b, c_prev_b,
                                            self.theta['lstm_sd_backward'],
                                            )

        # t < T
        condition = tf.less(t, N_s)

        # copy-through states if t > T
        h_next_f = tf.where(condition, h_next_f, h_prev_f)
        c_next_f = tf.where(condition, c_next_f, c_prev_f)
        h_next_b = tf.where(condition, h_next_b, h_prev_b)
        c_next_b = tf.where(condition, c_next_b, c_prev_b)

        return [h_next_f, h_next_b, c_next_f, c_next_b, tf.add(t, 1)]

    ##################
    # single LSTM cell
    ##################
    def lstm_cell(self, e_w, h_prev, c_prev, params):

        W_i = params['W_i']
        U_i = params['U_i']
        b_i = params['b_i']
        W_f = params['W_f']
        U_f = params['U_f']
        b_f = params['b_f']
        W_o = params['W_o']
        U_o = params['U_o']
        b_o = params['b_o']
        W_c = params['W_c']
        U_c = params['U_c']
        b_c = params['b_c']

        # forget
        i_t = tf.sigmoid(tf.matmul(e_w, W_i) + tf.matmul(h_prev, U_i) + b_i)
        # input
        f_t = tf.sigmoid(tf.matmul(e_w, W_f) + tf.matmul(h_prev, U_f) + b_f)
        # new memory
        c_tilde = tf.tanh(tf.matmul(e_w, W_c) + tf.matmul(h_prev, U_c) + b_c)
        # final memory
        c_next = tf.multiply(i_t, c_tilde) + tf.multiply(f_t, c_prev)
        # output
        o_t = tf.sigmoid(tf.matmul(e_w, W_o) + tf.matmul(h_prev, U_o) + b_o)
        # next hidden state
        h_next = tf.multiply(o_t, tf.tanh(c_next))

        return h_next, c_next

    ################################################
    # attention layer for words-to-sentence encoding
    ################################################
    def att_layer_ws(self, h_w, N_w):

        D_s = self.hyper_parameters['D_s']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']

        # shape=[2 * B * T_s * T_w, 2 * D_s]
        scores = tf.reshape(h_w, shape=[self.B * T_s * T_w, 2 * D_s])
        scores = tf.nn.tanh(tf.nn.xw_plus_b(scores,
                                            self.theta['att_ws']['W'],
                                            self.theta['att_ws']['b']))

        # shape=[2 * B * T_s * T_w, 1]
        scores = tf.matmul(scores, self.theta['att_ws']['v'])
        # shape=[2 * B, T_s, T_w]
        scores = tf.reshape(scores, shape=[self.B, T_s, T_w])

        # binary mask, shape = [2 * B, T_s, T_w]
        mask = tf.sequence_mask(N_w, maxlen=T_w, dtype=tf.float32)
        mask = (1.0 - mask) * -5000.0

        # shape = [2 * B, T_s, T_w]
        scores = scores + mask
        scores = tf.nn.softmax(scores, axis=2)

        # expand to shape=[B, T_s, T_w, 1]
        alpha = tf.expand_dims(scores, axis=3)
        # fill up to shape=[B, T_s, T_w, D_s]
        alpha = tf.tile(alpha, tf.stack([1, 1, 1, 2 * D_s]))
        # combine to get sentence representations, shape=[B, T_s, 2 * D_s]
        e_s = tf.reduce_sum(tf.multiply(alpha, h_w), axis=2, keepdims=False)

        return e_s

    ####################################################
    # attention layer for sentences-to-docuemnt encoding
    ####################################################
    def att_layer_sd(self, h_s, N_s):

        D_d = self.hyper_parameters['D_d']
        T_s = self.hyper_parameters['T_s']

        # shape=[2 * B * T_s, 2 * D_d]
        scores = tf.reshape(h_s, shape=[self.B * T_s, 2 * D_d])
        scores = tf.nn.tanh(tf.nn.xw_plus_b(scores,
                                            self.theta['att_sd']['W'],
                                            self.theta['att_sd']['b']))
        # shape=[2 * B * T_s, 1]
        scores = tf.matmul(scores, self.theta['att_sd']['v'])
        # shape=[2 * B, T_s]
        scores = tf.reshape(scores, shape=[self.B, T_s])

        # binary mask, shape = [2 * B, T_s]
        mask = tf.sequence_mask(N_s, maxlen=T_s, dtype=tf.float32)
        mask = (1.0 - mask) * -5000.0

        # shape = [2 * B, T_s]
        scores = scores + mask
        scores = tf.nn.softmax(scores, axis=1)

        # expand to shape=[B, T_s, 1]
        alpha = tf.expand_dims(scores, axis=2)
        # fill up to shape=[B, T_s, 2 * D_d]
        alpha = tf.tile(alpha, tf.stack([1, 1, 2 * D_d]))
        # combine to get doc representations, shape=[B, 2 * D_d]
        e_d = tf.reduce_sum(tf.multiply(alpha, h_s), axis=1, keepdims=False)

        return e_d

    def initialize_parameters(self, theta_init):

        theta = {}

        with tf.variable_scope('theta_cnn'):
            theta['cnn'] = self.initialize_cnn(theta_init['cnn'])

        with tf.variable_scope('theta_lstm_ws_forward'):
            theta['lstm_ws_forward'] = self.initialize_lstm(theta_init['lstm_ws_forward'])
        with tf.variable_scope('theta_lstm_ws_backward'):
            theta['lstm_ws_backward'] = self.initialize_lstm(theta_init['lstm_ws_backward'])

        with tf.variable_scope('theta_lstm_sd_forward'):
            theta['lstm_sd_forward'] = self.initialize_lstm(theta_init['lstm_sd_forward'])
        with tf.variable_scope('theta_lstm_sd_backward'):
            theta['lstm_sd_backward'] = self.initialize_lstm(theta_init['lstm_sd_backward'])

        with tf.variable_scope('theta_att_ws'):
            theta['att_ws'] = self.initialize_att(theta_init['att_ws'])
        with tf.variable_scope('theta_att_sd'):
            theta['att_sd'] = self.initialize_att(theta_init['att_sd'])

        with tf.variable_scope('theta_metric'):
            theta['metric'] = self.initialize_mlp(theta_init['metric'])

        return theta

    def initialize_mlp(self, theta_init):
        theta = {'W': tf.Variable(theta_init['W'],
                                  trainable=False,
                                  dtype=tf.float32,
                                  name='W',
                                  ),
                 'b': tf.Variable(theta_init['b'],
                                  trainable=False,
                                  dtype=tf.float32,
                                  name='b',
                                  ),
                 }
        return theta

    def initialize_cnn(self, theta_init):
        theta = {'W': tf.Variable(theta_init['W'],
                                  trainable=False,
                                  dtype=tf.float32,
                                  name='W',
                                  ),
                 'b': tf.Variable(theta_init['b'],
                                  trainable=False,
                                  dtype=tf.float32,
                                  name='b',
                                  ),
                 }
        return theta

    def initialize_att(self, theta_init):
        theta = {'W': tf.Variable(theta_init['W'],
                                  trainable=False,
                                  dtype=tf.float32,
                                  name='W',
                                  ),
                 'v': tf.Variable(theta_init['v'],
                                  trainable=False,
                                  dtype=tf.float32,
                                  name='v',
                                  ),
                 'b': tf.Variable(theta_init['b'],
                                  trainable=False,
                                  dtype=tf.float32,
                                  name='b',
                                  ),
                 }
        return theta

    def initialize_lstm(self, theta_init):
        theta = {'W_i': tf.Variable(theta_init['W_i'],
                                    trainable=False,
                                    dtype=tf.float32,
                                    name='W_i',
                                    ),
                 'U_i': tf.Variable(theta_init['U_i'],
                                    trainable=False,
                                    dtype=tf.float32,
                                    name='U_i',
                                    ),
                 'b_i': tf.Variable(theta_init['b_i'],
                                    trainable=False,
                                    dtype=tf.float32,
                                    name='b_i',
                                    ),
                 'W_f': tf.Variable(theta_init['W_f'],
                                    trainable=False,
                                    dtype=tf.float32,
                                    name='W_f',
                                    ),
                 'U_f': tf.Variable(theta_init['U_f'],
                                    trainable=False,
                                    dtype=tf.float32,
                                    name='U_f',
                                    ),
                 'b_f': tf.Variable(theta_init['b_f'],
                                    trainable=False,
                                    dtype=tf.float32,
                                    name='b_f',
                                    ),
                 'W_c': tf.Variable(theta_init['W_c'],
                                    trainable=False,
                                    dtype=tf.float32,
                                    name='W_c',
                                    ),
                 'U_c': tf.Variable(theta_init['U_c'],
                                    trainable=False,
                                    dtype=tf.float32,
                                    name='U_c',
                                    ),
                 'b_c': tf.Variable(theta_init['b_c'],
                                    trainable=False,
                                    dtype=tf.float32,
                                    name='b_c',
                                    ),
                 'W_o': tf.Variable(theta_init['W_o'],
                                    trainable=False,
                                    dtype=tf.float32,
                                    name='W_o',
                                    ),
                 'U_o': tf.Variable(theta_init['U_o'],
                                    trainable=False,
                                    dtype=tf.float32,
                                    name='U_o',
                                    ),
                 'b_o': tf.Variable(theta_init['b_o'],
                                    trainable=False,
                                    dtype=tf.float32,
                                    name='b_o',
                                    ),
                 }
        return theta

    ################################################
    # transform document to a tensor with embeddings
    ################################################
    def sliding_window(self, doc):

        T_w = self.hyper_parameters['T_w']
        hop_length = self.hyper_parameters['hop_length']

        tokens = doc.split()
        doc_new = []
        n = 0
        while len(tokens[n:n + T_w]) > 0:
            # split sentence into tokens
            sent_new = ''
            for token in tokens[n: n + T_w]:
                sent_new += token + ' '
            # add to new doc
            doc_new.append(sent_new.strip())
            # update stepsize
            n += hop_length

        return doc_new

    def doc2mat(self, docs):
        T_c = self.hyper_parameters['T_c']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']
        V_c = self.hyper_parameters['V_c']
        V_w = self.hyper_parameters['V_w']

        # batch size
        B = len(docs)
        N_w = np.zeros((B, T_s), dtype=np.int32)
        N_s = np.zeros((B,), dtype=np.int32)

        # word-based tensor, shape = [B, T_s, T_w]
        x_w = np.zeros((B, T_s, T_w), dtype=np.int32)
        # character-based tensor
        x_c = np.zeros((B, T_s, T_w, T_c), dtype=np.int32)

        # current document
        for i, doc in enumerate(docs):

            # apply sliding window to construct sentence like units
            doc = self.sliding_window(doc)
            N_s[i] = len(doc[:T_s])
            # current sentence
            for j, sentence in enumerate(doc[:T_s]):
                tokens = sentence.split()
                N_w[i, j] = len(tokens)
                # current tokens
                for k, token in enumerate(tokens):
                    if token in V_w:
                        x_w[i, j, k] = V_w[token]
                    else:
                        x_w[i, j, k] = V_w['<UNK>']
                    # current character
                    for l, chr in enumerate(token[:T_c]):
                        if chr in V_c:
                            x_c[i, j, k, l] = V_c[chr]
                        else:
                            x_c[i, j, k, l] = V_c['<UNK>']

        return x_w, N_w, N_s, x_c

    ###################################################
    # train siamese network (without data augmentation)
    ###################################################
    def inference(self, docs, sess):

        # word / character embeddings
        x_w, N_w, N_s, x_c = self.doc2mat(docs)

        # compute pred
        emb = sess.run(self.emb,
                       feed_dict={self.placeholders['x_w']: x_w,
                                  self.placeholders['x_c']: x_c,
                                  self.placeholders['N_w']: N_w,
                                  self.placeholders['N_s']: N_s,
                                  })
        return emb
