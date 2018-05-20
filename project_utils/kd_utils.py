import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import pickle
import sys
import re
import unidecode
from contextlib import contextmanager
import time
from multiprocessing import Pool
from functools import partial
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from operator import itemgetter

import keras as ks
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation
from keras import initializers, regularizers, constraints
from gensim.models.wrappers import FastText
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

# NULL_WORD_FILE_PATH = '../feature_data/null_words_v1.csv'
PREPROC_UNCASE = False


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


# PREPROCESSING PART
repl = {
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",

    ":-s": " bad ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "i'll": "i will",
    "its": "it is",
    "it's": "it is",
    "'s": " is",
    "that's": "that is",
    "weren't": "were not",
}


def replacement(text):
    keys = [i for i in repl.keys()]
    arr = str(text).split()
    res = ""
    for item in arr:
        j = str(item).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if item in keys:
            # print("inn")
            item = repl[item]
        res += item + " "
    res = re.sub('[^a-zA-Z ?!]+', '', str(res).lower())
    return res


def check_object_size(obj):
    """
    check the memory size of an object in RAM - used as an utility tool during coding process
    :param obj: an object to be check
    :return: string indicating how bit the object is
    """
    number_of_bytes = sys.getsizeof(obj)
    if number_of_bytes < 0:
        raise ValueError("!!! numberOfBytes can't be smaller than 0 !!!")
    step_to_greater_unit = 1024.
    number_of_bytes = float(number_of_bytes)
    unit = 'bytes'
    if (number_of_bytes / step_to_greater_unit) >= 1:
        number_of_bytes /= step_to_greater_unit
        unit = 'KB'
    if (number_of_bytes / step_to_greater_unit) >= 1:
        number_of_bytes /= step_to_greater_unit
        unit = 'MB'
    if (number_of_bytes / step_to_greater_unit) >= 1:
        number_of_bytes /= step_to_greater_unit
        unit = 'GB'
    if (number_of_bytes / step_to_greater_unit) >= 1:
        number_of_bytes /= step_to_greater_unit
        unit = 'TB'
    precision = 1
    number_of_bytes = round(number_of_bytes, precision)
    return str(number_of_bytes) + ' ' + unit


def load_data(pickle_file):
    """
    load pickle data from file
    :param pickle_file: path of pickle data
    :return: data stored in pickle file
    """
    load_file = open(pickle_file, 'rb')
    data = pickle.load(load_file)
    return data


def pickle_data(path, data, protocol=-1):
    """
    Pickle data to specified file
    :param path: full path of file where data will be pickled to
    :param data: data to be pickled
    :param protocol: pickle protocol, -1 indicate to use the latest protocol
    :return: None
    """
    file = path
    save_file = open(file, 'wb')
    pickle.dump(data, save_file, protocol=protocol)
    save_file.close()


def get_null_words_dict(null_words_file_path):
    nullwords_df_cols = ['nullword', 'replacement']
    nullwords_df = pd.read_csv(null_words_file_path)[nullwords_df_cols]
    nullwords_df = nullwords_df.loc[~pd.isnull(nullwords_df['replacement'])]
    nullwords_dict = {}
    for i, row in nullwords_df.iterrows():
        nullwords_dict[row['nullword']] = row['replacement']
    return nullwords_dict


def fuck_preprocess(text):
    text = str(text).lower()
    text = text.replace('corpsefucking', ' corpse fuck ')
    text = text.replace('fcken', ' fuck ')
    text = text.replace('fckin', ' fuck ')
    text = text.replace('fcking', ' fuck ')
    text = text.replace('fking', ' fuck ')
    text = text.replace('fonkin', ' fuck ')
    text = text.replace('fuckan', ' fuck ')
    text = text.replace('fuckass', ' fuck ass ')
    text = text.replace('fuckbags', ' fuck bags ')
    text = text.replace('fuckedy', ' fuck ')
    text = text.replace('fuckhole', ' fuck hole ')
    text = text.replace('fuckiest', ' fuck ')
    text = text.replace('fuckign', ' fuck ')
    text = text.replace('fuckingabf', ' fuck ')
    text = text.replace('fuckk', ' fuck ')
    text = text.replace('fuckon', ' fuck ')
    text = text.replace('fucksex', ' fuck sex ')
    text = text.replace('fuckstick', ' fuck stick ')
    text = text.replace('fuckwads', ' fuck ')
    text = text.replace('fuckyourself', ' fuck yourself ')
    text = text.replace('fukcing', ' fuck ')
    text = text.replace('fuked', ' fuck ')
    text = text.replace('fuking', ' fuck ')
    text = text.replace('fukkers', ' fucker ')
    text = text.replace('fukyou', ' fuck you ')
    text = text.replace('marcolfuck', ' marcol fuck ')
    text = text.replace('mothafuckin', ' mother fuck ')
    text = text.replace('motherfu', ' mother fuck ')
    text = text.replace('mothjer', ' mother fucker ')
    text = text.replace('ofuck', ' fuck ')
    text = text.replace('shitfuck', ' shit fuck ')
    text = text.replace('wikifuckers', ' fuck ')
    text = text.replace('fu*k', ' fuck ')  # ❤♥
    text = text.replace('fu*king', ' fuck ')
    text = text.replace('fuc*ing', ' fuck ')
    text = text.replace("f**k's", ' fuck ')
    text = text.replace('fucko', ' fuck off ')
    text = text.replace('fuckface', ' fuck face ')
    text = text.replace('fucked', ' fuck ')
    text = text.replace('fucccccckkkkk', ' fuck ')
    text = text.replace('fucked', ' fuck ')
    text = text.replace('fu*ker', ' fucker ')
    text = text.replace('fu*er', ' fucker ')
    text = text.replace('fu**', ' fuck ')
    text = text.replace('fu***', ' fuck ')
    text = text.replace('fuckish', ' fuck ')
    text = text.replace('fuckedy', ' fuck ')
    text = text.replace('fucyour', ' fuck your ')
    text = text.replace('fuc*in', ' fuck ')
    text = text.replace('fuckwad', ' fuck ')
    text = text.replace('fuckers', ' fucker ')
    text = text.replace("fucker's", ' fucker ')
    text = text.replace("fucknuckle", ' fuck nuckle ')
    text = text.replace('fuckin', ' fuck ')
    text = text.replace('fu*khead', ' fuck head ')
    text = text.replace('fuckingn', ' fuck ')
    text = text.replace("f**k's", ' fuck ')
    text = text.replace('fu*kwit', ' fuck ')
    text = text.replace('fucing', ' fuck ')
    text = text.replace('fùck8', ' fuck ')
    text = text.replace('fucck', ' fuck ')
    text = text.replace('fuc*', ' fuck ')
    text = text.replace('fucc', ' fuck ')
    text = text.replace('fuc*ers', ' fucker ')
    text = text.replace('fuc*ers', ' fucker ')
    text = text.replace('ḟucking', ' fuck ')
    text = text.replace('fucxxxk', ' fuck ')
    text = text.replace('fucxxxk', ' fuck ')
    text = text.replace('fuc@wikipedia', ' fuck wikipedia ')
    text = text.replace('fuc@here', ' fuck here ')
    text = text.replace('fukking', ' fuck ')
    text = text.replace('fukkin', ' fuck ')
    text = text.replace('fukker', ' fucker ')
    text = text.replace('fukkers', ' fucker ')
    text = text.replace('fukka', ' fuck ')
    text = text.replace('fukking', ' fuck ')
    text = text.replace('fukkkk', ' fuck ')
    text = text.replace('fuk1ng', ' fuck ')
    text = text.replace('fukkkk', ' fuck ')
    # text = text.replace('*', ' ')
    text = text.replace('fuck', ' fuck ')

    words = text.split()
    words = ' '.join(words)
    return words


def null_word_replace(text, null_words_dict):
    text = str(text).lower()
    for key, value in null_words_dict.items():
        text = text.replace(key, value)

    words = text.split()
    words = ' '.join(words)
    return words


# baste 靖国神社, 支那, 金将軍, bitch, 中華民國萬歲
def swear_word(text):
    text = text.replace('niggawhaa', ' nigger what ')  #
    text = text.replace('niggerloving', ' nigger ')
    text = text.replace('niggertard', ' nigger neighbor ')
    text = text.replace('niggerly', ' nigger ')
    text = text.replace('niggerlover', ' nigger lover ')
    text = text.replace('niggered', ' nigger ')
    text = text.replace('niggerloving ', ' nigger loving ')
    text = text.replace('niggggeeeerrrr', ' nigger ')
    text = text.replace('negro', ' nigger ')
    text = text.replace('niggr', ' nigger ')
    text = text.replace('niggerballs', ' nigger ball ')
    text = text.replace('nigguuuh', ' nigger ')
    text = text.replace('nigg', ' nigger ')
    text = text.replace('niggling', ' nigger ')
    text = text.replace('niggah', ' nigger ')
    text = text.replace('niggaaaazzz', ' nigger ')
    text = text.replace('niggggguuuuhhhh', ' nigger ')
    text = text.replace('niggus', ' nigger ')
    text = text.replace('niggard', ' nigger ')
    text = text.replace('niggas', ' nigger ')
    text = text.replace('niggaz', ' nigger ')
    text = text.replace('niggerkite', ' nigger kite ')
    text = text.replace('nigger', ' nigger ')
    # sh1ts
    text = text.replace('sh1ts', ' shit ')
    text = text.replace('sh1t', ' shit ')
    text = text.replace('shiot', ' shit ')
    text = text.replace('shioty', ' shit ')
    text = text.replace('shitdip', ' shit ')
    text = text.replace('shitler', ' shit ')
    text = text.replace('shitlol', ' shit ')
    text = text.replace('shitush', ' shit ')
    text = text.replace('shoit', ' shit ')
    text = text.replace('shiiiiiiiit', ' shit ')
    text = text.replace('shittest', ' shit ')
    text = text.replace('shiithead', ' shit head ')
    text = text.replace('shiit', ' shit ')
    text = text.replace('gaaaaaaaaayyyyy', ' gay ')
    text = text.replace('animalfucker', ' animal fucker ')
    text = text.replace('assclowns', ' ass clowns ')
    text = text.replace('assface', ' ass face ')
    text = text.replace('asskicked', ' ass kicked ')
    text = text.replace('asswhipe', ' ass whipe ')
    text = text.replace('badassness', ' bad ass ')
    text = text.replace('bicth', ' bitch ')
    text = text.replace('bitchass', ' bitch ass ')
    text = text.replace('bitchmattythewhite', ' bitch matty the white ')
    text = text.replace('bitchmother', ' bitch mother ')
    text = text.replace('bitchs', ' bitch ')
    text = text.replace('boymamas', ' boy mamas ')
    text = text.replace('cuntbag', ' cunt bag ')
    text = text.replace('cuntface', ' cunt face ')
    text = text.replace('cuntfranks', ' cunt franks ')
    text = text.replace('cuntliz', ' cunt liz ')
    text = text.replace('dickbag', ' dick bag ')
    text = text.replace('dickbig', ' dick big ')
    text = text.replace('dickbreath', ' dick breath ')
    text = text.replace('dickbutt', ' dick butt ')
    text = text.replace('dickheaditalic', ' dick head italic ')
    text = text.replace('failepic', ' fail epic ')
    text = text.replace('fatmansanger', ' fat mansanger ')
    text = text.replace('itiot', ' idiot ')
    text = text.replace('itsuck', ' it suck ')
    text = text.replace('jpgsuck', ' pg suck ')
    text = text.replace('kickwars', ' kick wars ')
    text = text.replace('oldlady', ' old lady ')
    text = text.replace('penistown', ' penis ')
    text = text.replace('pensnsnniensnsn', ' penis ')
    text = text.replace('pneis', ' penis ')
    text = text.replace('popsucker', ' pop sucker ')
    text = text.replace('sexcual', ' sexual ')
    text = text.replace('sexsex', ' sex ')
    text = text.replace('suckdickeer', ' suck dick ')
    text = text.replace('suckersyou', ' suck you ')
    text = text.replace('suckipedia', ' suck ')
    text = text.replace('suckish', ' suck ')
    text = text.replace('sucksfrozen', ' suck frozen ')
    text = text.replace('vbutt', ' butt ')
    text = text.replace('wikihomosexuals', ' homosexual ')
    text = text.replace('wikijews', ' jew ')
    text = text.replace('wikipedidiots', ' idiot ')
    text = text.replace('wikipedophiles', ' pedophile ')
    text = text.replace('wikiretards', ' retard ')
    text = text.replace('wikisucks', ' suck ')
    text = text.replace('wikitheclown', ' clown ')
    text = text.replace('wikiwankers', ' wanker ')
    text = text.replace('di*ks', ' dick ')
    text = text.replace('axxxss', ' ass ')
    text = text.replace('a$$hole', ' asshole ')
    text = text.replace('a$$hat', ' ass hat ')
    text = text.replace('a$$es', ' ass ')
    text = text.replace('a$$', ' ass ')
    text = text.replace('dick', ' dick ')
    text = text.replace('支那', ' fuck chinese ')

    words = text.split()
    words = ' '.join(words)
    return words


def preprocess(text):
    # Replace punctuation with tokens so we can use them in our model
    text = str(text).lower()
    text = unidecode.unidecode(text)  # ❉ ✤ ♦ = " ];]-:._, ^) @ % •: ►✔
    text = text.replace('❉', ' ')
    text = text.replace('✤', ' ')
    text = text.replace('♦', ' ')
    text = text.replace('•', ' ')
    text = text.replace('|', ' ')
    text = text.replace('`', ' ')
    text = text.replace('…', ' ')
    text = text.replace('►', ' ')
    text = text.replace('✔', ' check ')
    text = text.replace('<', ' < ')
    text = text.replace('>', ' > ')
    text = text.replace('[', ' [ ')
    text = text.replace(']', ' ] ')
    text = text.replace('{', ' { ')
    text = text.replace('}', ' } ')
    text = text.replace('=', ' = ')
    text = text.replace('+', ' + ')
    text = text.replace('&', ' & ')
    text = text.replace('%', ' % ')
    text = text.replace('$', ' $ ')
    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    text = text.replace('"', ' " ')
    text = text.replace("”", ' ” ')
    text = text.replace("'", " ' ")
    text = text.replace(';', ' ; ')
    text = text.replace('!', ' ! ')
    text = text.replace('?', ' ? ')
    text = text.replace('(', ' ( ')  #
    text = text.replace(')', ' ) ')
    text = text.replace('{', ' { ')  #
    text = text.replace('}', ' } ')
    text = text.replace('-', ' - ')
    text = text.replace('_', ' _ ')
    text = text.replace("\\", ' BLACKSLASH_MARK ')
    text = text.replace("/", ' SLASH_MARK ')
    text = text.replace('?', ' QUESTION_MARK ')
    text = text.replace('\n', ' NEW_LINE ')
    text = text.replace(':', ' COLON ')
    text = text.replace('tr0uble', ' trouble ')
    text = text.replace('0wn', ' own ')
    text = text.replace('encycl0pedia', ' encyclopedia ')
    text = text.replace('순혈주의 ', ' pure blood ')  #
    text = text.replace('谢谢', ' thanks ')
    text = text.replace('很好', ' very good ')
    text = re.sub("\'ve", " have ", text, flags=re.IGNORECASE)
    text = re.sub("n't", " not ", text, flags=re.IGNORECASE)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text, flags=re.IGNORECASE)
    text = re.sub("\'d", " would ", text, flags=re.IGNORECASE)
    text = re.sub("\'ll", " will ", text, flags=re.IGNORECASE)
    words = text.split()
    words = ' '.join(words)
    return words


def clean_str(text):
    # try:
    text = ' '.join([w for w in text.split()])
    if PREPROC_UNCASE:  text = text.lower()
    text = re.sub(u"é", u"e", text)
    text = re.sub(u"ē", u"e", text)
    text = re.sub(u"è", u"e", text)
    text = re.sub(u"ê", u"e", text)
    text = re.sub(u"à", u"a", text)
    text = re.sub(u"â", u"a", text)
    text = re.sub(u"ô", u"o", text)
    text = re.sub(u"ō", u"o", text)
    text = re.sub(u"ü", u"u", text)
    text = re.sub(u"ï", u"i", text)
    text = re.sub(u"ç", u"c", text)
    text = re.sub(u"\u2019", u"'", text)
    text = re.sub(u"\xed", u"i", text)
    text = re.sub(u"w\/", u" with ", text)

    text = re.sub(u"[^a-z0-9]", " ", text)
    text = u" ".join(re.split('(\d+)', text))
    text = re.sub(u"\s+", u" ", text).strip()
    text = re.sub("\'ve", " have ", text, flags=re.IGNORECASE)
    text = re.sub("n't", " not ", text, flags=re.IGNORECASE)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text, flags=re.IGNORECASE)
    text = re.sub("\'d", " would ", text, flags=re.IGNORECASE)
    text = re.sub("\'ll", " will ", text, flags=re.IGNORECASE)
    text = ''.join(text)
    # except:
    #     text = np.NaN
    return text


def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)


# null_words_list = get_null_words_dict(NULL_WORD_FILE_PATH)


def preproc_pipeline(text):
    # null_words_list=get_null_words_dict(NULL_WORD_FILE_PATH)
    res = clean_str(text)
    if PREPROC_UNCASE:
        res = replacement(res)
        res = fuck_preprocess(res)
        res = swear_word(res)
    # res = null_word_replace(res, null_words_list)
    # res = preprocess(res)
    return res


def apply_func(data=None, func=None, axis=0, raw=True):
    """
      @author: Olivier @address: https://www.kaggle.com/ogrellier
      Utility function that allows multi processing
      """
    return data.apply(func, axis=axis, raw=raw)


def apply_func_series(data=None, func=None):
    """
      @author: Olivier @address: https://www.kaggle.com/ogrellier
      Utility function that allows multi processing on series
      """
    return data.apply(func)


def multi_apply_series(df=None, feature=None, func=None, n_jobs=4):
    """
      @author: Olivier @address: https://www.kaggle.com/ogrellier
      Function that creates process pools to perform actions on pandas Series
      """
    p = Pool(n_jobs)
    f_ = p.map(partial(apply_func_series, func=func),
               np.array_split(df[feature], n_jobs))
    f_ = pd.concat(f_, axis=0, ignore_index=True)
    p.close()
    p.join()
    return f_.values


def multi_apply(df=None, feat_list=None, func=None, axis=0, raw=True, n_jobs=4):
    """
      @author: Olivier @address: https://www.kaggle.com/ogrellier
      Function that creates process pools to perform actions on DataFrames
      """
    p = Pool(n_jobs)
    f_ = p.map(partial(apply_func, func=func, axis=axis, raw=raw),
               np.array_split(df[feat_list], n_jobs))
    f_ = pd.concat(f_, axis=0, ignore_index=True)
    p.close()
    p.join()
    return f_.values


##################### Keras utility functions ######################

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        # self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        input_shape = K.int_shape(x)
        features_dim = self.features_dim
        step_dim = input_shape[1]
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b[:input_shape[1]]
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_w(self, x, mask=None):
        input_shape = K.int_shape(x)
        features_dim = self.features_dim
        step_dim = input_shape[1]
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b[:input_shape[1]]
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        return a

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def pair_loss(y_true, y_pred):
    """
    Taken from @CPMP
    in a Porto Seguro discussion
    """
    # Cast y_true to int32
    y_true = tf.cast(y_true, tf.int32)
    # dynamic partition  uses true labels to partition predictions
    # parts[0] contains y_pred values corresponding to y_true=0
    # parts[1] contains y_pred values corresponding to y_true=1
    parts = tf.dynamic_partition(y_pred, y_true, 2)
    y_pos = parts[1]
    y_neg = parts[0]

    # expand_dims :
    # y_pos was of shape [N_pos] and is now of shape [1, N_pos]
    # Ideally y_pos should contain 1's
    y_pos = tf.expand_dims(y_pos, 0)
    # y_neg was of shape [N_neg] and is now of shape [N_neg, 1]
    # Ideally y_neg should contain 0's
    y_neg = tf.expand_dims(y_neg, -1)

    # y_neg - y_pos creates a [N_neg, N_pos] tensor containing all
    # (negative, positive) samples pairs with their corresponding distance
    out = tf.nn.sigmoid(y_neg - y_pos)

    return tf.reduce_mean(out)


def run_model(x_train, y_train, x_valid, y_valid, keras_params, model_func, show_model_summary=False,
              save_weight_only=True):
    fit_batch_size = keras_params['fit_batch_size']
    predict_batch_size = keras_params['predict_batch_size']
    lr_init = keras_params['lr_init']
    nround = keras_params['nround']
    patience = keras_params['patience']
    cv_score = -1
    pred_valid_best = None
    early_stopping_count = 0
    best_model = None
    best_i = -1
    model = model_func()
    if show_model_summary: model.summary()
    for i in range(nround):
        ks.backend.set_value(model.optimizer.lr, lr_init / (i + 1))
        model.fit(x=x_train, y=y_train, validation_data=[x_valid, y_valid],
                  batch_size=fit_batch_size, epochs=1, verbose=1)
        pred_valid = model.predict(x_valid, batch_size=predict_batch_size, verbose=0)

        cv_i = np.mean(roc_auc_score(y_valid, pred_valid, average=None))
        if cv_i > cv_score:
            cv_score = cv_i
            print('best_score', cv_score, '@', f'epoch {i + 1}')
            pred_valid_best = pred_valid
            if save_weight_only:
                model.save_weights('../models/best_weight.k')
            else:
                best_model = ks.models.clone_model(model)
                best_model.set_weights(model.get_weights())
            best_i = i + 1
            early_stopping_count = 0
        else:
            early_stopping_count += 1
            print('early stoping is {}'.format(early_stopping_count))
            if early_stopping_count >= patience:
                print('early stopping...')
                break

    if save_weight_only:
        best_model = model
        best_model.load_weights('../models/best_weight.k')

    return pred_valid_best, best_model, best_i


def run_model_fulldata(x_train, y_train, x_test, keras_params, model_func, show_model_summary=False, inputlist=False):
    fit_batch_size = keras_params['fit_batch_size']
    predict_batch_size = keras_params['predict_batch_size']
    lr_init = keras_params['lr_init']
    nround = keras_params['nround']
    if not inputlist:
        test_pred = np.zeros((x_test.shape[0], 6))
    else:
        test_pred = np.zeros((x_test[0].shape[0], 6))

    model = model_func()
    if show_model_summary: model.summary()
    for i in range(nround):
        ks.backend.set_value(model.optimizer.lr, lr_init / (i + 1))
        model.fit(x=x_train, y=y_train, batch_size=fit_batch_size, epochs=1, verbose=1)
        test_pred = model.predict(x_test, batch_size=predict_batch_size, verbose=0)
    return test_pred


def cross_validate_keras_olivier(x_train, y_train, x_test, kfidx, keras_params, model_func, maxlen,
                                 show_model_summary=False, oof_test=True, inputlist=False, full_train_test=False):
    train_pred = np.zeros((x_train.shape[0], 6))
    test_pred = np.zeros((x_test.shape[0], 6))
    oof_test_pred = np.zeros((x_test.shape[0], 6))

    cv = -1
    model_rounds = []

    input_dict_sub = {
        "start_words": pad_sequences(x_test["sequences"].apply(lambda x: x[:maxlen]), maxlen=maxlen),
        "end_words": pad_sequences(x_test["sequences"].apply(lambda x: x[-maxlen:]), maxlen=maxlen)
    }

    for trn_idx, val_idx in kfidx:
        x_train_kf, x_val_kf = x_train[trn_idx], x_train[val_idx]
        # Now we need to split the input data dict
        input_dict_tra = {
            "start_words": pad_sequences(x_train_kf["sequences"].apply(lambda x: x[:maxlen]), maxlen=maxlen),
            "end_words": pad_sequences(x_train_kf["sequences"].apply(lambda x: x[-maxlen:]), maxlen=maxlen)
        }
        input_dict_val = {
            "start_words": pad_sequences(x_val_kf["sequences"].apply(lambda x: x[:maxlen]), maxlen=maxlen),
            "end_words": pad_sequences(x_val_kf["sequences"].apply(lambda x: x[-maxlen:]), maxlen=maxlen)
        }

        y_train_kf, y_val_kf = y_train[trn_idx], y_train[val_idx]

        val_pred, model, model_round = run_model(input_dict_tra, y_train_kf, input_dict_val, y_val_kf, keras_params,
                                                 model_func, show_model_summary=show_model_summary)

        model_rounds.append(model_round)
        train_pred[val_idx] += val_pred
        cv = np.mean(roc_auc_score(y_train, train_pred, average=None))
        if oof_test:
            oof_test_pred += model.predict(input_dict_sub, batch_size=keras_params['predict_batch_size'], verbose=1)
        K.clear_session()

    print('oof train ROC_AUC is {:.6f}'.format(cv))
    oof_test_pred /= len(kfidx)

    pickle_data('train_pred_intermediate_save.pkl', train_pred)
    pickle_data('oof_test_pred_intermediate_save.pkl', oof_test_pred)

    if full_train_test:
        keras_params['nround'] = int(np.mean(model_rounds))
        print('generating test prediction with {} rounds'.format(keras_params['nround']))
        test_pred = run_model_fulldata(x_train, y_train, x_test, keras_params, model_func,
                                       show_model_summary=show_model_summary, inputlist=inputlist)

    return train_pred, test_pred, oof_test_pred


def cross_validate_keras(x_train, y_train, x_test, kfidx, keras_params, model_func,
                         show_model_summary=False, oof_test=True, inputlist=False, full_train_test=False):
    if not inputlist:
        train_pred = np.zeros((x_train.shape[0], 6))
        test_pred = np.zeros((x_test.shape[0], 6))
        oof_test_pred = np.zeros((x_test.shape[0], 6))
    else:
        train_pred = np.zeros((x_train[0].shape[0], 6))
        test_pred = np.zeros((x_test[0].shape[0], 6))
        oof_test_pred = np.zeros((x_test[0].shape[0], 6))

    cv = -1
    model_rounds = []

    for trn_idx, val_idx in kfidx:
        if not inputlist:
            x_train_kf, x_val_kf = x_train[trn_idx], x_train[val_idx]
        else:
            x_train_kf = []
            x_val_kf = []
            for i in range(len(x_train)):
                x_train_kf.append(x_train[i][trn_idx])
                x_val_kf.append(x_train[i][val_idx])

        y_train_kf, y_val_kf = y_train[trn_idx], y_train[val_idx]

        val_pred, model, model_round = run_model(x_train_kf, y_train_kf, x_val_kf, y_val_kf, keras_params,
                                                 model_func, show_model_summary=show_model_summary)

        model_rounds.append(model_round)
        train_pred[val_idx] += val_pred
        cv = np.mean(roc_auc_score(y_train, train_pred, average=None))
        if oof_test:
            oof_test_pred += model.predict(x_test, batch_size=keras_params['predict_batch_size'], verbose=1)
        K.clear_session()

    print('oof train ROC_AUC is {:.6f}'.format(cv))
    oof_test_pred /= len(kfidx)

    pickle_data('train_pred_intermediate_save.pkl', train_pred)
    pickle_data('oof_test_pred_intermediate_save.pkl', oof_test_pred)

    if full_train_test:
        keras_params['nround'] = int(np.mean(model_rounds))
        print('generating test prediction with {} rounds'.format(keras_params['nround']))
        test_pred = run_model_fulldata(x_train, y_train, x_test, keras_params, model_func,
                                       show_model_summary=show_model_summary, inputlist=inputlist)

    return train_pred, test_pred, oof_test_pred


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)