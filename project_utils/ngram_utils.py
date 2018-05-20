# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for ngram

"""


def _unigrams(words):
    """
        Input: a list of words, e.g., ["I", "am", "Denny"]
        Output: a list of unigram
    """
    assert type(words) == list
    return words


def _bigrams(words, join_string, skip=0):
    """
       Input: a list of words, e.g., ["I", "am", "Denny"]
       Output: a list of bigram, e.g., ["I_am", "am_Denny"]
       I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for k in range(1,skip+2):
                if i+k < L:
                    lst.append( join_string.join([words[i], words[i+k]]) )
    else:
        # set it as unigram
        lst = _unigrams(words)
    return lst


def _trigrams(words, join_string, skip=0):
    """
       Input: a list of words, e.g., ["I", "am", "Denny"]
       Output: a list of trigram, e.g., ["I_am_Denny"]
       I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in range(L-2):
            for k1 in range(1,skip+2):
                for k2 in range(1,skip+2):
                    if i+k1 < L and i+k1+k2 < L:
                        lst.append( join_string.join([words[i], words[i+k1], words[i+k1+k2]]) )
    else:
        # set it as bigram
        lst = _bigrams(words, join_string, skip)
    return lst


def _fourgrams(words, join_string):
    """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
        Output: a list of trigram, e.g., ["I_am_Denny_boy"]
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 3:
        lst = []
        for i in range(L-3):
            lst.append( join_string.join([words[i], words[i+1], words[i+2], words[i+3]]) )
    else:
        # set it as trigram
        lst = _trigrams(words, join_string)
    return lst


def _uniterms(words):
    return _unigrams(words)


def _biterms(words, join_string):
    """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
        Output: a list of biterm, e.g., ["I_am", "I_Denny", "I_boy", "am_Denny", "am_boy", "Denny_boy"]
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for j in range(i+1,L):
                lst.append( join_string.join([words[i], words[j]]) )
    else:
        # set it as uniterm
        lst = _uniterms(words)
    return lst


def _triterms(words, join_string):
    """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
        Output: a list of triterm, e.g., ["I_am_Denny", "I_am_boy", "I_Denny_boy", "am_Denny_boy"]
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in xrange(L-2):
            for j in xrange(i+1,L-1):
                for k in xrange(j+1,L):
                    lst.append( join_string.join([words[i], words[j], words[k]]) )
    else:
        # set it as biterm
        lst = _biterms(words, join_string)
    return lst


def _fourterms(words, join_string):
    """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy", "ha"]
        Output: a list of fourterm, e.g., ["I_am_Denny_boy", "I_am_Denny_ha", "I_am_boy_ha", "I_Denny_boy_ha", "am_Denny_boy_ha"]
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 3:
        lst = []
        for i in xrange(L-3):
            for j in xrange(i+1,L-2):
                for k in xrange(j+1,L-1):
                    for l in xrange(k+1,L):
                        lst.append( join_string.join([words[i], words[j], words[k], words[l]]) )
    else:
        # set it as triterm
        lst = _triterms(words, join_string)
    return lst


_ngram_str_map = {
    1: "Unigram",
    2: "Bigram",
    3: "Trigram",
    4: "Fourgram",
    5: "Fivegram",
    12: "UBgram",
    123: "UBTgram",
}


def _ngrams(words, ngram, join_string=" "):
    """wrapper for ngram"""
    if ngram == 1:
        return _unigrams(words)
    elif ngram == 2:
        return _bigrams(words, join_string)
    elif ngram == 3:
        return _trigrams(words, join_string)
    elif ngram == 4:
        return _fourgrams(words, join_string)
    elif ngram == 12:
        unigram = _unigrams(words)
        bigram = [x for x in _bigrams(words, join_string) if len(x.split(join_string)) == 2]
        return unigram + bigram
    elif ngram == 123:
        unigram = _unigrams(words)
        bigram = [x for x in _bigrams(words, join_string) if len(x.split(join_string)) == 2]
        trigram = [x for x in _trigrams(words, join_string) if len(x.split(join_string)) == 3]
        return unigram + bigram + trigram


_nterm_str_map = {
    1: "Uniterm",
    2: "Biterm",
    3: "Triterm",
    4: "Fourterm",
    5: "Fiveterm",
}


def _nterms(words, nterm, join_string=" "):
    """wrapper for nterm"""
    if nterm == 1:
        return _uniterms(words)
    elif nterm == 2:
        return _biterms(words, join_string)
    elif nterm == 3:
        return _triterms(words, join_string)
    elif nterm == 4:
        return _fourterms(words, join_string)


###### ngram_feature_creating_functions
def feat_gen_bigrams(train, test, q1_field, q2_field):
    print('generating train bigram')
    train_bigram_q1 = train[q1_field].apply(lambda x: set((ngram._bigrams(nlp._tokenize(x), '_'))))
    train_bigram_q2 = train[q2_field].apply(lambda x: set((ngram._bigrams(nlp._tokenize(x), '_'))))

    print('generating test bigram')
    test_bigram_q1 = test[q1_field].apply(lambda x: set((ngram._bigrams(nlp._tokenize(x), '_'))))
    test_bigram_q2 = test[q2_field].apply(lambda x: set((ngram._bigrams(nlp._tokenize(x), '_'))))

    train_bigram_df = pd.concat([train_bigram_q1, train_bigram_q2], axis=1)
    test_bigram_df = pd.concat([test_bigram_q1, test_bigram_q2], axis=1)

    train_bigram_df.columns = ['q1_bigrams', 'q2_bigrams']
    test_bigram_df.columns = ['q1_bigrams', 'q2_bigrams']

    return train_bigram_df, test_bigram_df


def feat_gen_trigrams(train, test, q1_field, q2_field):
    print('generating train trigram')
    train_trigram_q1 = train[q1_field].apply(lambda x: set((ngram._trigrams(nlp._tokenize(x), '_'))))
    train_trigram_q2 = train[q2_field].apply(lambda x: set((ngram._trigrams(nlp._tokenize(x), '_'))))

    print('generating test trigram')
    test_trigram_q1 = test[q1_field].apply(lambda x: set((ngram._trigrams(nlp._tokenize(x), '_'))))
    test_trigram_q2 = test[q2_field].apply(lambda x: set((ngram._trigrams(nlp._tokenize(x), '_'))))

    train_trigram_df = pd.concat([train_trigram_q1, train_trigram_q2], axis=1)
    test_trigram_df = pd.concat([test_trigram_q1, test_trigram_q2], axis=1)

    train_trigram_df.columns = ['q1_trigrams', 'q2_trigrams']
    test_trigram_df.columns = ['q1_trigrams', 'q2_trigrams']
    return train_trigram_df, test_trigram_df


def feat_gen_bigram_cooc_count(train_bigram_df, test_bigram_df):
    train_bigram_cooc_count = pd.Series(
        [len(set.intersection(*z)) for z in zip(train_bigram_df.q1_bigrams, train_bigram_df.q2_bigrams)])
    test_bigram_cooc_count = pd.Series(
        [len(set.intersection(*z)) for z in zip(test_bigram_df.q1_bigrams, test_bigram_df.q2_bigrams)])
    train_bigram_cooc_count.name = 'bigram_cooc_count'
    test_bigram_cooc_count.name = 'bigram_cooc_count'
    train_bigram_df = pd.concat([train_bigram_df, train_bigram_cooc_count], axis=1)
    test_bigram_df = pd.concat([test_bigram_df, test_bigram_cooc_count], axis=1)

    return train_bigram_df, test_bigram_df


def feat_gen_trigram_cooc_count(train_trigram_df, test_trigram_df):
    train_trigram_cooc_count = pd.Series(
        [len(set.intersection(*z)) for z in zip(train_trigram_df.q1_trigrams, train_trigram_df.q2_trigrams)])
    test_trigram_cooc_count = pd.Series(
        [len(set.intersection(*z)) for z in zip(test_trigram_df.q1_trigrams, test_trigram_df.q2_trigrams)])
    train_trigram_cooc_count.name = 'trigram_cooc_count'
    test_trigram_cooc_count.name = 'trigram_cooc_count'
    train_trigram_df = pd.concat([train_trigram_df, train_trigram_cooc_count], axis=1)
    test_trigram_df = pd.concat([test_trigram_df, test_trigram_cooc_count], axis=1)
    return train_trigram_df, test_trigram_df


def feat_gen_bigram_cooc_ratio(train_bigram_df, test_bigram_df):
    print('generating train bigram coorcrance ratio')
    train_bigram_cooc_ratio = train_bigram_df.apply(lambda x: np_utils._try_divide(x[2], len(x[0]) * len(x[1])), axis=1)
    print('generating test bigram coorcrance ratio')
    test_bigram_cooc_ratio = test_bigram_df.apply(lambda x: np_utils._try_divide(x[2], len(x[0]) * len(x[1])), axis=1)

    train_bigram_cooc_ratio.name = 'bigram_cooc_ratio'
    test_bigram_cooc_ratio.name = 'bigram_cooc_ratio'
    train_bigram_df = pd.concat([train_bigram_df, train_bigram_cooc_ratio], axis=1)
    test_bigram_df = pd.concat([test_bigram_df, test_bigram_cooc_ratio], axis=1)

    return train_bigram_df, test_bigram_df


def feat_gen_trigram_cooc_ratio(train_trigram_df, test_trigram_df):
    print('generating train trigram coorcrance ratio')
    train_trigram_cooc_ratio = train_trigram_df.apply(lambda x: np_utils._try_divide(x[2], len(x[0]) * len(x[1])),
                                                      axis=1)
    print('generating test trigram coorcrance ratio')
    test_trigram_cooc_ratio = test_trigram_df.apply(lambda x: np_utils._try_divide(x[2], len(x[0]) * len(x[1])), axis=1)

    train_trigram_cooc_ratio.name = 'trigram_cooc_ratio'
    test_trigram_cooc_ratio.name = 'trigram_cooc_ratio'
    train_trigram_df = pd.concat([train_trigram_df, train_trigram_cooc_ratio], axis=1)
    test_trigram_df = pd.concat([test_trigram_df, test_trigram_cooc_ratio], axis=1)

    return train_trigram_df, test_trigram_df


def feat_gen_fourgrams(train, test, q1_field, q2_field):
    print('generating train fourgrams')
    train_fourgrams_q1 = train[q1_field].apply(lambda x: set((ngram._fourgrams(nlp._tokenize(x), '_'))))
    train_fourgrams_q2 = train[q2_field].apply(lambda x: set((ngram._fourgrams(nlp._tokenize(x), '_'))))

    print('generating test fourgram')
    test_fourgrams_q1 = test[q1_field].apply(lambda x: set((ngram._fourgrams(nlp._tokenize(x), '_'))))
    test_fourgrams_q2 = test[q2_field].apply(lambda x: set((ngram._fourgrams(nlp._tokenize(x), '_'))))

    train_fourgram_df = pd.concat([train_fourgrams_q1, train_fourgrams_q2], axis=1)
    test_fourgram_df = pd.concat([test_fourgrams_q1, test_fourgrams_q2], axis=1)

    train_fourgram_df.columns = ['q1_fourgrams', 'q2_fourgrams']
    test_fourgram_df.columns = ['q1_fourgrams', 'q2_fourgrams']

    return train_fourgram_df, test_fourgram_df


def feat_gen_fourgram_cooc_count(train_fourgram_df, test_fourgram_df):
    train_fourgram_cooc_count = pd.Series(
        [len(set.intersection(*z)) for z in zip(train_fourgram_df.q1_fourgrams, train_fourgram_df.q2_fourgrams)])
    test_fourgram_cooc_count = pd.Series(
        [len(set.intersection(*z)) for z in zip(test_fourgram_df.q1_fourgrams, test_fourgram_df.q2_fourgrams)])
    train_fourgram_cooc_count.name = 'fourgram_cooc_count'
    test_fourgram_cooc_count.name = 'fourgram_cooc_count'
    train_fourgram_df = pd.concat([train_fourgram_df, train_fourgram_cooc_count], axis=1)
    test_fourgram_df = pd.concat([test_fourgram_df, test_fourgram_cooc_count], axis=1)

    return train_fourgram_df, test_fourgram_df


def feat_gen_fourgram_cooc_ratio(train_fourgram_df, test_fourgram_df):
    print('generating train fourgram coorcrance ratio')
    train_fourgram_cooc_ratio = train_fourgram_df.apply(lambda x: np_utils._try_divide(x[2], len(x[0]) * len(x[1])),
                                                        axis=1)
    print('generating test fourgram coorcrance ratio')
    test_fourgram_cooc_ratio = test_fourgram_df.apply(lambda x: np_utils._try_divide(x[2], len(x[0]) * len(x[1])),
                                                      axis=1)

    train_fourgram_cooc_ratio.name = 'fourgram_cooc_ratio'
    test_fourgram_cooc_ratio.name = 'fourgram_cooc_ratio'
    train_fourgram_df = pd.concat([train_fourgram_df, train_fourgram_cooc_ratio], axis=1)
    test_fourgram_df = pd.concat([test_fourgram_df, test_fourgram_cooc_ratio], axis=1)

    return train_fourgram_df, test_fourgram_df


def feat_gen_ngram_ldiff(train_ngram_df, test_ngram_df, n=2):
    features_name = ''
    if n == 1: features_name = 'unigram_diff'
    if n == 2: features_name = 'bigram_diff'
    if n == 3: features_name = 'trigram_diff'
    if n == 4: features_name = 'fourgram_diff'

    print('generating feature: train_{}'.format(features_name))
    train_ngram_diff = train_ngram_df.apply(lambda x: np.abs(len(x[0]) - len(x[1])), axis=1)
    train_ngram_diff.name = features_name
    print('generating feature: test_{}'.format(features_name))
    test_ngram_diff = test_ngram_df.apply(lambda x: np.abs(len(x[0]) - len(x[1])), axis=1)
    test_ngram_diff.name = features_name
    train_ngram_df = pd.concat([train_ngram_df, train_ngram_diff], axis=1)
    test_ngram_df = pd.concat([test_ngram_df, test_ngram_diff], axis=1)

    return train_ngram_df, test_ngram_df


def feat_gen_ngram_ldiff_ratio(train_ngram_df, test_ngram_df, n=2):
    features_name = ''
    if n == 1: features_name = 'unigram_diff'
    if n == 2: features_name = 'bigram_diff'
    if n == 3: features_name = 'trigram_diff'
    if n == 4: features_name = 'fourgram_diff'

    print('generating feature: train_{}'.format(features_name + '_ratio'))
    train_ngram_diff = train_ngram_df.apply(lambda x: np_utils._try_divide(x[features_name], len(x[0]) * len(x[1])),
                                            axis=1)
    train_ngram_diff.name = features_name + '_ratio'
    print('generating feature: test_{}'.format(features_name + '_ratio'))
    test_ngram_diff = test_ngram_df.apply(lambda x: np_utils._try_divide(x[features_name], len(x[0]) * len(x[1])),
                                          axis=1)
    test_ngram_diff.name = features_name + '_ratio'
    train_ngram_df = pd.concat([train_ngram_df, train_ngram_diff], axis=1)
    test_ngram_df = pd.concat([test_ngram_df, test_ngram_diff], axis=1)

    return train_ngram_df, test_ngram_df


# ----------------------------------------------------------------------------
# How many ngrams of obs are in target?
# Obs: [AB, AB, AB, AC, DE, CD]
# Target: [AB, AC, AB, AD, ED]
# ->
# IntersectCount: 4 (i.e., AB, AB, AB, AC)
# IntersectRatio: 4/6
def feat_gen_ngram_interset_count_ratio(train_ngram_df, test_ngram_df, n=2):
    feature_name = ''
    if n == 1: feature_name = 'unigram_interset'
    if n == 2: feature_name = 'bigram_interset'
    if n == 3: feature_name = 'trigram_interset'
    if n == 4: feature_name = 'fourgram_interset'

    def interset_count(s1, s2):
        s = 0.
        for w1 in s1:
            for w2 in s2:
                if dist._is_str_match(w1, w2, 0.85):
                    s += 1.
                    break
        return s

    print('generating feature {} for train'.format(feature_name + '_count'))
    train_ngram_df_intcount = train_ngram_df.apply(lambda x: interset_count(x[0], x[1]), axis=1)
    train_ngram_df_intcount.name = feature_name + '_count'
    train_ngram_df = pd.concat([train_ngram_df, train_ngram_df_intcount], axis=1)

    print('generating feature {} for train'.format(feature_name + '_ratio'))
    train_ngram_df_intratio = train_ngram_df.apply(
        lambda x: np_utils._try_divide(x[feature_name + '_count'], len(x[0]) + len(x[1])), axis=1)
    train_ngram_df_intratio.name = feature_name + '_ratio'
    train_ngram_df = pd.concat([train_ngram_df, train_ngram_df_intratio], axis=1)

    print('generating feature {} for train'.format(feature_name + '_count'))
    test_ngram_df_intcount = test_ngram_df.apply(lambda x: interset_count(x[0], x[1]), axis=1)
    test_ngram_df_intcount.name = feature_name + '_count'
    test_ngram_df = pd.concat([test_ngram_df, test_ngram_df_intcount], axis=1)
    print('generating feature {} for test'.format(feature_name + '_ratio'))
    test_ngram_df_intratio = test_ngram_df.apply(
        lambda x: np_utils._try_divide(x[test_ngram_df_intcount.name], len(x[0]) + len(x[1])), axis=1)
    test_ngram_df_intratio.name = feature_name + '_ratio'
    test_ngram_df = pd.concat([test_ngram_df, test_ngram_df_intratio], axis=1)
    return train_ngram_df, test_ngram_df


def feat_gen_ngram_unique_count_ratio(train_ngram_df, test_ngram_df, n=2):
    feature_name = ''
    if n == 1: feature_name = 'unigram_unique'
    if n == 2: feature_name = 'bigram_unique'
    if n == 3: feature_name = 'trigram_unique'
    if n == 4: feature_name = 'fourgram_unique'
    print('generating feature {} for train'.format(feature_name + '_count'))
    train_ngram_unique_count = train_ngram_df.apply(lambda x: len(set.symmetric_difference(x[0], x[1])), axis=1)
    train_ngram_unique_count.name = feature_name + '_count'
    train_ngram_df = pd.concat([train_ngram_df, train_ngram_unique_count], axis=1)

    print('generating feature {} for train'.format(feature_name + '_ratio'))
    train_ngram_unique_ratio = train_ngram_df.apply(
        lambda x: np_utils._try_divide(x[train_ngram_unique_count.name], len(x[0]) * len(x[1])), axis=1)
    train_ngram_unique_ratio.name = feature_name + '_ratio'
    train_ngram_df = pd.concat([train_ngram_df, train_ngram_unique_ratio], axis=1)

    print('generating feature {} for test'.format(feature_name + '_count'))
    test_ngram_unique_count = test_ngram_df.apply(lambda x: len(set.symmetric_difference(x[0], x[1])), axis=1)
    test_ngram_unique_count.name = feature_name + '_count'
    test_ngram_df = pd.concat([test_ngram_df, test_ngram_unique_count], axis=1)

    print('generating feature {} for test'.format(feature_name + '_ratio'))
    test_ngram_unique_ratio = test_ngram_df.apply(
        lambda x: np_utils._try_divide(x[test_ngram_unique_count.name], len(x[0]) * len(x[1])), axis=1)
    test_ngram_unique_ratio.name = feature_name + '_ratio'
    test_ngram_df = pd.concat([test_ngram_df, test_ngram_unique_ratio], axis=1)

    return train_ngram_df, test_ngram_df


# intersect positions features
def feat_gen_ngram_interpos_count_ratio(train_ngram_df, test_ngram_df, n=2):
    def _inter_pos_list(obs, target):
        """
            Get the list of positions of obs in target
        """
        pos_list = [0]
        if len(obs) != 0 and len(target) != 0:
            pos_list = [i for i, o in enumerate(obs, start=1) if o in target]
            if len(pos_list) == 0:
                pos_list = [0]
        return pos_list

    def _inter_norm_pos_list(obs, target):
        pos_list = _inter_pos_list(obs, target)
        N = len(obs) + len(target)
        nl = []
        return [np_utils._try_divide(i, N) for i in pos_list]

    def x_describe(pos_list):
        pos_list = np.array(pos_list)
        if len(pos_list) > 0:
            return pos_list.min(), np.median(pos_list), pos_list.mean(), pos_list.max(), pos_list.std()
        else:
            return 0, 0, 0, 0, 0

    feature_name = ''
    if n == 1: feature_name = 'unigram_ip'
    if n == 2: feature_name = 'bigram_ip'
    if n == 3: feature_name = 'trigram_ip'
    if n == 4: feature_name = 'fourgram_ip'
    print('calculating {} features for train'.format(feature_name))
    train_ipl = train_ngram_df.apply(lambda x: _inter_pos_list(x[0], x[1]), axis=1)
    train_ngram_df[feature_name + 'min'], train_ngram_df[feature_name + 'med'], train_ngram_df[feature_name + 'mean'], \
    train_ngram_df[feature_name + 'max'], train_ngram_df[feature_name + 'std'] = zip(*train_ipl.map(x_describe))

    print('calculating {} features for train'.format(feature_name + 'n'))
    train_ipnl = train_ngram_df.apply(lambda x: _inter_norm_pos_list(x[0], x[1]), axis=1)
    train_ngram_df[feature_name + 'nmin'], train_ngram_df[feature_name + 'nmed'], train_ngram_df[
        feature_name + 'nmean'], \
    train_ngram_df[feature_name + 'nmax'], train_ngram_df[feature_name + 'nstd'] = zip(*train_ipnl.map(x_describe))

    print('calculating {} features for test'.format(feature_name))
    test_ipl = test_ngram_df.apply(lambda x: _inter_pos_list(x[0], x[1]), axis=1)
    test_ngram_df[feature_name + 'min'], test_ngram_df[feature_name + 'med'], test_ngram_df[feature_name + 'mean'], \
    test_ngram_df[feature_name + 'max'], test_ngram_df[feature_name + 'std'] = zip(*test_ipl.map(x_describe))

    print('calculating {} features for test'.format(feature_name + 'n'))
    test_ipnl = test_ngram_df.apply(lambda x: _inter_norm_pos_list(x[0], x[1]), axis=1)
    test_ngram_df[feature_name + 'nmin'], test_ngram_df[feature_name + 'nmed'], test_ngram_df[feature_name + 'nmean'], \
    test_ngram_df[feature_name + 'nmax'], test_ngram_df[feature_name + 'nstd'] = zip(*test_ipnl.map(x_describe))
    return train_ngram_df, test_ngram_df


def feat_gen_jaccard_coef(train_ngram_df, test_ngram_df, n=2):
    if n == 1: feature_name = 'unigram_jaccard'
    if n == 2: feature_name = 'bigram_jaccard'
    if n == 3: feature_name = 'trigram_jaccard'
    if n == 4: feature_name = 'fourgram_jaccard'

    print('calculating {} features for train'.format(feature_name))
    train_ngram_jaccard = train_ngram_df.apply(lambda x: dist._jaccard_coef(x[0], x[1]), axis=1)
    train_ngram_jaccard.name = feature_name
    train_ngram_df = pd.concat([train_ngram_df, train_ngram_jaccard], axis=1)

    print('calculating {} features for test'.format(feature_name))
    test_ngram_jaccard = test_ngram_df.apply(lambda x: dist._jaccard_coef(x[0], x[1]), axis=1)
    test_ngram_jaccard.name = feature_name
    test_ngram_df = pd.concat([test_ngram_df, test_ngram_jaccard], axis=1)

    return train_ngram_df, test_ngram_df


def feat_gen_dice(train_ngram_df, test_ngram_df, n=2):
    if n == 1: feature_name = 'unigram_dice'
    if n == 2: feature_name = 'bigram_dice'
    if n == 3: feature_name = 'trigram_dice'
    if n == 4: feature_name = 'fourgram_dice'

    print('calculating {} features for train'.format(feature_name))
    train_ngram_dice = train_ngram_df.apply(lambda x: dist._dice_dist(x[0], x[1]), axis=1)
    train_ngram_dice.name = feature_name
    train_ngram_df = pd.concat([train_ngram_df, train_ngram_dice], axis=1)

    print('calculating {} features for test'.format(feature_name))
    test_ngram_dice = test_ngram_df.apply(lambda x: dist._dice_dist(x[0], x[1]), axis=1)
    test_ngram_dice.name = feature_name
    test_ngram_df = pd.concat([test_ngram_df, test_ngram_dice], axis=1)
    return train_ngram_df, test_ngram_df


def feat_gen_edngram(train_ngram_df, test_ngram_df, n=2):
    def edit_ngram(obj, target):
        val_list = []
        for w1 in obj:
            _val_list = []
            for w2 in target:
                s = dist._edit_dist(w1, w2)
                _val_list.append(s)
            if len(_val_list) == 0:
                _val_list = [config.MISSING_VALUE_NUMERIC]
            val_list.append(_val_list)
        if len(val_list) == 0:
            val_list = [[config.MISSING_VALUE_NUMERIC]]
        return np.array(val_list).max(axis=1).min()

    if n == 1: feature_name = 'unigram_ed'
    if n == 2: feature_name = 'bigram_ed'
    if n == 3: feature_name = 'trigram_ed'
    if n == 4: feature_name = 'fourgram_ed'
    print('calculating {} features for train'.format(feature_name))
    train_edn = train_ngram_df.apply(lambda x: edit_ngram(x[0], x[1]), axis=1)
    train_edn.neme = feature_name
    train_ngram_df = pd.concat([train_ngram_df, train_edn], axis=1)

    print('calculating {} features for test'.format(feature_name))
    test_edn = test_ngram_df.apply(lambda x: edit_ngram(x[0], x[1]), axis=1)
    test_edn.name = feature_name
    test_ngram_df = pd.concat([test_ngram_df, test_edn], axis=1)
    return train_ngram_df, test_ngram_df


def feat_gen_char_ngram(df, m_q1, m_q2, cv_char):
    unigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 1])
    ix_unigrams = np.sort(list(unigrams.values()))
    print('Unigrams:', len(unigrams))
    bigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 2])
    ix_bigrams = np.sort(list(bigrams.values()))
    print('Bigrams: ', len(bigrams))
    trigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 3])
    ix_trigrams = np.sort(list(trigrams.values()))
    print('Trigrams:', len(trigrams))

    v_num = (m_q1[:, ix_unigrams] > 0).minimum((m_q2[:, ix_unigrams] > 0)).sum(axis=1)
    v_den = (m_q1[:, ix_unigrams] > 0).maximum((m_q2[:, ix_unigrams] > 0)).sum(axis=1)
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]
    df['unigram_jaccard'] = v_score

    # We take into account each letter more than once
    v_num = m_q1[:, ix_unigrams].minimum(m_q2[:, ix_unigrams]).sum(axis=1)
    v_den = m_q1[:, ix_unigrams].sum(axis=1) + m_q2[:, ix_unigrams].sum(axis=1)
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]
    df['unigram_all_jaccard'] = v_score

    v_num = m_q1[:, ix_unigrams].minimum(m_q2[:, ix_unigrams]).sum(axis=1)
    v_den = m_q1[:, ix_unigrams].maximum(m_q2[:, ix_unigrams]).sum(axis=1)
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]
    df['unigram_all_jaccard_max'] = v_score

    v_num = (m_q1[:, ix_bigrams] > 0).minimum((m_q2[:, ix_bigrams] > 0)).sum(axis=1)
    v_den = (m_q1[:, ix_bigrams] > 0).maximum((m_q2[:, ix_bigrams] > 0)).sum(axis=1)
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]
    df['bigram_jaccard'] = v_score

    v_num = m_q1[:, ix_bigrams].minimum(m_q2[:, ix_bigrams]).sum(axis=1)
    v_den = m_q1[:, ix_bigrams].sum(axis=1) + m_q2[:, ix_bigrams].sum(axis=1)
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]
    df['bigram_all_jaccard'] = v_score

    v_num = m_q1[:, ix_bigrams].minimum(m_q2[:, ix_bigrams]).sum(axis=1)
    v_den = m_q1[:, ix_bigrams].maximum(m_q2[:, ix_bigrams]).sum(axis=1)
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]
    df['bigram_all_jaccard_max'] = v_score

    v_num = (m_q1 > 0).minimum((m_q2 > 0)).sum(axis=1)
    v_den = (m_q1 > 0).maximum((m_q2 > 0)).sum(axis=1)
    v_den[np.where(v_den == 0)] = 1
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]
    df['trigram_jaccard'] = v_score

    v_num = m_q1.minimum(m_q2).sum(axis=1)
    v_den = m_q1.sum(axis=1) + m_q2.sum(axis=1)
    v_den[np.where(v_den == 0)] = 1
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]
    df['trigram_all_jaccard'] = v_score

    # We take into account each letter more than once
    # Normalize the maximum value, and not the sum
    v_num = m_q1.minimum(m_q2).sum(axis=1)
    v_den = m_q1.maximum(m_q2).sum(axis=1)
    v_den[np.where(v_den == 0)] = 1
    v_score = np.array(v_num.flatten()).astype(np.float32)[0, :] / np.array(v_den.flatten())[0, :]
    df['trigram_all_jaccard_max'] = v_score

    return df


def StatCoocTF(list1, list2):
    val_list = []
    for w1 in list1:
        s = 0.
        for w2 in list2:
            if dist._is_str_match(w1, w2, config.STR_MATCH_THRESHOLD):
                s += 1.
        val_list.append(s)
    if len(val_list) == 0:
        val_list = [config.MISSING_VALUE_NUMERIC]
    return val_list


def feat_gen_StatCoocTF(train_ngram_df, test_ngram_df, ngram_name):
    train_mean_series = pd.Series()
    train_std_series = pd.Series()
    test_mean_series = pd.Series()
    test_std_series = pd.Series()

    train_mean_series.name = ngram_name + '_StatCoocTF_' + 'mean'
    test_mean_series.name = ngram_name + '_StatCoocTF_' + 'mean'
    train_std_series.name = ngram_name + '_StatCoocTF_' + 'std'
    test_std_series.name = ngram_name + '_StatCoocTF_' + 'std'

    for i, row in tqdm(train_ngram_df.iterrows()):
        val_list = np.array(StatCoocTF(row['q1_' + ngram_name], row['q2_' + ngram_name]))
        train_mean_series.set_value(i, val_list.mean())
        train_std_series.set_value(i, val_list.std())

    for i, row in tqdm(test_ngram_df.iterrows()):
        val_list = np.array(StatCoocTF(row['q1_' + ngram_name], row['q2_' + ngram_name]))
        test_mean_series.set_value(i, val_list.mean())
        test_std_series.set_value(i, val_list.std())

    train_df = pd.DataFrame(index=train_ngram_df.index)
    test_df = pd.DataFrame(index=test_ngram_df.index)

    train_df[train_mean_series.name] = train_mean_series
    train_df[train_std_series.name] = train_std_series
    test_df[test_mean_series.name] = test_mean_series
    test_df[test_std_series.name] = test_std_series

    return train_df, test_df


def feat_gen_StatCoocTF_multi_proc(df, ngram_name):
    mean_series = pd.Series()
    std_series = pd.Series()

    mean_series.name = ngram_name + '_StatCoocTF_' + 'mean'
    std_series.name = ngram_name + '_StatCoocTF_' + 'std'
    for i, row in df.iterrows():
        val_list = np.array(StatCoocTF(row['q1_' + ngram_name], row['q2_' + ngram_name]))
        mean_series.set_value(i, val_list.mean())
        std_series.set_value(i, val_list.std())

    rtn_df = pd.DataFrame(index=df.index)
    rtn_df[mean_series.name] = mean_series
    rtn_df[std_series.name] = std_series
    return rtn_df


if __name__ == "__main__":

    text = "I am Denny boy ha"
    words = text.split(" ")

    assert _ngrams(words, 1) == ["I", "am", "Denny", "boy", "ha"]
    assert _ngrams(words, 2) == ["I am", "am Denny", "Denny boy", "boy ha"]
    assert _ngrams(words, 3) == ["I am Denny", "am Denny boy", "Denny boy ha"]
    assert _ngrams(words, 4) == ["I am Denny boy", "am Denny boy ha"]

    assert _nterms(words, 1) == ["I", "am", "Denny", "boy", "ha"]
    assert _nterms(words, 2) == ["I am", "I Denny", "I boy", "I ha", "am Denny", "am boy", "am ha", "Denny boy", "Denny ha", "boy ha"]
    assert _nterms(words, 3) == ["I am Denny", "I am boy", "I am ha", "I Denny boy", "I Denny ha", "I boy ha", "am Denny boy", "am Denny ha", "am boy ha", "Denny boy ha"]
    assert _nterms(words, 4) == ["I am Denny boy", "I am Denny ha", "I am boy ha", "I Denny boy ha", "am Denny boy ha"]
