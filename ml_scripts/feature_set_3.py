def splittext(x):
    return x.replace('.', ' ').replace(',', ' ').replace(':', ' ').replace(';', ' ').replace('#', ' ').replace('!',
                                                                                                               ' ').split(
        ' ')
    # return x.split(' ')


traindf['qlenchar'] = traindf.question_text.apply(len)
traindf['qlenword'] = traindf.question_text.apply(lambda x: len(splittext(x)))
traindf['alenchar'] = traindf.answer_text.apply(len)
traindf['alenword'] = traindf.answer_text.apply(lambda x: len(splittext(x)))

traindf['difflenchar'] = traindf.qlenchar - traindf.alenchar
traindf['difflenword'] = traindf.qlenword - traindf.alenword

traindf['divlenchar'] = traindf.qlenchar / traindf.alenchar
traindf['divlenword'] = traindf.qlenword / traindf.alenword

traindf['idivlenchar'] = traindf.alenchar / traindf.qlenchar
traindf['idivlenword'] = traindf.alenword / traindf.qlenword

traindf['subreddit_le'] = LabelEncoder().fit_transform(traindf.subreddit)
traindf['qid'] = LabelEncoder().fit_transform(traindf.question_id)

traindf['qdt_dow'] = pd.to_datetime(traindf.question_utc, origin='unix', unit='s').dt.dayofweek
traindf['qdt_hour'] = pd.to_datetime(traindf.question_utc, origin='unix', unit='s').dt.hour

traindf['adt_dow'] = pd.to_datetime(traindf.answer_utc, origin='unix', unit='s').dt.dayofweek
traindf['adt_hour'] = pd.to_datetime(traindf.answer_utc, origin='unix', unit='s').dt.hour

traindf['question_score_l1p'] = np.log1p(traindf.question_score)
traindf['answer_score_l1p'] = np.log1p(traindf.answer_score)

traindf['qboldwords'] = traindf.question_text.apply(lambda x: np.sum(x.isupper() for x in splittext(x) if len(x) > 1))
traindf['aboldwords'] = traindf.answer_text.apply(lambda x: np.sum(x.isupper() for x in splittext(x) if len(x) > 1))