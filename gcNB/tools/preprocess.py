import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from names.preprocess import read_data


def filter_neutral_genders(df):
    names = df.groupby(['name', 'gender'], as_index=False).agg({'number': sum})
    names = names.pivot('name', 'gender', 'number').reset_index().fillna(0)
    names["Mpercent"] = ((names["M"] - names["F"]) / (names["M"] + names["F"]))
    names['gender'] = np.where(names['Mpercent'] > 0.001, 'male', 'female')
    names.set_index("name", inplace=True)
    return names


def vectorize(names):
    char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    X = char_vectorizer.fit_transform(names.index)

    # Convert this matrix to Compressed Sparse Column format
    X = X.tocsc()
    Y = (names.gender == 'male').values.astype(np.int)

    return X, Y


def preprocess(data_path):
    df = read_data(data_path)
    names = filter_neutral_genders(df)
    X, Y = vectorize(names)
    return X, Y