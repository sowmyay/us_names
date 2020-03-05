import numpy as np

from names.preprocess import read_data


def filter_neutral_genders(df):
    names = df.groupby(['name', 'gender'], as_index=False).agg({'number': sum})
    names = names.pivot('name', 'gender', 'number').reset_index().fillna(0)
    names["Mpercent"] = ((names["M"] - names["F"]) / (names["M"] + names["F"]))
    names['gender'] = np.where(names['Mpercent'] > 0.001, 'male', 'female')
    # names.set_index("name", inplace=True)
    return names


def preprocess(data_path):
    df = read_data(data_path)
    return filter_neutral_genders(df)
