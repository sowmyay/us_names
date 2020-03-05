from names.preprocess import read_data


def preprocess(data_path):
    df = read_data(data_path)
    names = df.drop_duplicates(['state', 'name'], keep='last')
    return names