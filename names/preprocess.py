import pandas as pd


def read_data(datapath):
    df = pd.DataFrame(columns=["state", "gender", "year", "name", "number"])
    for path in datapath.glob('*.TXT'):
        state = pd.read_csv(path, names=["state", "gender", "year", "name", "number"], header=None)
        df = df.append(state)
    df = df.astype({'number': 'int64'})
    return df


