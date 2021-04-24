import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv', type=str, help='path to annotation file')
    parser.add_argument('--output', type=str, help='path to output file')
    args = parser.parse_args()
    df_res = pd.DataFrame()
    df = pd.read_csv(args.csv)
    df_res['file'] = df['file']
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(dtype=int)
    for column in df.columns[1:-1]:
        features = df[column].values
        fit = label_encoder.fit_transform(features)
        features = one_hot_encoder.fit_transform(fit.reshape(-1,1))
        df_res[label_encoder.classes_] = features.toarray()
    df_res.to_csv(args.output, index=False)
