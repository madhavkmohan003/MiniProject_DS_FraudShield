import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    df = df.drop(['Unnamed: 0', 'first', 'last', 'street', 'city', 'state', 'trans_num', 'cc_num'], axis=1)

    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])

    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

    df = df.drop(['trans_date_trans_time', 'dob'], axis=1)

    le = LabelEncoder()
    for col in ['merchant', 'category', 'gender', 'job']:
        df[col] = le.fit_transform(df[col])

    return df