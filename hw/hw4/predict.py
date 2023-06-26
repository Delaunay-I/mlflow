#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-y", "--year", required=True,
	help="year for the dataset")
ap.add_argument("-m", "--month", required=True,
	help="month for the dataset")
args = vars(ap.parse_args())


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

year = int(args["year"])
month = int(args["month"])
df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet')
df.head(5)


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


np.std(y_pred)


# Create a ride_id column
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.concat([df['ride_id'], pd.DataFrame(y_pred, columns=["results"])], axis=1)
df_result.to_parquet(
    "model_output.parquet",
    engine='pyarrow',
    compression=None,
    index=False
)

print(f"mean predicted duration: {df['results'].mean}")
