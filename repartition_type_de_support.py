import boto3
import pandas as pd

s3 = boto3.client("s3")

bucket = "landry"
key = "Challenge 4 - Antennes sur images/Données data.gouv/data/dataset_antennes/SUP_ANTENNE.txt"

obj = s3.get_object(Bucket=bucket, Key=key)

df = pd.read_csv(obj["Body"], sep=";", encoding="utf-8")

print(df.head())