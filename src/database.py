from sqlalchemy import create_engine
from urllib.parse import quote_plus
import pandas as pd

password = quote_plus("piyush@1723")

engine = create_engine(
    f"mysql+mysqlconnector://root:piyush%401723@localhost/sentiment_analysis"
)

df = pd.read_csv(
    "data/raw/twitter_training.csv",
    header=None,
    names=["tweet_id", "entity", "sentiment", "tweet"]
)
df = df.dropna(subset=["tweet"])

print("Rows:", len(df))

df.to_sql(
    "tweets",
    con=engine,
    if_exists="append",
    index=False
)

print(f"{len(df)} rows imported successfully!")