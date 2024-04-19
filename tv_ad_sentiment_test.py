from transformers import pipeline, AutoTokenizer, DataCollatorWithPadding
import pandas as pd
import numpy as np

# necessary, as csv was not utf 8, and instead windows 1252
# import chardet

# define paths
path_prepared = './rory_tv_ads_text.csv'
path_predictions = './ads_sentiment.csv'

# rawdata = open(path_prepared, 'rb').read()
# result = chardet.detect(rawdata)
# encoding = result['encoding']

# print (encoding)

# import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
   return tokenizer(examples["text"], truncation=True)

# import text
df = pd.read_csv(path_prepared, encoding='Windows-1252')

# test import
# print(df.head())
# print(len(df))
# print(df.dtypes)
# print("done")

# Need to clean as there are floaats
df_clean = df[df['text'].apply(lambda x: isinstance(x, str))]
# print([type(item) for item in text_list if not isinstance(item, str)])

tokenized_test = df_clean.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Turn text into list (probably not necessary anymore)
# text = df_clean['text']
text_list = df_clean['text'].tolist()
# print(df_clean['text'][:10])

# Run sentiment analysis
results = sentiment_pipeline(text_list)
# positive_results = [result for result in results if result['label'] == 'POSITIVE']
# print(positive_results[:10])

# Save results to csv
# results = sentiment_pipeline(text)
# print(type(results))
df_clean['sentiment'] = results
df_clean['sentiment'].to_csv(path_predictions, index=False)