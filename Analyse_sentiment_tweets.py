from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
tweet = """I've got nothing to do today but smile."""

# preprocess tweet
tweet_words=[]
for word in tweet.split():
    if word.startswith('@') and len(word)>1:
        word= '@user'
    elif word.startswith('http'):
        word="http"
    tweet_words.append(word)
tweet_proc=" ".join(tweet_words)
print(tweet_proc)

#load model 
roberta="cardiffnlp/twitter-roberta-base-sentiment"
model=AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer=AutoTokenizer.from_pretrained(roberta)
labeles=['Negative','Neutral','Positive']

#sentiment analysis

encoded_tweet = tokenizer(tweet_proc,return_tensors='pt')

output=model(**encoded_tweet)
scores=softmax(output[0][0].detach().numpy())
print(labeles[np.argmax(scores)])
