from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pyttsx3
import numpy as np

# Fonction pour détecter la langue
def detect_language(text):
    # Vous pouvez utiliser une bibliothèque de détection de la langue, par exemple 'langdetect'
    # pour déterminer la langue du texte ici.
    # Vous devrez l'installer via pip si vous ne l'avez pas déjà fait.
    from langdetect import detect
    return detect(text)

tweet = """Je suis heureux"""

# Détecter la langue du tweet
lang = detect_language(tweet)

# Prétraitement du tweet en fonction de la langue détectée
tweet_words = []
if lang == 'en':
    for word in tweet.split():
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    tweet_proc = " ".join(tweet_words)
elif lang == 'fr':
    for word in tweet.split():
        if word.startswith('@') and len(word) > 1:
            word = '@utilisateur'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    tweet_proc = " ".join(tweet_words)
print(tweet_proc)

# Chargement du modèle multilingue
model_name = "bert-base-multilingual-cased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Labels pour la classification de sentiment
labels = ['Negative', 'Neutral', 'Positive']

# Analyse de sentiment
encoded_tweet = tokenizer(tweet_proc, return_tensors='pt', truncation=True, padding=True)
output = model(**encoded_tweet)
scores = softmax(output.logits.detach().numpy())
predicted_label = labels[np.argmax(scores)]
print(f"Sentiment: {predicted_label}")
