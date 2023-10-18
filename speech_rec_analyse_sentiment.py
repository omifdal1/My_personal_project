import speech_recognition as sr

recognizer = sr.Recognizer()

with sr.Microphone() as mic:
    recognizer.adjust_for_ambient_noise(mic, duration=0.2)
final_text=''
while True:
    try:
        with sr.Microphone() as mic:
            audio = recognizer.listen(mic, timeout=5)  # Enregistrement pendant 5 secondes max
            text = recognizer.recognize_google(audio)
            text = text.lower()
            final_text+=text
            print(text)
            N=int(input("0 pour finir 1 pour continuer"))
            if N==0:  # Vérifier si l'utilisateur dit "fermer"
                break
    except sr.WaitTimeoutError:
        pass  # Ignorer les timeouts
    except sr.UnknownValueError:
        pass  # Ignorer les entrées inintelligibles
    except sr.RequestError as e:
        print(f"Erreur de reconnaissance vocale : {e}")
        break
print(final_text)
print("Merci pour votre speech")


import string
from collections import Counter

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

final_text = final_text.lower()
final_text = final_text.translate(str.maketrans('', '', string.punctuation))

# Using word_tokenize because it's faster than split()
tokenized_words = word_tokenize(final_text, "english")
print(tokenized_words)


# Removing Stop Words
print(stopwords.words('english'))
final_words = []
for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)

# Lemmatization - From plural to single + Base form of a word (example better-> good)
lemma_words = []
for word in final_words:
    word = WordNetLemmatizer().lemmatize(word)
    lemma_words.append(word)

print("lemma: ",lemma_words)

emotion_list = []
with open('c:/Users/hp/Desktop/emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        if word in lemma_words:
            emotion_list.append(emotion)

print(emotion_list)
w = Counter(emotion_list)
print(w)


def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['neg'] > score['pos']:
        print("Negative Sentiment")
    elif score['neg'] < score['pos']:
        print("Positive Sentiment")
    else:
        print("Neutral Sentiment")


sentiment_analyse(final_text)

fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()