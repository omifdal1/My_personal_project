from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Tokenize and encode the text
text = "je suis heureux"
tokens = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')

# Make a prediction
with torch.no_grad():
    outputs = model(**tokens)
    logits = outputs.logits

# Convert logits to probabilities
probs = torch.softmax(logits, dim=1)

# Get the predicted sentiment label
predicted_label = torch.argmax(probs, dim=1)

# Decode the label
sentiments = {0: 'negative', 1: 'neutral', 2: 'positive'}
sentiment = sentiments[predicted_label.item()]
print(f"Sentiment: {sentiment}")
