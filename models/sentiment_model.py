from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return (tokenizer, model)

def get_sentiment(text, model_tuple):
    tokenizer, model = model_tuple
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.softmax(logits, dim=1).squeeze().tolist()
    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    max_index = scores.index(max(scores))
    return labels[max_index], scores[max_index]
