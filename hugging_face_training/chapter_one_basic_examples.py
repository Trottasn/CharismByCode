import pandas as pd
from transformers import pipeline

def is_positive_sentiment(text):
    """
    hugging-face-supported task - caches related weights
    'text-classification' is model for sentiment analysis and classification
    """
    classifier = pipeline('text-classification')
    # each prediction is a Python dictionary
    outputs = classifier(input_text)
    print(pd.DataFrame(outputs))
    return outputs[0]['label']

def named_entity_recognition(text):
    ner_tagger = pipeline("ner", aggregation_strategy="simple")
    outputs = ner_tagger(text)
    print(pd.DataFrame(outputs))

def question_answering(text):
    reader = pipeline("question-answering")
    question = "Who created earth?"
    outputs = reader(question=question, context=text)
    print(outputs)

def summarization(text):
    summarizer = pipeline("summarization")
    outputs = summarizer(text, clean_up_tokenization_spaces=True)
    print(outputs[0]['summary_text'])

def translation(text):
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-it")
    outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
    print(outputs[0]['translation_text'])
