import sys
import os
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pymongo import MongoClient
from typing import List

# Just need this once then delete/comment out
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample training data for content categorization
training_data = [
    ('science', 'The discovery of the Higgs Boson particle was a major milestone in modern physics.'),
    ('technology', 'Artificial Intelligence is transforming the way we interact with technology.'),
    ('sports', 'The soccer team won the championship after a thrilling final match.'),
    ('politics', 'The senator proposed a new bill aiming to reform healthcare.'),
]

# Function to extract keywords
def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    keywords = [word for word, pos in tagged if pos in ('NN', 'NNS', 'NNP', 'NNPS')]
    return keywords[:max_keywords]

# Function to extract sentiment
def extract_sentiment(text: str) -> str:
    sentiment_score = TextBlob(text).sentiment.polarity
    return 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'

# Function to train and predict content category
def content_category(training_data, text: str) -> str:
    categories, texts = zip(*training_data)
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(texts)
    clf = MultinomialNB().fit(X_train, categories)

    X_test = vectorizer.transform([text])
    predicted_category = clf.predict(X_test)[0]

    return predicted_category

# Function to analyze text
def analyze_text(text_list: List[str]) -> List[dict]:
    analyzed_texts = []

    for text in text_list:
        keywords = extract_keywords(text)
        sentiment = extract_sentiment(text)
        category = content_category(training_data, text)

        analyzed_texts.append({
            'text': text[:50],
            'keywords': keywords,
            'sentiment': sentiment,
            'category': category,
        })

    return analyzed_texts

# Function to store analyzed texts in MongoDB
def store_in_mongodb(analyzed_texts: List[dict]):
    client = MongoClient('uri')
    db = client['text_analysis_db']
    collection = db['analyzed_texts']
    collection.insert_many(analyzed_texts)

# Function to read text file
def read_text_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        content = file.read()
    return content

# Main script
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python script.py <file_path>')
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        print(f'Error: {file_path} is not a valid file.')
        sys.exit(1)

    text_content = read_text_file(file_path)
    analyzed_texts = analyze_text([text_content])

    store_in_mongodb(analyzed_texts)
    print('Text analysis completed and stored in MongoDB.')
