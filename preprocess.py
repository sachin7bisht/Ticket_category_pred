import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  # lemmatizer
from sklearn.feature_extraction import stop_words

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stemmer = WordNetLemmatizer()


def preprocess(text):
    text = re.sub(r"won\'t", "will not", text)  # decontracting the words
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    text = re.sub(r'\W', ' ', str(text))  # Remove all the special characters

    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # remove all single characters 

    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)  # replace all the words except "A-Za-z_" with space

    text = re.sub(r'[^\w\s]', '', text)

    # convert to lower and remove stopwords discard words whose len < 2
    text = ' '.join(e for e in text.split() if e.lower() not in stopwords.words('english') and len(e) > 2)

    text = text.lower().strip()  # strip

    # Lemmatization
    tokens = text.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [word for word in tokens if len(word) > 2]

    new_text = ' '.join(tokens)

    return new_text
