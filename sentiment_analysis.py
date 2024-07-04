import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


dataset = pd.read_csv("train.csv")
testset = pd.read_csv("test.csv")
dataset.dropna(inplace=True)
dataset.reset_index(drop=True, inplace=True)

testset.dropna(inplace=True)
testset.reset_index(drop=True, inplace=True)
#print("Dataset preview :")
dataset.head()
#print("Testset preview :")
testset.head()
#print("Dataset dimensions: ", dataset.shape)
#print(dataset.info())

#print("\r")

#print("Testset dimensions: ", testset.shape)
#print(dataset.info())
corpus_d = dataset['text']
corpus_t = testset['text']
corpus_d.head()
def removeLinks(corpus):
    for i in range(0, len(corpus)) : 
        corpus[i] = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", corpus[i])
    return corpus
corpus_d = removeLinks(corpus_d)
corpus_t = removeLinks(corpus_t)
# Stopwords
stop_words = set(stopwords.words('english'))
stop_words.remove('not')

# Lemmatizer
lm = WordNetLemmatizer()

def performOperations(inStream):
    
    # Lower the string
    outStream = str.lower(inStream)
        # Only keep alphabets
    outStream = re.sub('[^a-zA-Z]', ' ', outStream) 
    
    # Remove the stopwords from stream
    outStream = ' '.join([word for word in outStream.split() if word not in stop_words])

    # Lemmatize
    outStream = ' '.join([lm.lemmatize(word, pos='v') for word in outStream.split()])
    
    return outStream
# Perform Operations on dataset
filtered_corpus_d = []
for i in range(len(corpus_d)):
    filtered_corpus_d.append( performOperations(corpus_d[i]))
    
# Perform Operations on testset
filtered_corpus_t = []
for i in range(len(corpus_t)):
    filtered_corpus_t.append( performOperations(corpus_t[i]))
def getUniqueWords(sentences):
    
    # Initialize a defaultdict to store word counts
    word_counts = defaultdict(int)
    
    # Tokenize each sentence and update word counts
    for sentence in sentences:
        words = word_tokenize(sentence.lower()) # Convert to lowercase for case insensitivity
        for word in words:
            word_counts[word] += 1
    
    return dict(word_counts)
# Vocabulary

vocabulary_d = getUniqueWords(filtered_corpus_d) 
vocabulary_t = getUniqueWords(filtered_corpus_t)
# Sort the dictionary by values in descending order
vocabulary_d = dict(sorted(vocabulary_d.items(), key=lambda item: item[1], reverse=True))
vocabulary_t = dict(sorted(vocabulary_t.items(), key=lambda item: item[1], reverse=True))
def plotUniqueWords(vocabulary):
    df = pd.DataFrame(vocabulary.items(), columns=['Word', 'Frequency'])
    # Sort the DataFrame by frequency
    df = df[0:50].sort_values(by='Frequency', ascending=False)

    # Plotting with Seaborn
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Word', y='Frequency', data=df, palette=custom_palette)
    plt.xticks(rotation=90)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Word Frequency in the Dictionary')
    plt.tight_layout()
    plt.show()
# Plot dataset vocabulary
#plotUniqueWords(vocabulary_d)
# Create a dataframe
dataframe = pd.DataFrame(filtered_corpus_d, columns=["Sentence"])
testframe = pd.DataFrame(filtered_corpus_t, columns=["Sentence"])

# Append the dependant variable
dataframe["Sentiment"] = dataset["sentiment"]
testframe["Sentiment"] = testset["sentiment"]

# Encode the sentiment variable
label_encoder = LabelEncoder()
dataframe['SentimentEncoded'] = label_encoder.fit_transform(dataframe['Sentiment'])
testframe['SentimentEncoded'] = label_encoder.transform(testframe['Sentiment'])
dataframe

# Using Random Forrest Classifier

clf = Pipeline([
('vectorizer_tfidf', TfidfVectorizer()),
( 'RandomForest', RandomForestClassifier(n_estimators=20, criterion="entropy", random_state=0))
])

clf.fit(dataframe['Sentence'].tolist(), dataframe['SentimentEncoded'].tolist())

y_pred = clf.predict(testframe['Sentence'].tolist())
#print(classification_report(testframe['SentimentEncoded'].tolist(),y_pred))
def predict_sentiment(sentence):
    # Preprocess the input sentence
    processed_sentence = performOperations(sentence)
    
    # Make prediction
    prediction = clf.predict([processed_sentence])[0]
    
    # Convert numerical prediction back to sentiment label
    sentiment = label_encoder.inverse_transform([prediction])[0]
    
    return sentiment

'''# User interface
while True:
    user_input = input("Enter a sentence (or 'quit' to exit): ")
    
    if user_input.lower() == 'quit':
        break
    
    sentiment = predict_sentiment(user_input)
    print(f"The sentiment of the sentence is: {sentiment}")'''
# ... (your existing code)

# Add the predict_sentiment function and user interface here

'''# Example usage:
print("Sentiment Prediction Tool")
print("-------------------------")
while True:
    user_input = input("Enter a sentence (or 'quit' to exit): ")
    
    if user_input.lower() == 'quit':
        print("Thank you for using the Sentiment Prediction Tool. Goodbye!")
        break
    
    sentiment = predict_sentiment(user_input)
    print(f"The sentiment of the sentence is: {sentiment}")
    print()  # Add a blank line for readability'''
