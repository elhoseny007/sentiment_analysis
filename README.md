sentiment analysis with Deep_learning
📌 Project Overview

This repository contains an end-to-end sentiment analysis pipeline trained on the Sentiment140 dataset (1.6M Tweets).
The model predicts whether a tweet is:

Negative (0)

Neutral (2)

Positive (4)

The implementation covers data preprocessing, visualization, deep learning with GRU/LSTM, and evaluation.

⚙️ Key Features

✅ Language detection: Automatically detect tweet language using langdetect.

✅ Text cleaning: Remove punctuation, special characters, and stopwords for multiple languages.

✅ Normalization: Apply lemmatization to unify words (e.g., running → run).

✅ Visualization:

Top 50 frequent words

WordCloud of most common terms

✅ Deep Learning model: Bidirectional GRU stacked layers with dropout for generalization.

✅ Training strategies:

EarlyStopping (stop training when val_accuracy doesn’t improve)

ReduceLROnPlateau (reduce learning rate on stagnation)

ModelCheckpoint (save best model automatically)

✅ Evaluation: Accuracy, classification report, confusion matrix.

📂 Dataset

📌 Sentiment140 dataset – 1.6M labeled tweets.

Dataset file:

training.1600000.processed.noemoticon.csv


Columns after renaming in preprocessing:

Target → Sentiment label (0 = Negative, 2 = Neutral, 4 = Positive)

Ids → Tweet ID

Date → Timestamp of tweet

Flag → Query info (unused)

User → Username of the tweet

Text → Actual tweet

🛠️ Installation

Clone repository:

git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis


Install dependencies:

pip install -r requirements.txt


Or install core libraries:

pip install langdetect gensim wordcloud spacy plotly seaborn nltk tensorflow keras


Download NLTK resources before running:

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

🔄 Preprocessing Pipeline

Steps performed in code:

Remove Punctuation

tweet = re.sub(r'[^\w\s]', '', tweet)


Language Detection + Stopword Removal

lang = detect(tweet)
if lang in all_stopwords:
    tweet = " ".join([w for w in tweet.split() if w.lower() not in all_stopwords[lang]])


Lemmatization (normalize words)

lemmatizer = WordNetLemmatizer()
df['Clean_Text'] = df['Clean_Text'].apply(lambda t: " ".join([lemmatizer.lemmatize(w) for w in t.split()]))


Tokenization & Padding

Use Tokenizer and pad_sequences to prepare inputs.

Maximum length: 50 tokens per tweet.

🧠 Model Architecture

Built using Keras Functional API:

Embedding Layer → Learn dense representations of words

Bidirectional GRU Layer (256 units) → Context from both directions

Dropout (0.3) → Prevent overfitting

Bidirectional GRU Layer (128 units)

Dense Layer (256 units, ReLU)

Output Layer (Softmax) → 3 classes (Negative, Neutral, Positive)

Loss: categorical_crossentropy
Optimizer: Adam

🚀 Training
history = model.fit(
    X_train, y_train,
    batch_size=1024,
    epochs=5,
    validation_split=0.1,
    callbacks=callbacks
)


Callbacks used:

ModelCheckpoint → Save best .keras model

ReduceLROnPlateau → Lower LR on plateau

EarlyStopping → Stop if no val_accuracy improvement

📊 Evaluation
predictions = model.predict(X_test)
print("Test Accuracy:", accuracy_score(true, predicted))
print(classification_report(true, predicted))


Accuracy achieved: ~81%

Outputs a full classification report with precision, recall, F1-score

📈 Visualizations

Top 50 most frequent words in dataset

WordCloud for most common terms

Training curves:

Loss vs Epochs

Accuracy vs Epochs

💾 Saving

Cleaned data saved to:

/content/drive/MyDrive/sentiment_analysis/data_cleaned.csv


Trained model saved to:

/content/drive/MyDrive/sentiment_analysis/sentiment_model.keras

🔮 Future Work

Add Transformer-based models (BERT, RoBERTa, DistilBERT)

Hyperparameter tuning with larger embedding sizes

Deploy as a Flask / FastAPI web app

Integrate real-time sentiment prediction with Twitter API

📝 License

This project is licensed under the MIT License.
