import string
import os
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import nltk

# Check and download NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# === File paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(BASE_DIR, 'sample.txt')
EMOTION_PATH = os.path.join(BASE_DIR, 'emotion.txt')
OUTPUT_BAR = os.path.join(BASE_DIR, 'emotion_bar.png')
OUTPUT_CLOUD = os.path.join(BASE_DIR, 'wordcloud.png')

# === Read and preprocess text ===
with open(SAMPLE_PATH, encoding='utf-8') as f:
    text = f.read()

lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
tokenized_words = word_tokenize(cleaned_text)

# Remove stopwords and punctuation
final_words = [
    word.strip(string.punctuation)
    for word in tokenized_words
    if word not in stopwords.words('english') and word.strip(string.punctuation) != ''
]

# === Extract emotions ===
emotion_list = []
with open(EMOTION_PATH, 'r', encoding='utf-8') as file:
    for line in file:
        clear_line = line.replace('\n', '').replace('"', '').replace("'", '').strip()
        if ':' in clear_line:
            word, emotion = clear_line.split(':')
            word = word.strip().lower().strip(string.punctuation)
            emotion = emotion.strip()
            if word in final_words:
                emotion_list.append(emotion)

# === Emotion frequency analysis ===
w = Counter(emotion_list)
print("\nEmotion Counts:", w)

# === Sentiment analysis using VADER ===
def sentiment_analyse(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    print("\nSentiment Scores:", score)

    if score['neg'] > score['pos']:
        print("😠 Negative Sentiment")
    elif score['pos'] > score['neg']:
        print("😊 Positive Sentiment")
    else:
        print("😐 Neutral Vibe")

sentiment_analyse(cleaned_text)

# === Emotion Bar Graph ===
if w:
    colors = plt.cm.tab10(range(len(w)))
    fig, ax1 = plt.subplots()
    ax1.bar(w.keys(), w.values(), color=colors)
    fig.autofmt_xdate()
    plt.title('Emotion Analysis')
    plt.tight_layout()
    plt.savefig(OUTPUT_BAR)
    plt.show()
else:
    print("No emotions matched the text.")

# === WordCloud ===
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(final_words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Key Words", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_CLOUD)
plt.show()
