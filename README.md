# Sentiment & Emotion Analysis in Python

This project analyzes text data to identify emotions and sentiment using:
- Custom emotion dictionary
- VADER sentiment analyzer
- WordCloud and Emotion Bar Visualization

## Features
- Emotion mapping using a detailed dictionary
- Positive/Negative/Neutral sentiment classification
- Word cloud of key words
- Emotion bar graph visualization
- Cleaner outputs 

## Files Included
- `sentiment.py` – Main analysis script
- `sample.txt` – Sample text file for analysis
- `emotion.txt` – Custom emotion dictionary
- `wordcloud.png` & `emotion_bar.png` – Output images
- `requirements.txt` – Required Python libraries

## How to Run
- 1.Install dependencies using requirements.txt
- 2.Place your input text in sample.txt
- 3.Run the script: sentiment.py

## Output
Upon execution, the script will:
- 1.Display sentiment scores using VADER (Positive, Negative, Neutral)
- 2.Count and display the frequency of detected emotions
- 3.Generate:
     emotion_bar.png: A bar graph showing emotion distribution
     wordcloud.png: A word cloud of key non-stop words in the input text
