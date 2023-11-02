from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext

st.header('Sentiment Analysis')

# Function to get emoji based on sentiment score
def get_sentiment_emoji(score):
    if score >= 0.5:
        return "😄"  # Happy emoji
    elif score <= -0.5:
        return "😞"  # Sad emoji
    else:
        return "😐"  # Neutral emoji

with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        polarity = round(blob.sentiment.polarity, 2)
        st.write('Polarity: ', polarity, get_sentiment_emoji(polarity))
        subjectivity = round(blob.sentiment.subjectivity, 2)
        st.write('Subjectivity: ', subjectivity)

    pre = st.text_input('Clean Text: ')
    if pre:
        cleaned_text = cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                      stopwords=True, lowercase=True, numbers=True, punct=True)
        st.write(cleaned_text)

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if upl:
        df = pd.read_excel(upl)
        del df['Unnamed: 0']
        df['score'] = df['tweets'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head(10))

        @st.cache
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
