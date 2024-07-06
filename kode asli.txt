import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import string
from wordcloud import WordCloud
from nlp_id.lemmatizer import Lemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nlp_id.stopword import StopWord


st.header('Dashboard Analisis Sentimen')

with st.expander('Analisis Berkas CSV'):
    st.subheader('Unggah Berkas')
    upl = st.file_uploader('Unggah berkas CSV')

    def case_folding(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(" \d+", '', text)
        text = text.strip('')
        text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', " ")
        text = text.encode('ascii', 'replace').decode('ascii')
        text = text.translate(str.maketrans(" ", " ", string.punctuation))
        text = re.sub('\s+', ' ', text)
        return text

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    lemmatizer = Lemmatizer()

    stopword = StopWord()

    slang_dictionary = pd.read_csv('https://raw.githubusercontent.com/nikovs/data-science-portfolio/master/topic%20modelling/colloquial-indonesian-lexicon.csv')
    slang_dict = pd.Series(slang_dictionary['formal'].values, index=slang_dictionary['slang']).to_dict()

    def Slangwords(text):
        for word in text.split():
            if word in slang_dict.keys():
                text = text.replace(word, slang_dict[word])
        return text

    def case_folding_final(text):
        text = re.sub(" \d+", '', text)
        text = text.strip()
        return text

    def tokenization(teks):
        text_list = []
        for txt in teks.split(" "):
            text_list.append(txt)
        return text_list

    def preprocess_text(df):
        df['review_processed'] = ''
        for i, row in df.iterrows():
            text = row['review']
            clean_text = case_folding(text)
            clean_text = stemmer.stem(clean_text)
            clean_text = lemmatizer.lemmatize(clean_text)
            clean_text = stopword.remove_stopword(clean_text)
            clean_text = Slangwords(clean_text)
            clean_text = case_folding_final(clean_text)
            clean_text = tokenization(clean_text)
            df.at[i, 'review_processed'] = clean_text
        return df

    def sentiment_analysis_lexicon_indonesia(text, list_positive, list_negative):
        score = 0
        for word in text:
            if word in list_positive:
                score += 1
            if word in list_negative:
                score -= 1
        polarity = ''
        if score > 0:
            polarity = 'positif'
        elif score < 0:
            polarity = 'negatif'
        else:
            polarity = 'netral'
        return score, polarity

    if upl:
        df = pd.read_csv(upl)
        st.subheader('Info Data')
        st.write(df.head())

        st.subheader('Masukan Kamus Kata Positif')
        pos = st.file_uploader('Positive Lexicon')

        st.subheader('Masukan Kamus Kata Negatif')
        neg = st.file_uploader('Negatif Lexicon')

        if pos:
            df_positive = pd.read_csv(pos, header=None)
            list_positive = list(df_positive.loc[:, 0])
        else:
            list_positive = []

        if neg:
            df_negative = pd.read_csv(neg, header=None)
            list_negative = list(df_negative.loc[:, 0])
        else:
            list_negative = []

        df_clean = preprocess_text(df)
        hasil = df_clean['review_processed'].apply(lambda x: sentiment_analysis_lexicon_indonesia(x, list_positive, list_negative))
        hasil = list(zip(*hasil))
        df_clean['skor_polaritas'] = hasil[0]
        df_clean['polaritas'] = hasil[1]

        color = ['#CDFAD5', '#F6FDC3', '#FF8080']
        name = df_clean['polaritas'].unique()
        label = df_clean.polaritas.value_counts()
        explode = (0.05, 0.05, 0.05)
        fig1, ax1 = plt.subplots()
        text_prop = {'family': 'monospace', 'fontsize': 'small', 'fontweight': 'light'}
        ax1.pie(label, explode=explode, labels=name, colors=color, autopct='%1.1f%%',
                shadow=False, startangle=90, textprops=text_prop)
        ax1.axis('equal')
        st.pyplot(fig1)
        st.write(label)
        st.write(df_clean)

        positive_reviews = df_clean[df_clean['polaritas'] == 'positif']
        negative_reviews = df_clean[df_clean['polaritas'] == 'negatif']

        positive_words = ' '.join([' '.join(review) for review in positive_reviews['review_processed']])
        negative_words = ' '.join([' '.join(review) for review in negative_reviews['review_processed']])

        positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_words)
        negative_wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_words)

        fig2, ax2 = plt.subplots()
        ax2.imshow(positive_wordcloud, interpolation='bilinear')
        ax2.axis('off')
        st.subheader("WordCloud Ulasan Positif")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.imshow(negative_wordcloud, interpolation='bilinear')
        ax3.axis('off')
        st.subheader("WordCloud Ulasan Negatif")
        st.pyplot(fig3)

        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False)

        csv = convert_df(df_clean)

        st.download_button(
            label="Unduh data sebagai CSV",
            data=csv,
            file_name='sentimen.csv',
            mime='text/csv',
        )
