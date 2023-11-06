import json
import os
import re
import sqlite3
import string

import gensim
import pandas as pd
import pyshorteners
import requests
from bs4 import BeautifulSoup
from gensim import corpora, models
from tabulate import tabulate

"arabalar, motorun oluşturduğu gücün 4 tekere aktarılması sonucu hareket eden araçlardır"
"motor ise piston ve krank milinin, hava yakıt karışımının yanmasıyla oluşan enerjiyle çalışır"
"motordan elde edilen bu hareket enerjisi, şanzuman ve şaft ile diferansiyele aktarılır ve hareket sağlanır"

"mercedes bmw ve audi alman üretimi araba markalarıdır"
"bu markaların çoğu 2. Dünya Savaşında, tank ve uçak üretmişlerdir"
"üretilen bu uçaklar, savaşta çok önemli bir rol almışlardır"


class GoogleSearch:
    def __init__(self):
        self.base_url = "https://www.google.com/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
        }

    def search_google(self, text, num_results=3):
        try:
            params = {"q": text, "num": num_results}
            response = requests.get(self.base_url, headers=self.headers, params=params)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            search_results = soup.find_all("div", class_="tF2Cxc")

            s = pyshorteners.Shortener()

            links = []
            for result in search_results:
                link = result.a.get("href")
                links.append(s.tinyurl.short(link))

            return pd.Series(links, name="Search Results")
        except Exception as e:
            print(f"Error: {e}")
            return pd.Series([], name="Search Results")


class DatabaseHandler:
    def __init__(self, database_file):
        self.database_file = database_file
        self.conn = sqlite3.connect(database_file)
        self.cursor = self.conn.cursor()
        self.create_database()

    def create_database(self):
        self.cursor.execute("PRAGMA foreign_keys=ON")
        self.cursor.execute("BEGIN")
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS lda_results (id INTEGER PRIMARY KEY, input_text TEXT, Dominant_Topic INTEGER, Perc_Contribution REAL, Topic_Keywords TEXT)")
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS google_search_results (id INTEGER PRIMARY KEY, links TEXT)")
        self.cursor.execute("COMMIT")
        self.conn.commit()

    def save_lda_results(self, input_text, dominant_topic, perc_contribution, topic_keywords):
        self.cursor.execute(
            "INSERT INTO lda_results (input_text, Dominant_Topic, Perc_Contribution, Topic_Keywords) VALUES (?, ?, ?, ?)",
            (str(input_text), int(dominant_topic.iloc[0]), float(perc_contribution.iloc[0]),
             str(topic_keywords.head(10)))
        )
        self.conn.commit()

    def save_google_search_results(self, links):
        links_list = links.tolist()
        links_json = json.dumps(links_list)

        self.cursor.execute(
            "INSERT INTO google_search_results (links) VALUES (?)",
            (links_json,)
        )
        self.conn.commit()

    def close_database(self):
        self.conn.close()


def download_file(url, filename):
    if not os.path.isfile(filename):
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(response.text)
            print(f"File downloaded succesfully: {filename}")
        else:
            print("Download Error!", response.status_code)


class TextProcessor:

    def __init__(self):
        url = "https://raw.githubusercontent.com/ahmetaa/zemberek-nlp/master/experiment/src/main/resources/stop-words.tr.txt"
        filename = "stop-words-tr.txt"
        download_file(url, filename)
        stop_words_file = filename
        self.stop_words = self.load_stop_words(stop_words_file)

    @staticmethod
    def load_stop_words(stop_words_file):
        with open(stop_words_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]

    @staticmethod
    def word_tokenize(sentence):
        acronym_each_dot = r"(?:[a-zğçşöüı]\.){2,}"
        acronym_end_dot = r"\b[a-zğçşöüı]{2,3}\."
        suffixes = r"[a-zğçşöüı]{3,}' ?[a-zğçşöüı]{0,3}"
        numbers = r"\d+[.,:\d]+"
        any_word = r"[a-zğçşöüı]+"
        punctuations = r"[a-zğçşöüı]*[.,!?;:]"
        word_regex = "|".join([acronym_each_dot,
                               acronym_end_dot,
                               suffixes,
                               numbers,
                               any_word,
                               punctuations])

        sentence = re.compile("%s" % word_regex, re.I).findall(sentence)
        return sentence

    def initial_clean(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        text = self.word_tokenize(text)
        return text

    def remove_stop_words(self, text):
        stop_words = self.stop_words
        return [word for word in text if word not in stop_words]

    def apply_all(self, text):
        return self.remove_stop_words(self.initial_clean(text))


class LDAModel:

    def __init__(self, lda_model_file):
        self.text_processor = TextProcessor()
        self.query_df = pd.DataFrame(
            columns=['input_text', 'tokenized_texts', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])
        self.dictionary = None
        self.lda_model = None

    def train_lda_model(self, corpus):
        if self.dictionary is not None:
            self.lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=self.dictionary, passes=128,
                                                             update_every=1)
        else:
            print("Cannot train LDA model on an empty corpus")

    def process_input(self, input_text):
        tokenized_text = self.text_processor.apply_all(input_text)
        self.query_df = pd.DataFrame()
        if tokenized_text:
            if self.dictionary is None:
                self.dictionary = corpora.Dictionary([tokenized_text])
            else:
                self.dictionary.add_documents([tokenized_text])
            self.query_df = self.query_df._append({'input_text': input_text, 'tokenized_texts': tokenized_text,
                                                   'Dominant_Topic': None, 'Perc_Contribution': None,
                                                   'Topic_Keywords': None},
                                                  ignore_index=True)
            self.lda_model = None
            if self.lda_model is None:
                self.train_lda_model(corpus=[self.dictionary.doc2bow(tokenized_text)])

    def dominant_topic(self):
        if self.lda_model is not None:
            corpus = [self.dictionary.doc2bow(tokens) for tokens in self.query_df['tokenized_texts'] if tokens]
            sent_topics_df = pd.DataFrame()
            for i, row in enumerate(self.lda_model[corpus]):
                row = sorted(row, key=lambda x: (x[1]), reverse=True)
                for j, (topic_num, prop_topic) in enumerate(row):
                    if j == 0:
                        wp = self.lda_model.show_topic(topic_num, topn=30)

                        all_keywords = ", ".join([word for word, prop in wp])
                        topic_keywods = ", ".join([word for word, prop in wp[:5]])
                        subtopic_keywods = ", ".join([word for word, prop in wp[5:]])

                        sent_topics_df = sent_topics_df._append(
                            pd.Series(
                                [input_text, int(topic_num), round(prop_topic, 4), topic_keywods, subtopic_keywods,
                                 all_keywords,
                                 str(google_search.search_google(all_keywords, num_results=3))]),
                            ignore_index=True)
                    else:
                        break

            sent_topics_df.columns = ['Input_Text', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords',
                                      'Subtopic_Keywords', 'Keywords', 'Links']
            self.query_df = pd.concat([sent_topics_df], axis=1)
        else:
            print("LDA model is not trained yet")


if __name__ == "__main__":
    lda_model_file = 'mOdel.gensim'
    database_file = 'lda_database.db'

    lda_model = LDAModel(lda_model_file)
    db_handler = DatabaseHandler(database_file)
    google_search = GoogleSearch()

    inputs = ''

    while True:
        input_text = input("Text ('q' for quit):")
        if input_text.lower() == 'q':
            db_handler.close_database()
            break

        inputs += " " + input_text

        lda_model.process_input(inputs)
        lda_model.dominant_topic()

        db_handler.save_lda_results(input_text=lda_model.query_df['Input_Text'],
                                    dominant_topic=lda_model.query_df['Dominant_Topic'],
                                    perc_contribution=lda_model.query_df['Perc_Contribution'],
                                    topic_keywords=lda_model.query_df['Keywords'])

        db_handler.save_google_search_results(links=lda_model.query_df['Links'])

        selected_columns = lda_model.query_df[['Topic_Keywords', 'Subtopic_Keywords', 'Links']]
        print(tabulate(selected_columns, headers='keys', maxcolwidths=[None, 50, 50, None]))
