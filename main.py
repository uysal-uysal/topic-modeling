# 05.11.2023 - Mehmet Uysal - Topic Modeling

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


class GoogleSearch:
    def __init__(self):
        self.base_url = "https://www.google.com/search"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
        }

    def search_google(self, text, num_results=3):
        # search related links according to the topic of input text
        try:
            params = {"q": text, "num": num_results}
            response = requests.get(self.base_url, headers=self.headers, params=params)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            search_results = soup.find_all("div", class_="tF2Cxc")

            s = pyshorteners.Shortener()

            # store links
            links = []
            for result in search_results:
                link = result.a.get("href")

                # shorten the links
                links.append(s.tinyurl.short(link))

            return pd.Series(links, name="Search Results")
        except Exception as e:
            print(f"Error: {e}")
            return pd.Series([], name="Search Results")


class DatabaseHandler:
    def __init__(self, database_file):
        # initialize database and database connection
        try:
            self.database_file = database_file
            self.conn = sqlite3.connect(database_file)
            self.cursor = self.conn.cursor()
            self.create_database()
        except sqlite3.Error as e:
            print(f"Database CONNECTION error: {e}")

    def create_database(self):
        # create table for lda_results and google_search_results
        try:
            self.cursor.execute("PRAGMA foreign_keys=ON")
            self.cursor.execute("BEGIN")
            self.cursor.execute(
                "CREATE TABLE IF NOT EXISTS lda_results (id INTEGER PRIMARY KEY, input_text TEXT, Dominant_Topic INTEGER, Perc_Contribution REAL, Topic_Keywords TEXT)")
            self.cursor.execute(
                "CREATE TABLE IF NOT EXISTS google_search_results (id INTEGER PRIMARY KEY, links TEXT)")
            self.cursor.execute("COMMIT")
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database CREATION error: {e}")

    def save_lda_results(self, input_text, dominant_topic, perc_contribution, topic_keywords):
        # save lda_results
        try:
            self.cursor.execute(
                "INSERT INTO lda_results (input_text, Dominant_Topic, Perc_Contribution, Topic_Keywords) VALUES (?, ?, ?, ?)",
                (str(input_text), int(dominant_topic.iloc[0]), float(perc_contribution.iloc[0]),
                 str(topic_keywords.head(10)))
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database SAVE(lda_results) error: {e}")

    def save_google_search_results(self, links):
        # save google_search_results
        try:
            links_list = links.tolist()
            links_json = json.dumps(links_list)

            self.cursor.execute(
                "INSERT INTO google_search_results (links) VALUES (?)",
                (links_json,)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database SAVE(google_search_results) error: {e}")

    def close_database(self):
        # close the database connection
        self.conn.close()


def download_file(url, filename):
    # download function for download required text files
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
        # initialize TextProcessor class
        url = "https://raw.githubusercontent.com/ahmetaa/zemberek-nlp/master/experiment/src/main/resources/stop-words.tr.txt"
        filename = "stop-words-tr.txt"

        # download and load Turkish stop-words.txt
        download_file(url, filename)
        stop_words_file = filename
        self.stop_words = self.load_stop_words(stop_words_file)

    @staticmethod
    def load_stop_words(stop_words_file):
        # read stop-words.tr.txt file
        with open(stop_words_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]

    @staticmethod
    def word_tokenize(sentence):
        # define regular expressions for different types of words and patterns
        acronym_each_dot = r"(?:[a-z]+\.){2,}"
        acronym_end_dot = r"\b[a-z]+\."
        suffixes = r"[a-z]+['’]?[a-z]*"
        numbers = r"\d+[.,:\d]*"
        any_word = r"[a-z]+"
        punctuations = r"[a-z]*[.,!?;:]"

        # combine patterns using the (or) operator
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
        # remove stop-words from input text
        stop_words = self.stop_words
        return [word for word in text if word not in stop_words]

    def apply_all(self, text):
        # process the input
        return self.remove_stop_words(self.initial_clean(text))


class LDAModel:

    def __init__(self):
        # initialize LDAModel class
        self.text_processor = TextProcessor()
        self.query_df = pd.DataFrame(
            columns=['input_text', 'tokenized_texts', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])
        self.dictionary = None
        self.lda_model = None

    def train_lda_model(self, corpus):
        # train lda_model
        if self.dictionary is not None:
            self.lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=self.dictionary, passes=128,
                                                             update_every=1)
        else:
            print("Cannot train LDA model on an empty corpus")

    def process_input(self, input_text):
        # tokenize input text using text_processor
        tokenized_text = self.text_processor.apply_all(input_text)

        # create dataframe for store the query results
        self.query_df = pd.DataFrame()
        if tokenized_text:
            # if dictionary is not created, create with tokenized text
            if self.dictionary is None:
                self.dictionary = corpora.Dictionary([tokenized_text])
            else:
                # if dictionary exists, add tokenized text
                self.dictionary.add_documents([tokenized_text])
            # create a new row to dataframe with input text and tokenized text
            self.query_df = self.query_df._append({'input_text': input_text, 'tokenized_texts': tokenized_text,
                                                   'Dominant_Topic': None, 'Perc_Contribution': None,
                                                   'Topic_Keywords': None},
                                                  ignore_index=True)
            self.lda_model = None

            # if LDA model is not trained, train it with the current tokenized text
            if self.lda_model is None:
                self.train_lda_model(corpus=[self.dictionary.doc2bow(tokenized_text)])

    def dominant_topic(self):
        if self.lda_model is not None:
            # create corpus of document-term frequency
            corpus = [self.dictionary.doc2bow(tokens) for tokens in self.query_df['tokenized_texts'] if tokens]

            # create dataframe for store the query results
            sent_topics_df = pd.DataFrame()

            # iterate through each document's LDA topic distribution
            for i, row in enumerate(self.lda_model[corpus]):
                # sort topics by their contribution to the document
                row = sorted(row, key=lambda x: (x[1]), reverse=True)

                # iterate through the sorted topics
                for j, (topic_num, prop_topic) in enumerate(row):
                    # highest contribution
                    if j == 0:
                        # Get the top words for the dominant topic
                        wp = self.lda_model.show_topic(topic_num, topn=30)

                        all_keywords = ", ".join([word for word, prop in wp])
                        topic_keywords = ", ".join([word for word, prop in wp[:5]])
                        subtopic_keywords = ", ".join([word for word, prop in wp[5:]])

                        # create new row with input text, dominant topic number, contribution, keywords, and links
                        sent_topics_df = sent_topics_df._append(
                            pd.Series(
                                [input_text, int(topic_num), round(prop_topic, 4), topic_keywords, subtopic_keywords,
                                 all_keywords,
                                 str(google_search.search_google(all_keywords, num_results=3))]),
                            ignore_index=True)
                    else:
                        break

            # set column names
            sent_topics_df.columns = ['Input_Text', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords',
                                      'Subtopic_Keywords', 'Keywords', 'Links']

            # update the query dataframe with the dominant topic results
            self.query_df = pd.concat([sent_topics_df], axis=1)
        else:
            print("LDA model is not trained yet")


if __name__ == "__main__":
    database_file = 'lda_database.db'

    lda_model = LDAModel()
    db_handler = DatabaseHandler(database_file)
    google_search = GoogleSearch()

    inputs = ''

    while True:
        # take input from user
        input_text = input("Input Text ('q' for quit):")

        # if input text equals to 'q', end the program and close database connection
        if input_text.lower() == 'q':
            db_handler.close_database()
            break

        inputs += " " + input_text

        lda_model.process_input(inputs)
        lda_model.dominant_topic()

        # save lda_results to SQLite database
        db_handler.save_lda_results(input_text=lda_model.query_df['Input_Text'],
                                    dominant_topic=lda_model.query_df['Dominant_Topic'],
                                    perc_contribution=lda_model.query_df['Perc_Contribution'],
                                    topic_keywords=lda_model.query_df['Keywords'])

        # save google_search_results to SQLite database
        db_handler.save_google_search_results(links=lda_model.query_df['Links'])

        # print the results with selected columns to console
        selected_columns = lda_model.query_df[['Topic_Keywords', 'Subtopic_Keywords', 'Links']]
        print(tabulate(selected_columns, headers='keys', maxcolwidths=[None, 50, 50, None]))
