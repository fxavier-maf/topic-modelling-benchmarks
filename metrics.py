import os

import pandas as pd
import numpy as np
from gensim.models import CoherenceModel, LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk


def calculate_coherence_score(docs):
    # todo: check for language based tokenisation
    # texts = [doc.split() for doc in docs]
    texts = [[word for word in simple_preprocess(doc) if word not in set(stopwords.words('english'))] for doc in docs]

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(texts)

    # Create a corpus from the dictionary and tokenized documents
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Set up the LDA model
    lda_model = LdaModel(corpus=corpus, num_topics=2, id2word=dictionary, passes=15)

    # Using the CoherenceModel to determine coherence
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()
    return coherence_score


def make_dir(folder_path):
    import os

    # Create a folder called Octis-data-prep in the current directory
    current_directory = os.getcwd()
    path = os.path.join(current_directory, folder_path)

    # Check if the folder already exists, if not create it
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{folder_path}' created at {path}")
    else:
        print(f"Folder '{folder_path}' already exists at {path}")


class calculateOctisScores:
    @staticmethod
    def make_dataset(**kwargs):
        data_path = kwargs['source_datapath']
        text_column = kwargs['text_column_name']
        label_column = kwargs['label_column_name']
        partition_column = "partition"
        data_name = kwargs['data_name']

        df = pd.read_csv(data_path)
        if (text_column not in df) or (label_column not in df):
            print("Please confirm that {text_column} and {label_column}\
                  are both present in the {data_path}")

        df[text_column] = df[text_column].apply(lambda v: " ".join(v.split()))
        df[partition_column] = np.random.choice(['train', 'test', 'val'], size=len(df), p=[0.7, 0.2, 0.1])
        make_dir("Octis-data-prep")
        target_dir = os.sep.join(["Octis-data-prep", data_name])
        make_dir(target_dir)

        # Writing all rows of the data file to a TSV format without the column names
        df[[
            text_column,
            partition_column,
            label_column,
        ]].to_csv(os.path.join(target_dir, 'corpus.tsv'), sep='\t', index=False, header=False)
        df[label_column].to_csv(os.path.join(target_dir, 'labels.txt'), sep='\n', index=False, header=False)
        print(f"Data written to TSV format without column names at {os.path.join(target_dir, 'data.tsv')}")

        words = df['response'].str.split().explode().unique()
        with open(os.path.join(target_dir, 'vocab.txt'), 'w') as file:
            file.write("\n".join(words))
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    responses = pd.read_csv("/Users/fibinse/Downloads/topic-modelling-benchmark-dataset.csv")
    # responses = responses.head(10)
    # metrics = []
    # for topic in responses.main_topic.unique():
    #     docs = responses[responses.main_topic == topic]['response'].tolist()
    #     coherence_score = calculate_coherence_score(docs)
    #     metrics.append({'topic': topic,
    #                     'coherence_score': coherence_score,
    #                     'row_count': len(docs)})

    # metrics_df = pd.DataFrame(metrics)
    # print(metrics_df)
    calculateOctisScores.make_dataset(
        **{
            'source_datapath': "/Users/fibinse/Downloads/topic-modelling-benchmark-dataset.csv",
            'text_column_name': 'response',
            'label_column_name': 'main_topic',
            'data_name': 'survey-response-data'
        }
    )