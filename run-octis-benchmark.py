import os
import string
from pathlib import Path

from octis.preprocessing.preprocessing import Preprocessing
os.chdir(os.path.pardir)

target_folder = Path('/Users/fibinse/testing_ground/topic-modelling-benchmarks/Octis-data-prep/survey-response-data')
# Initialize preprocessing
preprocessor = Preprocessing(vocabulary=None, max_features=None,
                             remove_punctuation=True, punctuation=string.punctuation,
                             lemmatize=True, stopword_list='english',
                             min_chars=1, min_words_docs=0)
# preprocess
dataset = preprocessor.preprocess_dataset(documents_path=str(target_folder.joinpath('corpus.tsv')), 
                                          labels_path=str(target_folder.joinpath('labels.txt')))

# save the preprocessed dataset
dataset.save('survey_response')