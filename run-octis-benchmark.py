import os
import string
from pathlib import Path

from octis.preprocessing.preprocessing import Preprocessing
from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA
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
dataset.save('octis-benchmark-survey_response')

dataset = Dataset()
dataset.load_custom_dataset_from_folder('octis-benchmark-survey_response')

model = LDA(num_topics=25)  # Create model
model_output = model.train_model(dataset) # Train the model

from octis.evaluation_metrics.diversity_metrics import TopicDiversity

metric = TopicDiversity(topk=10) # Initialize metric
topic_diversity_score = metric.score(model_output) # Compute score of the metric
import pdb; pdb.set_trace()