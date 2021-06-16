from dataset import GermEval2021
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


corpus = GermEval2021('data/GermEval21_Toxic_Train.csv', fold=0, seed=1234)
label_dict = corpus.make_label_dictionary()

word_embeddings = [WordEmbeddings('de')]
document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=128)

classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=True)
trainer = ModelTrainer(classifier, corpus)

trainer.train('logs/demo',
              learning_rate=0.1,
              anneal_factor=0.5,
              patience=5,
              mini_batch_size=32,
              max_epochs=10)
