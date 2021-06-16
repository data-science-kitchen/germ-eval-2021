from dataset import GermEval2021
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from torch.optim import AdamW


corpus = GermEval2021('data/GermEval21_Toxic_Train.csv', fold=0, seed=1234)
label_dict = corpus.make_label_dictionary()

word_embeddings = [WordEmbeddings('de')]
document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=64)

classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)
trainer = ModelTrainer(classifier, corpus, optimizer=AdamW)

trainer.train('logs/demo',
              learning_rate=0.05,
              anneal_factor=0.5,
              mini_batch_size=32,
              max_epochs=10)
