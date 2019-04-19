"""
A simple demo of katehmm
Done by Kate (Shang Ling) HSU, 12 Apr 2018
"""
import katehmm
from nltk.corpus import treebank as corpus
from nltk.tokenize import word_tokenize
#import nltk

train_data = corpus.tagged_sents(tagset='universal')
test_data = word_tokenize("I dreamed a dream in times gone by.")

trainer = katehmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)

ansK = tagger.tag(test_data)
print(ansK)
#ansN = nltk.pos_tag(test_data)
#l = list(set(ansK)-set(ansN))
