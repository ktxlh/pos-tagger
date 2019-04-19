"""
To test the accuracy of katehmm

Method                      Tagset          Accuracy
Test with Traning Data      Universal       93.93%
Test with Traning Data      Penn Treebank   93.25%  
10-Fold Cross Validation    Universal       89.43%
10-Fold Cross Validation    Penn Treebank   83.62%  

"""
import katehmm as hmm
import nltk
from nltk.corpus import treebank as corpus ####
from nltk.tokenize import word_tokenize

def validation(corpus, train_data, test_data, ans_data):
    
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(train_data)

    correct = 0.
    total = 0.
    for sentIndex in range(len(test_data)):
        data = test_data[sentIndex]
        total += len(data)
        ans_tagger = tagger.tag(data)
        for wordIndex in range(len(data)):
            if ans_tagger[wordIndex] == ans_data[sentIndex][wordIndex]:
                correct += 1
    print("Accuracy: %.4f"%(correct/total))
    return (correct, total)

def main():
    k = 10    # k-cross validation
    correctSum = 0.
    totalSum = 0.
    untagged = corpus.sents()
    tagged = corpus.tagged_sents()    # optional parameter: tagset='universal'
    share = int(len(tagged) / k)

    print("####",k,"Fold Cross Validation ####")
    for i in range(k):
        print("Round", i + 1, end = '\t')
        testRange = (i * share, (i + 1) * share)
        test_data = untagged[testRange[0] : testRange[1]]
        train_data = tagged[ :testRange[0]] + tagged[testRange[1]: ]
        ans_data = tagged[testRange[0] : testRange[1]]

        eva = validation(corpus, train_data, test_data, ans_data)
        correctSum += eva[0]
        totalSum += eva[1]
    print("### Average accuracy: %.4f"%(correctSum/totalSum),"###")

main()