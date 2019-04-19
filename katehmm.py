"""
Part-of-Speech Tagger
In Python3.6.3
Using HMM (Viterbi algorithm)
Done by Kate (Shang Ling) HSU, 11 Apr 2018

Please see demo.py for simple usage.
"""
import math
FLOAT_INF = 1.7e+308
class HiddenMarkovModelTrainer:
    def __init__(self):
        self.symbols = []            # the set of words existing
        self.states = []             # the set of POS tags existing
        self.priors = {'_sum' : 0.}  # initial count     {'TAG' : count, ...}
        self.outputs = {}            # emission count    {'TAG' : {'observation': count, ...}, ...}
        self.transitions = {}        # transition count  {'TAG' : {'nextTAG': count, ...}, ...}
        # all of above record tag '_sum' for further logarithmic probability computation

    def train_supervised(self, train_data): # returns Tagger
        for sentence in train_data:
            for wordIndex in range(len(sentence)):
                word = sentence[wordIndex][0]
                tag = sentence[wordIndex][1]

                # update symbols and states set
                if word not in self.symbols:
                    self.symbols.append(word)
                if tag not in self.states:
                    self.states.append(tag)

                # count outputs
                if tag not in self.outputs:
                    self.outputs[tag] = {'_sum' : 0.}
                if word not in self.outputs[tag]:
                    self.outputs[tag][word] = 0.
                self.outputs[tag][word] += 1
                self.outputs[tag]['_sum'] += 1
                
                # count either priors or transitions
                if wordIndex == 0:
                    if tag not in self.priors:
                        self.priors[tag] = 0.
                    self.priors[tag] += 1
                    self.priors['_sum'] += 1
                else:
                    previousTag = sentence[wordIndex-1][1]   # index [1] for its 'TAG'
                    if previousTag not in self.transitions:
                        self.transitions[previousTag] = {'_sum' : 0.}
                    if tag not in self.transitions[previousTag]:
                        self.transitions[previousTag][tag] = 0.
                    self.transitions[previousTag][tag] += 1
                    self.transitions[previousTag]['_sum'] += 1
        
        # replace count with log probability
        for tag in self.transitions:
            self._countToProb(self.transitions[tag])
        for tag in self.outputs:
            self._countToProb(self.outputs[tag])
        self._countToProb(self.priors)

        return HiddenMarkovModelTagger(self.symbols, self.states, self.transitions, self.outputs, self.priors)

    def _countToProb(self, dic):
        dicSum = dic['_sum']
        del dic['_sum']
        for key in dic:
            if dic[key] > 0.:
                if dicSum > 0.:
                    dic[key] = math.log(dic[key]) - math.log(dicSum)
            else:                       # prob is in [0, 1] => log(prob) is in [-INF, 0]
                dic[key] = -FLOAT_INF   # set it -INF if it does not exist

class HiddenMarkovModelTagger:
    def __init__(self, symbols, states, transitions, outputs, priors):
        self.symbols = symbols          # set of words existing
        self.states = states            # set of POS tags existing
        self.transitions = transitions  # transition probability  {'TAG' : {'nextTAG': count, ...}, ...}
        self.outputs = outputs          # emission probability    {'TAG' : {'observation': count, ...}, ...}
        self.priors = priors            # initial probability     {'TAG' : count, ...}
        self.viterbi = {}               # viterbi algo            {wordIndex : {'TAG' : (prob, 'path'), ...}, ...}    

    def tag(self, splitSentence):   # returns a list of tuples [('word','TAG'), ...]

        # initialization
        self.viterbi[0] = {}
        for tag in self.states:
            initViterbi = 0.
            if tag in self.priors:
                initViterbi += self.priors[tag]
            if splitSentence[0] in self.outputs[tag]:
                initViterbi += self.outputs[tag][splitSentence[0]]
            self.viterbi[0][tag] = (initViterbi, '_'+tag)

        # recursion
        for wordIndex in range(1, len(splitSentence)):
            word = splitSentence[wordIndex]
            self.viterbi[wordIndex] = {}

            found = False
            for tag in self.states:
                newViterbi = (-FLOAT_INF,'_NFTBA_')
                if tag in self.outputs and word in self.outputs[tag]:
                    newViterbi = self._maxViterbi(wordIndex, tag, word)
                    found = True
                self.viterbi[wordIndex][tag] = newViterbi
            if not found:
                for tag in self.states:
                    bestLast = self._maxViterbi(wordIndex, tag)
                    newViterbi = (bestLast[0],bestLast[1])
                    self.viterbi[wordIndex][tag] = newViterbi

        # termination
        finalViterbi = self._maxViterbi(len(splitSentence))
        
        # form the return list
        taggedSentence = []
        splitViterbi = list(filter(None, finalViterbi[1].split('_')))
        for wordIndex in range(len(splitSentence)):
            taggedSentence.append((splitSentence[wordIndex], splitViterbi[wordIndex]))
        return taggedSentence

    def _maxViterbi(self, wordIndex, tag='', word=''):
        maxProb = -FLOAT_INF
        bestPath = '_'
        for prevTag in self.states:
            newProb = self.viterbi[wordIndex-1][prevTag][0]
            
            if tag != '':
                if prevTag in self.transitions and tag in self.transitions[prevTag]:
                    newProb += self.transitions[prevTag][tag]
                else:
                    newProb = -FLOAT_INF
                if word != '':
                    newProb += self.outputs[tag][word]

            if newProb > maxProb:
                maxProb = newProb
                bestPath = self.viterbi[wordIndex-1][prevTag][1] + '_' + tag      
        
        if maxProb == -FLOAT_INF:
            for prevTag in self.states:
                newProb = self.viterbi[wordIndex-1][prevTag][0]
                
                if word != '':
                    newProb += self.outputs[tag][word]

                if newProb > maxProb:
                    maxProb = newProb
                    bestPath = self.viterbi[wordIndex-1][prevTag][1] + '_' + tag  

        return (maxProb, bestPath)

