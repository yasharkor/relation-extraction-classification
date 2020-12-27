import argparse
import csv
import operator
from  math import log
from sklearn.model_selection import KFold
from  sklearn.metrics import confusion_matrix, classification_report


#NOTES
# Bayes classifier will be argmax(prob(sentence|class)*Prob(class))
# The prob(sentence|class) can be seen as the product of all prob(word|class) in the sentence
# Prob(class) = Number of sentences of that class / number of total sentences
# Prob(word|class) = Count of that word in all sentences of that class / total words in that class
# Will implement laplace smoothing to cover cases where the word is not in that class
# Prob(word|class) = (Count of that word in all sentences of that class + 1) / (total number of words of that class + len(vocabulary of all classes))
# If a test word is not in the training vocabulary then remove that word from the probability counts





# Takes the training file and returns a list of tuples in the form
# (list of words, class)
# Does not record words which are not alpha numeric
def preProcessSentences(trainFile):
    output = []
    ids = []

    # Open and read the file
    with open(trainFile, 'r', encoding="utf8") as trainCsv:
        reader = csv.reader(trainCsv, quotechar='"')
        next(reader) # skip header

        # Iterate through rows and extract words
        for row in reader:
            words = []
            for word in row[1].split(' '):
              if word.isalnum():
                  words.append(word)
            output.append((words, row[2]))
            ids.append(row[0])
    return output, ids


class BOW_Classifier:
    def __init__(self):
        self.classProbabilities = {}
        self.wordProbabilities = {}
        self.vocabulary = []

    # Takes a list of tuples in the form (list of words, class)
    # and calculates probabilities as required for naive bayes classification
    def train(self, sentences):
        classCounts = {} # The dictionary of the form {class: number of sentences of that class}
        wordCounts = {} # The 2D dictionary of the form {class: {word: number of occurences of that word in that class}}
        classWordCounts = {} # The dictionary of the form {class: total number of words in that class}
        # Iterate through sentences
        for sentence, C in sentences:
            # Count the class
            if C in classCounts.keys():
                classCounts[C] += 1
            else:
                classCounts[C] = 1
                classWordCounts[C] = 0 # If it's not in the class counts then it's not in any of the counts
                wordCounts[C] = {}
            
            # Expand the vocabulary
            self.vocabulary = list(set(self.vocabulary) | set(sentence))
            
            #Iterate through words
            for word in sentence:
                classWordCounts[C] += 1 # Save total count of words of that class

                # Increment count of word given class
                if word in wordCounts[C].keys():
                    wordCounts[C][word] += 1
                else:
                    wordCounts[C][word] = 1

        # Calculate class probabilities
        for c, count in classCounts.items():
            self.classProbabilities[c] = count / len(sentences)

        # Calculate probability of a word given a class
        for c in wordCounts.keys():
            self.wordProbabilities[c] = {}
            # Store the probability of an unknown word
            self.wordProbabilities[c]['<UNK>'] = 1 / (classWordCounts[c] + len(self.vocabulary))
            for word, count in wordCounts[c].items():
                self.wordProbabilities[c][word] = (count + 1) / (classWordCounts[c] + len(self.vocabulary))
        
    
    # Takes a list of words (sentence) and returns the class with the maximum probability
    def predict(self, sentence):
        # Get a list of classes
        classes = [c for c in self.classProbabilities.keys()]
        probs = {} 

        # Check each class  
        for c in classes:
            prob = 1

            # Iterate through each word and get the product of all probabilities
            for word in sentence:
                # If not in the vocabulary at all ignore entirely
                if word not in self.vocabulary:
                    continue
                # If only not in the class vocabulary use unknown word, laplace smoothing
                if word not in self.wordProbabilities[c].keys():
                    prob *= self.wordProbabilities[c]['<UNK>']
                # If word in training data for this class use that probability
                else:
                    prob *= self.wordProbabilities[c][word]
            # multiply probability of the class with the product of words|class probabilities
            probs[c] = self.classProbabilities[c] * prob
        
        # Return the argmax
        return max(probs.items(), key=operator.itemgetter(1))[0]

# calculate accuracy
def getAccuracy(actual_pred_list):
    correct = 0
    wrongList = []
    for actual_pred in actual_pred_list:
        if actual_pred[0] == actual_pred[1]:
            correct += 1
        else:
            wrongList.append(actual_pred)
    return correct/len(actual_pred_list)

# Take training sentences and implement 3 fold cross validation and return mean accuracy
def train_accuracy(sentence):
    nsplit = 3
    kf = KFold(n_splits = nsplit)
    AverageAccuracy = 0
    for train_index, test_index in kf.split(sentence):
        Xtrain, Xtest, actual_pred_list = [], [], []
        for i in train_index:
            Xtrain.append(sentence[i])
        for i in test_index:
            Xtest.append(sentence[i])
        # Initialize the classifier
        classifier = BOW_Classifier()
        # Train the classifier
        classifier.train(Xtrain)
        # Get the test sentences

        for i in range(len(Xtest)):
            # get a prediction from the classifier
            pred = classifier.predict(Xtest[i][0])
            actual = Xtest[i][1]

            # write the output row
            actual_pred_list.append((actual, pred))

        AverageAccuracy += getAccuracy(actual_pred_list)  / nsplit
    return AverageAccuracy


def main():
    # Parse arguments
    argparser = argparse.ArgumentParser(description="Bag of Words classifier")
    argparser.add_argument('--train', required=True, help="The path to the training data file")
    argparser.add_argument('--test', required=True, help="The path to the testing data file")
    argparser.add_argument('--output', required=True, help="The path to the output csv file")
    argparser.add_argument('--dev', required=False, help="whether to include original labels or not", choices=['True', 'False'], default=False)
    # argparser.add_argument('--dev', required=False, help="whether to include original labels or not",)

    args = argparser.parse_args()
    trainFilePath = args.train
    testFilePath = args.test
    outputFilePath = args.output
    devFlag = args.dev == 'True'

    # Preprocess training sentences, ignoring ids
    sentences, _ = preProcessSentences(trainFilePath)
    # calculate 3 fold cross validation accuracy
    accuracy = train_accuracy(sentences)
    print('3 fold training accuracy is equal to: {}'.format(accuracy))

    # Initialize the classifier
    classifier = BOW_Classifier()
    # Train the classifier on all training samples
    classifier.train(sentences)

    # Get the test sentences
    sentences, ids = preProcessSentences(testFilePath)

    # Open the output file and create a csv writer
    with open(outputFilePath, 'w', encoding='utf8', newline="") as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(['original_label', 'classifier_assigned_label', 'row_id'])
        actual_pred_list = []
        # iterate through all sentences
        for i in range(len(sentences)):
            # get a prediction from the classifier
            pred = classifier.predict(sentences[i][0])
            # For dev we leave actual column blank
            if devFlag:
                actual = ''
            else:
                actual = sentences[i][1]
                actual_pred_list.append((actual,pred))
            rowID = ids[i]
            # write the output row
            writer.writerow([actual, pred, rowID])
    if devFlag == False:
        accuracy = getAccuracy(actual_pred_list)
        print('Test accuracy is equal to: {}'.format(accuracy))
        y_true = [pair[0] for pair in actual_pred_list]
        y_pred = [pair[1] for pair in actual_pred_list]
        print('confusion matrix on Test data: ')
        print(confusion_matrix(y_true, y_pred).transpose())
        print('classification report on Test data: ')
        print(classification_report(y_true, y_pred, digits = 3))



if __name__ == "__main__":
    main()
