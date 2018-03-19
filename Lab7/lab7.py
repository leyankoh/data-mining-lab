import pandas as pd 
import os 
import csv
import random
import string

with open(os.path.join("data", "SMSSpamCollection.csv")) as f:
    lines = [row for row in csv.reader(f.read().splitlines())]
    for row in lines:
        print row

classification = [row[0] for row in lines]
messages = [row[1] for row in lines]

print "Num of spam rows: {}".format(classification.count("spam"))
print "Num of non-spam rows: {}".format(classification.count("ham"))

hamMsg = [row for row in lines if row[0] == "ham"]
spamMsg = [row for row in lines if row[0]== "spam"]

def get_train_data(data, num):
    # create random list of ham messages
    rand_list = random.sample(range(0, len(ham)), num) # generate 1000 random indexes
    rand_hamMsg = []
    for i in rand_list:
        rand_hamMsg.append(data[i])

    return rand_hamMsg

ham = get_train_data(hamMsg, 747)
mergedSet = ham + spamMsg

def get_train_test(data, percentage): # percentage should be percentage that you want to be training
    setLength = int(len(data) * percentage)
    rand_list = random.sample(range(0, len(data)), setLength)
    #target = [row[0] for row in data] # get list of classifications
    #dataset = [row[1] for row in data] # get list of data
    train = []
    test = []
    for i in range(len(data)):
        if i in rand_list:
            train.append(data[i])
        else:
            test.append(data[i])
    train_x = [row[1] for row in train]
    train_Y = [row[0] for row in train]
    test_x = [row[1] for row in test]
    test_Y = [row[0] for row in test]
    return train_x, train_Y, test_x, test_Y

train_x, train_Y, test_x, test_Y = get_train_test(mergedSet, 0.8)

def get_matrix(data):
    # data should just be the list of messages
    # function returns
    matrixList = []
    for message in data:
     # message length
     # length in characters
     # number of digits
     # punctuation
     # uppercase letters
        msgLength = [len(message.split())]
        characterLength = [len(message)]
        numDigits = [sum(c.isdigit() for c in message)]
        numPunc = [len([c for c in message if c in string.punctuation])]
        numUpper = [sum(c.isupper() for c in message)]

        matrix = msgLength + characterLength + numDigits + numPunc + numUpper

        matrixList.append(matrix)

     return matrixList


train_matrix = get_matrix(train_x)

hamMessages = [row[1] for row in ham]
ham_matrix = get_matrix(hamMessages)

spamMessages = [row[1] for row in spamMsg]
spam_matrix = get_matrix(spamMessages)

# plotting exploratory analysis
import matplotlib.pyplot as plt

for i in range(5):
    plt.hist([row[i] for row in ham_matrix])
    plt.hist([row[i] for row in spam_matrix])
    plt.show()
