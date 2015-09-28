# Preliminary text parser for CS 4740, Project 1, Fall 2015
# Minae Kwon, Lylla Younes

import csv
import sys
import requests
from urllib2 import *
import numpy as np
from numpy import *
from nltk.tokenize import sent_tokenize, word_tokenize


#use this function to generate a bag of words from a series of books in a corpus
#not accounting for any punctuation
#this fuction takes in an array of text files
#enter in the following form: ['text_file1','text_file2']
def write_bag(books):
    corpus = []
    new_corpus = []
    for b in range(len(books)):
        print 'Reading: ' + books[b]
        book = open(books[b], 'r')
        lines = []
        words2 = []
        for line in book:
            lines.append(line)
        for element in lines:
            while (element.find(' ') != -1):
                index1 = element.find(' ') + 1
                word = element[0:(index1-1)]
                words2.append(word)
                element = element[index1:]
        for j in range(len(words2)):
            corpus.append(words2[j])
            
    for i in range(len(corpus)):
        if(len(corpus[i]) >= 1):
            if((corpus[i][-1] == '.') | (corpus[i][-1] == '?') | (corpus[i][-1] == '!')):
                temp = corpus[i][-1]
                new_corpus.append(corpus[i][0:-1])
                new_corpus.append(temp)
                new_corpus.append('<s>')
            elif((corpus[i][-1] == ',') | (corpus[i][-1] == ';') | (corpus[i][-1] == ':')):
                temp = corpus[i][-1]
                new_corpus.append(corpus[i][0:-1])
                new_corpus.append(temp)
            else:
                new_corpus.append(corpus[i])
                
    vocab = {}
    for i in range(len(new_corpus)):
        if vocab.has_key(new_corpus[i]):
            vocab[new_corpus[i]] = vocab[new_corpus[i]] + 1
        else:
            vocab[new_corpus[i]] = 1
                
    for key in vocab.keys():
        if (vocab[key] == 1):
            index = new_corpus.index(key)
            new_corpus[index] = '<unk>'
    
    return(new_corpus)




def vocab(books):
    corpus = write_bag(books)
    vocab = {}
    for i in range(len(corpus)):
        if vocab.has_key(corpus[i]):
            vocab[corpus[i]] = vocab[corpus[i]] + 1
        else:
            vocab[corpus[i]] = 1
    return vocab
    
            
#use this to generate add markers to the end of sentences
#in a given corpus. this fuction takes in an array of text files
#enter in the following form: ['text_file1','text_file2']
def parse_sent():
    #corpus = write_bag(books)
    corpus = ['Hi','and,','why!']
    new_corpus = []
    for i in range(len(corpus)):
        print i
        if((corpus[i][-1] == '.') | (corpus[i][-1] == '?') | (corpus[i][-1] == '!')):
            print 'end punct'
            temp = corpus[i][-1]
            new_corpus.append(corpus[i][0:-1])
            new_corpus.append(temp)
            new_corpus.append('<s>')
        elif((corpus[i][-1] == ',') | (corpus[i][-1] == ';') | (corpus[i][-1] == ':')):
            print 'middle punct'
            temp = corpus[i][-1]
            new_corpus.append(corpus[i][0:-1])
            new_corpus.append(temp)
        else:
            new_corpus.append(corpus[i])
    return new_corpus


#this function generates a dictionary of bigrams from a corpus
#it takes in an array of text files
#enter in the following form: ['text_file1','text_file2']
def write_bigrams(books):
    corpus = write_bag(books)
    b_dict = {}
    #bag = ['Here','I','am,','a', 'lovely', 'a', 'damn', 'a', 'damn','sentence.'] 
    i=0
    while(i != (len(corpus)-1)):
        if(b_dict.has_key(corpus[i])):
            if(b_dict.get(corpus[i]).has_key(corpus[i+1])):
                (b_dict.get(corpus[i]))[corpus[i+1]] = b_dict.get(corpus[i]).get(corpus[i+1]) + 1
            else:
                b_dict.get(corpus[i])[corpus[i+1]] = 1
        else:
            word = corpus[i]
            next_word = corpus[i+1]
            b_dict[word] = {corpus[i+1]: 1}
        i+= 1
    return b_dict


#generates list of all capital words in a given corpus
#takes in a list of text files
def get_capitalized(books):
    corpus = write_bag(books)
    capitalized = []
    for i in range(len(corpus)):
        if (corpus[i][0].isupper()):
            capitalized.append(corpus[i])
    return capitalized

#generates a dictionary of word types and their number
#of occurrence in a given corpus
#takes in a list of text files
def dictionary(books):
    text = write_bag(books)
    words = {}    
    for word in text:
            if (words.has_key(word)):
                words[word] = words[word] + 1
            else:
                words[word] = 1
    return words


#this function generates random senteces
#using a unigram model. 
def unigram(books):
    text = write_bag(books)
    #initializing arrays/variables
    start_array = []
    word_array = []
    empty_string = ''

    #appending words to start array 
    for word in text:
     if(word[0].isupper()):
                start_array.append(word)
                    
    #randomly generating a start word
    start = random.choice(start_array)
    
    #randomly generating a word
    random_word = random.choice(text)
    
    #intitalizing sentence
    sentence = start    

    while(random_word!="."):
        sentence += " "+  random_word + " "
        random_word = random.choice(text)
    
    return (sentence)


#this function generates random sentences based on
#a bigram model. 
def bigrams(books):
    b_dict = write_bigrams(books)
    start = ''
    sentence = ''
    
    start_array = b_dict.get("<s>").keys()
    ran = len(start_array)
    random1 = int(random.randint(0,ran))
    start = start_array[random1]
    sentence += start
    
    hvalue = max(b_dict[start].values())
    
    candidates = []
    for i in b_dict[start]:
        if(b_dict[start][i] == hvalue):
            candidates.append(i)
    length = len(candidates) - 1
    nex = candidates[int(random.randint(0,length))]

    sentence = start+ " "+ nex
    
    end = int(random.randint(2,20))
    j = int(1)
    
    while(j <= end):
        hvalue = max(b_dict[nex].values())
        
        cand = []
        for i in b_dict[nex]:
            if(b_dict[nex][i] == hvalue):
                cand.append(i)
        nex = cand[int(random.randint(0,(len(cand)-1)))]
        sentence+= (' ' + nex)
        j = j + 1

    print sentence
    
    
def bigram_dict(books):
    t_dict = type_dict(books)
    text = write_bag(books)
    b_dict = {}
#store entries in b_dict   
    for i in range(len(t_dict)):
        for x in range(len(t_dict)):
            if (t_dict.keys()[i]!=t_dict.keys()[x]):
                b_dict[t_dict.keys()[i]] = {t_dict.keys()[x]:0}
    print b_dict
    
#increment entries in b_dict based on text
    for i in range(len(text)):
        if text[i+1] in b_dict[text[i]].keys():
            b_dict[text[i]][text[i+1]] = b_dict[text[i]][text[i+1]] +1
    #print b_dict
    return b_dict
    

        
#training is a list of books in a corpus
#test is one book 
def perplexity(training, test):
    print "ENTERING"
    array = types2(training)
    print 'Array '
    print array
    text = write_bag(test)
    N = len(text)
    print "TEXT"
    print text
    sumofprobs = 0.0
    for i in range(N): #fix
        bigram_p1 = text[i]
        print "BI1: " + str(bigram_p1)
        bigram_p2 = text[i+1]
        prob = float((array[i][i+1]) / 1000000000.0)
        sumofprobs += (-1.0 * np.log(prob))
    return math.exp((sumofprobs)/N)



# Preliminary text parser for CS 4740, Project 1, Fall 2015
# Minae Kwon, Lylla Younes

import csv
import sys
import requests
from urllib2 import *
import numpy as np
from numpy import *
from nltk.tokenize import sent_tokenize, word_tokenize


#use this function to generate a bag of words from a series of books in a corpus
#not accounting for any punctuation
#this fuction takes in an array of text files
#enter in the following form: ['text_file1','text_file2']
def write_bag(books):
    corpus = []
    new_corpus = []
    for b in range(len(books)):
        print 'Reading: ' + books[b]
        book = open(books[b], 'r')
        lines = []
        words2 = []
        for line in book:
            lines.append(line)
        for element in lines:
            while (element.find(' ') != -1):
                index1 = element.find(' ') + 1
                word = element[0:(index1-1)]
                words2.append(word)
                element = element[index1:]
        for j in range(len(words2)):
            corpus.append(words2[j])
            
    for i in range(len(corpus)):
        if(len(corpus[i]) >= 1):
            if((corpus[i][-1] == '.') | (corpus[i][-1] == '?') | (corpus[i][-1] == '!')):
                temp = corpus[i][-1]
                new_corpus.append(corpus[i][0:-1])
                new_corpus.append(temp)
                new_corpus.append('<s>')
            elif((corpus[i][-1] == ',') | (corpus[i][-1] == ';') | (corpus[i][-1] == ':')):
                temp = corpus[i][-1]
                new_corpus.append(corpus[i][0:-1])
                new_corpus.append(temp)
            else:
                new_corpus.append(corpus[i])
                
    vocab = {}
    for i in range(len(new_corpus)):
        if vocab.has_key(new_corpus[i]):
            vocab[new_corpus[i]] = vocab[new_corpus[i]] + 1
        else:
            vocab[new_corpus[i]] = 1
                
    # for key in vocab.keys():
    #     if (vocab[key] == 1):
    #         index = new_corpus.index(key)
    #         new_corpus[index] = '<unk>'
    
    return(new_corpus)




def vocab(books):
    corpus = write_bag(books)
    vocab = {}
    for i in range(len(corpus)):
        if vocab.has_key(corpus[i]):
            vocab[corpus[i]] = vocab[corpus[i]] + 1
        else:
            vocab[corpus[i]] = 1
    return vocab
    
            
#use this to generate add markers to the end of sentences
#in a given corpus. this fuction takes in an array of text files
#enter in the following form: ['text_file1','text_file2']
def parse_sent():
    #corpus = write_bag(books)
    corpus = ['Hi','and,','why!']
    new_corpus = []
    for i in range(len(corpus)):
        print i
        if((corpus[i][-1] == '.') | (corpus[i][-1] == '?') | (corpus[i][-1] == '!')):
            print 'end punct'
            temp = corpus[i][-1]
            new_corpus.append(corpus[i][0:-1])
            new_corpus.append(temp)
            new_corpus.append('<s>')
        elif((corpus[i][-1] == ',') | (corpus[i][-1] == ';') | (corpus[i][-1] == ':')):
            print 'middle punct'
            temp = corpus[i][-1]
            new_corpus.append(corpus[i][0:-1])
            new_corpus.append(temp)
        else:
            new_corpus.append(corpus[i])
    return new_corpus


#this function generates a dictionary of bigrams from a corpus
#it takes in an array of text files
#enter in the following form: ['text_file1','text_file2']
def write_bigrams(books):
    corpus = write_bag(books)
    b_dict = {}
    #bag = ['Here','I','am,','a', 'lovely', 'a', 'damn', 'a', 'damn','sentence.'] 
    i=0
    while(i != (len(corpus)-1)):
        if(b_dict.has_key(corpus[i])):
            if(b_dict.get(corpus[i]).has_key(corpus[i+1])):
                (b_dict.get(corpus[i]))[corpus[i+1]] = b_dict.get(corpus[i]).get(corpus[i+1]) + 1
            else:
                b_dict.get(corpus[i])[corpus[i+1]] = 1
        else:
            word = corpus[i]
            next_word = corpus[i+1]
            b_dict[word] = {corpus[i+1]: 1}
        i+= 1
    return b_dict


#generates list of all capital words in a given corpus
#takes in a list of text files
def get_capitalized(books):
    corpus = write_bag(books)
    capitalized = []
    for i in range(len(corpus)):
        if (corpus[i][0].isupper()):
            capitalized.append(corpus[i])
    return capitalized

#generates a dictionary of word types and their number
#of occurrence in a given corpus
#takes in a list of text files
def dictionary(books):
    text = write_bag(books)
    words = {}    
    for word in text:
            if (words.has_key(word)):
                words[word] = words[word] + 1
            else:
                words[word] = 1
    return words


#this function generates random senteces
#using a unigram model. 
def unigram(books):
    text = write_bag(books)
    #initializing arrays/variables
    start_array = []
    word_array = []
    empty_string = ''

    #appending words to start array 
    for word in text:
     if(len(word) > 0 and (word[0].isupper())):
                start_array.append(word)
                    
    #randomly generating a start word
    start = random.choice(start_array)
    
    #randomly generating a word
    random_word = random.choice(text)
    
    #intitalizing sentence
    sentence = start    

    while(random_word!="."):
        sentence += " "+  random_word + " "
        random_word = random.choice(text)
    
    return (sentence)


#this function generates random sentences based on
#a bigram model. 
def bigrams(books):
    b_dict = write_bigrams(books)
    start = ''
    sentence = ''
    
    start_array = b_dict.get("<s>").keys()
    ran = len(start_array)
    random1 = int(random.randint(0,ran))
    start = start_array[random1]
    sentence += start
    
    hvalue = max(b_dict[start].values())
    
    candidates = []
    for i in b_dict[start]:
        if(b_dict[start][i] == hvalue):
            candidates.append(i)
    length = len(candidates) - 1
    nex = candidates[int(random.randint(0,length))]

    sentence = start+ " "+ nex
    
    end = int(random.randint(2,20))
    j = int(1)
    
    while(j <= end):
        hvalue = max(b_dict[nex].values())
        
        cand = []
        for i in b_dict[nex]:
            if(b_dict[nex][i] == hvalue):
                cand.append(i)
        nex = cand[int(random.randint(0,(len(cand)-1)))]
        sentence+= (' ' + nex)
        j = j + 1

    print sentence
    
    
def bigram_dict(books):
    t_dict = type_dict(books)
    text = write_bag(books)
    b_dict = {}
#store entries in b_dict   
    for i in range(len(t_dict)):
        for x in range(len(t_dict)):
            if (t_dict.keys()[i]!=t_dict.keys()[x]):
                b_dict[t_dict.keys()[i]] = {t_dict.keys()[x]:0}
    print b_dict
    
#increment entries in b_dict based on text
    for i in range(len(text)):
        if text[i+1] in b_dict[text[i]].keys():
            b_dict[text[i]][text[i+1]] = b_dict[text[i]][text[i+1]] +1
    #print b_dict
    print b_dict
    
def bigrams_good_turing():
#increment count for each entry by 1
    for word in b_dict:
        for entry in word:
            b_dict[word][entry] = b_dict[word][entry] +1

# this snippet of code was borrowed from Stack Overflow
# to truncate long decimals, see url below:
# http://stackoverflow.com/questions/783897/truncating-floats-in-python
def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

#this function first creates an array of all the types of
#words in the text. then it creates an array of the seen
#and unseen bigrams in the text
def types(books):
    text = write_bag(books)
    types = []
    probabilities = []
    index_dict = {}
    vals = {}
    sum_p = 0.0
    N = float(len(text))
    
    print "Creating array of word types..."
    for word in text:
        if not(word in types):
            types.append(word)
            index_dict[word] = types.index(word)      

    #create array
    print "Creating array of bigram counts..."
    array = np.arange(len(types)*len(types)).reshape((len(types), len(types)))
    array[:,:] = 0.0
    for i in range(len(text)-1):
        word1 = text[i]
        word2 = text[i+1]
        index_w1 = index_dict.get(word1)
        index_w2 = index_dict.get(word2)
        array[index_w1,index_w2] = array[index_w1,index_w2] + 1.0
    
    #we now have an array of adjusted counts (+1)
    #now we want to compute the GT and Laplacian probabilities based on
    #the value of c. If c < 5, compute GT, otherwise, Laplacian     
    
    print "Creating array for bigram ocurrences ..."
    for i in range(len(array)):
        for j in range(len(array)):
            if (vals.has_key(array[i][j])):
                vals[array[i][j]] = vals[array[i][j]] + 1.0
            else:
                vals[array[i][j]] = 1.0
    
    for index, c in np.ndenumerate(array):
        array[index] += 1.0
        if (array[index] < 5.0):
            #compute GT probability
            print "Modifying array for Good Turing smoothing probabilities..."
            Nc = vals[array[index]]
            Nc1 = vals[array[index] + 1.0]
            if (vals.has_key(array[index])):
                new_count = ((array[index] * Nc1) / Nc)
                prob = ((new_count / N)) * 1000000000
                array[index] = prob
        else:
            #compute Laplacian probability
            print "Modifying array for Laplacian smoothing probabilities..."
            for i, val in np.ndenumerate(array[index[0]]):
                sum_p += val
            temp = float(sum_p) + float(len(types))
            prob = (float(array[index]) / temp) * 1000000000
            array[index] = prob
    return array
        


def good_turing(books):
    array = types(books)
    c = 0
    values = []
    vals = {}
    N = float(len(write_bag(books)))
    
    for i in range(len(array)): #appending values to array 
        for j in range(len(array)):
            values.append(array[i][j])    
    
    for i in range(len(values)): #creating vals dictionary 
        if values[i] in vals.keys():
            vals[values[i]] = (vals[values[i]]+1.0)
        else:
            vals[values[i]] = 1.0
   
    for i in range(len(array)):
        for j in range(len(array)):
            c = array[i][j]
            if (c < 5):
                Nc = vals[c]
                Nc1 = vals[float(c)+1.0]
                
                if (vals.has_key(c)):
                    new_count = ((c * Nc1) / Nc)
                    prob = new_count / N
                    array[i][j] = prob
            else:
                c = float(c) / N
                    
    print array


def unknowns(books):
    corpus = write_bag(books)
    vocab_dict = vocab(books)
    for key in vocab_dict.keys():
        if (vocab_dict[key] == 1):
            index = corpus.index(key)
            corpus[index] = '<unk>'
    print corpus


#this function first creates an array of all the types of
#words in the text. then it creates an array of the seen
#and unseen bigrams in the text
def types2(books):
    text = write_bag(books)
    types = []
    probabilities = []
    index_dict = {}
    vals = {}
    sum_p = 0.0
    N = float(len(text))
    
    print "Creating array of word types..."
    for word in text:
        if not(word in types):
            types.append(word)
            index_dict[word] = types.index(word)      

    #create array
    print "Creating array of bigram counts..."
    array = np.arange(len(types)*len(types)).reshape((len(types), len(types)))
    array[:,:] = 0.0
    for i in range(len(text)-1):
        word1 = text[i]
        word2 = text[i+1]
        index_w1 = index_dict.get(word1)
        index_w2 = index_dict.get(word2)
        array[index_w1,index_w2] = array[index_w1,index_w2] + 1.0
    
    #we now have an array of adjusted counts (+1)
    #now we want to compute the GT and Laplacian probabilities based on
    #the value of c. If c < 5, compute GT, otherwise, Laplacian     
    
    print "Creating array for bigram ocurrences ..."
    for i in range(len(array)):
        for j in range(len(array)):
            if (vals.has_key(array[i][j])):
                vals[array[i][j]] = vals[array[i][j]] + 1.0
            else:
                vals[array[i][j]] = 1.0
    
    #print 'first array: '
    #print array
    
    Nc1 = vals[1]
    Nc2 = vals[2]
    Nc3 = vals[3]
    Nc4 = vals[4]
    Nc5 = vals[5]
        
    #compute the GT probabilities for counts < 5
    array[array == 0] = (((1.0 * Nc2) / Nc1) / N) * 1000000000
    array[array == 1] = (((2.0 * Nc2) / Nc1) / N) * 1000000000
    array[array == 2] = (((3.0 * Nc3) / Nc2) / N) * 1000000000
    array[array == 3] = (((4.0 * Nc4) / Nc3) / N) * 1000000000
    array[array == 4] = (((5.0 * Nc5) / Nc4) / N) * 1000000000
    
    #print 'End'
    print array


def numpii ():
    types = ['e','e','e','r','r','l','l','k','d','d','d','e','o','i','y','t']
    print "len " + str(len(types))
    array = np.arange(len(types)*len(types)).reshape((len(types), len(types)))
    array[:,:] = 0.0
    print array
    x = 5.0
    array[array == 0] = x
    print array
    array += 1
    print array
        


def write_types(books):
    t_dict = {}
    for word in corpus:
        if t_dict.has_key(word):
            t_dict[word] = t_dict[word] +1
        else:
            t_dict[word] = 1
    return t_dict


def trygrams (books):
    print "Writing bag of words..."
    corpus = write_bag(books)
    print "Writing bigrams dictionary..."
    bi_dict = write_bigrams(books)
    
    
    print "Writing types..."
    t_dict = {}
    for word in corpus:
        if t_dict.has_key(word):
            t_dict[word] = t_dict[word] +1
        else:
            t_dict[word] = 1

    #get counts
    Nc1 = 0
    Nc2 = 0
    Nc3 = 0
    Nc4 = 0
    Nc5 = 0
    
    unseen_bigrams = 0
    V = len(t_dict.keys())
    total = 0
    
    print "Getting bigram counts..."
    for value in bi_dict.values():
        for val in value.values():
            if (val == 1):
                Nc1 += 1.0
                total += 1.0
            elif (val == 2):
                Nc2 += 1.0
                total += 1.0
            elif (val == 3):
                Nc3 += 1.0
                total += 1.0
            elif (val == 4):
                Nc4 += 1.0
                total +=1
            elif (val == 5):
                Nc5 += 1.0
                total += 1.0
            else:
                total += 1.0
    
    Nc0 = (len(corpus) * len(corpus)) - total
    
    print "Modifying Small Counts With Good Turing"
    for key in bi_dict.keys():
        print bi_dict[key]
        for key2 in bi_dict[key].keys():
            count = bi_dict[key][key2]
            if (count < 5):
                if (count == 1):
                    print 'ONE'
                    Nc = Nc1
                    Nc_plus = Nc2
                    modified_count_b = (((count+1.0) * Nc_plus) / Nc) *1000.0
                    print 'MODIFIED ' + str(modified_count_b)
                    unseen_bigrams = (V-(t_dict[key])) * (Nc1 / Nc0)
                    print 'UNSEEN'
                    print (Nc1 / Nc0)
                    prob = ((modified_count_b/1000.0) / (t_dict[key] + unseen_bigrams)) 
                    bi_dict[key][key2] = prob
                elif (bi_dict[key][key2] == 2):
                    print 'TWO'
                    Nc = Nc2
                    Nc_plus = Nc3
                    modified_count_b = (((count+1.0) * Nc_plus) / Nc) * 1000.0
                    unseen_bigrams = (V-(t_dict[key])) * (Nc1 / Nc0)
                    prob = ((modified_count_b/1000.0) / (t_dict[key] + unseen_bigrams)) 
                    bi_dict[key][key2] = prob
                elif (bi_dict[key][key2] == 3):
                    print 'THREE'
                    Nc = Nc3
                    Nc_plus = Nc4
                    modified_count_b = (((count+1.0) * Nc_plus) / Nc) * 1000.0
                    unseen_bigrams = (V-(t_dict[key])) * (Nc1 / Nc0)
                    prob = ((modified_count_b/1000.0) / (t_dict[key] + unseen_bigrams)) 
                    bi_dict[key][key2] = prob
                elif (bi_dict[key][key2] == 4):
                    print 'FOUR'
                    Nc = Nc4
                    Nc_plus = Nc5
                    modified_count_b = (((count+1.0) * Nc_plus) / Nc) * 1000.0
                    unseen_bigrams = (V-(t_dict[key])) * (Nc1 / Nc0)
                    prob = ((modified_count_b/1000.0) / (t_dict[key] + unseen_bigrams)) 
                    bi_dict[key][key2] = prob
            else:
                print 'ELSE'
                count_b = count
                unseen_bigrams = (V-(t_dict[key])) * (Nc1 / Nc0)
                prob = (count_b / (t_dict[key] + unseen_bigrams)) 
                bi_dict[key][key2] = prob
                
            print prob 
    
    print "Final Bigram Dictionary With Modified Counts:"
    print bi_dict
    
            

#training is a list of books in a corpus
#test is one book 
def perplexity(training, test):
    print "ENTERING"
    prob_dict = trygrams(training)
    text = write_bag(test)
    N = len(text)
    sumofprobs = 0.0
    print 'ARRAY'
    print array
    for i in range(N): #fix
        bigram_p1 = text[i]
        print "BI1: " + str(bigram_p1)
        bigram_p2 = text[i+1]
        prob = float((prob_dict[i][i+1]) / 1000000000.0)
        sumofprobs += (-1.0 * np.log(prob))
    return math.exp((sumofprobs)/N)

            
            
    


    
    
    
    
    
    


    
    
    
    
    
    
