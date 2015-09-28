# Source Code for CS 4740, Project 1, Fall 2015
# Minae Kwon, Lylla Younes
# September 20, 2015

import csv
import sys
import requests
import math
from urllib2 import *
import numpy as np
from numpy import *
from nltk.tokenize import sent_tokenize, word_tokenize


#use this function to generate a bag of words from a series of 
#books in a corpus not accounting for any punctuation
#this fuction takes in an array of text files and returns an array of 
#all the words in said files enter in the following form: ['text_file1','text_file2']
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



#generates a vocabulary of all the words in a corpus
#takes in a list of text files (books in the corpus)
#this function returns a dictionary of words and their occurrences
def vocab(books):
    corpus = write_bag(books)
    vocab = {}
    for i in range(len(corpus)):
        if vocab.has_key(corpus[i]):
            vocab[corpus[i]] = vocab[corpus[i]] + 1
        else:
            vocab[corpus[i]] = 1
    return vocab


#this function generates a dictionary of bigrams from a corpus
#it takes in an array of text files
#enter in the following form: ['text_file1','text_file2']
def write_bigrams(books):
    corpus = write_bag(books)
    b_dict = {}
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


#this function generates random senteces
#using a unigram model. it returns a randomly
#generated sentence as a string
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

#disregard - helper function for bigram random
#sentence generator
def bigram_next_word(word, b_dict):
    values = []
    next_word = ''

    #getting values
    for i in b_dict[word]:
        if (len(i)>0):
            values.append(b_dict[word][i])
    
    if(len(b_dict[word])>1):
        counter = 1
        for counter in range(len(b_dict[word])-1):
            counter = counter+1
            b_dict[word][b_dict[word].keys()[counter]] = values[counter] + b_dict[word].values()[counter-1]
    
    #generating random number
    rand = random.randint(0,max(b_dict[word].values()))
    print "rand number:" + str(rand)
    
    #choosing next word
    if (rand <= min(values)):
        next_word = b_dict[word].keys()[0]
    else:
        for i in range(len(b_dict[word])-1):
            if ((b_dict[word].values()[i] < rand) and (rand <= b_dict[word].values()[i+1])):
                next_word = b_dict[word].keys()[i+1]

    print "next word:" + next_word
    return next_word

#disregard - helper function for bigram random
#sentence generator
def bigram_start_word(b_dict): 
    start_dict = {}
    values = []
    start_word = ''
    for word in b_dict:
        if (len(word)>0) and (word[0].isupper()):
            start_dict.update({word: b_dict[word]})
    

    #updating values to counts in start_array 
    for word in start_dict:
        num_occ = sum(start_dict[word].values())
        values.append(num_occ)
        start_dict[word] = num_occ
        #print values

    #adjusting values in start_array
    if(len(start_dict)>1):
        counter = 1
        for counter in range(len(start_dict)-1):
            counter = counter+1
            start_dict[start_dict.keys()[counter]] = values[counter] + start_dict.values()[counter-1]

    #generating random number
    rand = random.randint(0,max(start_dict.values()))
    #print "rand number:" + str(rand)

    #choosing start word
    if (rand <= min(start_dict.values())):
        start_word = start_dict.keys()[0]
    else:
        for i in range(len(start_dict)-1):
            if ((start_dict.values()[i] < rand) and (rand <= start_dict.values()[i+1])):
                start_word = start_dict.keys()[i+1]
    print 'START WORD'
    print start_word
    return start_word

#this function is our bigram random sentence generator
#it takes in an array of text files and prints
#a randomly generated sentence in the form of a string
def b_sent_gen(book):
    b_dict = write_bigrams(book)
    sentence = ''
    start_word = bigram_start_word(b_dict)
    next_word = bigram_next_word(start_word, b_dict)
    sentence = start_word + " " + next_word
    while(next_word != '<s>'):
        next_word = bigram_next_word(next_word, b_dict)
        if next_word == '<s>':
            print sentence
        else:
            sentence = sentence + " " + next_word
    print sentence
    
#this function takes in an array of text files
#and returns our principle data structure:
#a dictionary of dictionaries. The keys of the outer dictionary
#are words in the corpus and the values are other dictionaries
#of the words that follow those initial keys, and their counts
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

#generates a dictionary of word types and their counts
def type_dict(books):
    text = write_bag(books)
    t_dict = {}
    for word in text:
        if t_dict.has_key(word):
            t_dict[word] = t_dict[word] +1
        else:
            t_dict[word] = 1
    print t_dict

    V = len(t_dict.keys())
    for i in range(len(t_dict.keys())):
        f_word = t_dict[t_dict.keys()[i]]
    return t_dict

#Implement add-one smoothing for unigrams
def uni_add1(books):
    corpus = write_bag(books)
    N = float(len(corpus))
    t_dict = type_dict(books)
    V = len(t_dict)
    for key in t_dict.keys():
        #compute unigram probabilites
        new_val = (t_dict[key] + 1.0) / (N+V)
        t_dict[key] = new_val
    return t_dict
        
#Implement add-one smoothing for bigrams
def bi_add1(books):
    corpus = write_bag(books)
    N = float(len(corpus))
    bi_dict = write_bigrams(books)
    t_dict = type_dict(books)
    V = len(t_dict)
    for key in bi_dict.keys():
        
        for key2 in bi_dict[key].keys():
            new_val = (bi_dict[key][key2] + 1.0) / (t_dict[key]+V)
            bi_dict[key][key2] = new_val
    print bi_dict
    
    
def bi_add1_adjusted(books):
    pass
    
            
#Implement Good Turing Smoothing Algorithm on a corpus
#modifying counts for elements with counts less than 5
#This function takes in an array of text files, and
#returns a dictionary of bigrams and their probabilities
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
    
    Nc0 = (len(t_dict) * len(t_dict)) - total
    
    print "Modifying Small Counts With Good Turing"
    for key in bi_dict.keys():
        print bi_dict[key]
        for key2 in bi_dict[key].keys():
            count = bi_dict[key][key2]
            if (count < 5):
                if (count == 1):
                    Nc = Nc1
                    Nc_plus = Nc2
                    modified_count_b = (((count+1.0) * Nc_plus) / Nc) *1000.0
                    unseen_bigrams = (V-(t_dict[key])) * (Nc1 / Nc0)
                    prob = ((modified_count_b/1000.0) / (t_dict[key] + unseen_bigrams)) 
                    bi_dict[key][key2] = prob
                elif (bi_dict[key][key2] == 2):
                    Nc = Nc2
                    Nc_plus = Nc3
                    modified_count_b = (((count+1.0) * Nc_plus) / Nc) * 1000.0
                    unseen_bigrams = (V-(t_dict[key])) * (Nc1 / Nc0)
                    prob = ((modified_count_b/1000.0) / (t_dict[key] + unseen_bigrams)) 
                    bi_dict[key][key2] = prob
                elif (bi_dict[key][key2] == 3):
                    Nc = Nc3
                    Nc_plus = Nc4
                    modified_count_b = (((count+1.0) * Nc_plus) / Nc) * 1000.0
                    unseen_bigrams = (V-(t_dict[key])) * (Nc1 / Nc0)
                    prob = ((modified_count_b/1000.0) / (t_dict[key] + unseen_bigrams)) 
                    bi_dict[key][key2] = prob
                elif (bi_dict[key][key2] == 4):
                    Nc = Nc4
                    Nc_plus = Nc5
                    modified_count_b = (((count+1.0) * Nc_plus) / Nc) * 1000.0
                    unseen_bigrams = (V-(t_dict[key])) * (Nc1 / Nc0)
                    prob = ((modified_count_b/1000.0) / (t_dict[key] + unseen_bigrams)) 
                    bi_dict[key][key2] = prob
            else:
                count_b = count
                unseen_bigrams = (V-(t_dict[key])) * (Nc1 / Nc0)
                prob = (count_b / (t_dict[key] + unseen_bigrams)) 
                bi_dict[key][key2] = prob
                
            print prob 
    
    print "Final Bigram Dictionary With Modified Counts:"
    return bi_dict
    
            
#This function computes the perplexity of a test book
#and a training corpus. The training attribute is a list of
#books in a corpus and the test book is one text file
#containing the book to be tested. This function returns
#the perplexity value, a floating point number.
def perplexity(training, test):
    print "ENTERING"
    bi_dict = bi_add1(training)
    text = write_bag(test)
    N = len(text)
    sumofprobs = 0.0
    no = 0
    
    t_dict = {}
    for word in text:
        if t_dict.has_key(word):
            t_dict[word] = t_dict[word] +1
        else:
            t_dict[word] = 1
    
    for i in range(N-1): #fix
        bi_p1 = text[i]
        bi_p2 = text[i+1]
        
        if (bi_dict.has_key(bi_p1)): #has first
            if (bi_dict[bi_p1].has_key(bi_p2)): #has second
                prob = bi_dict[bi_p1][bi_p2]
            elif (bi_dict[bi_p1].has_key('<unk>')): #first and not second
                prob = bi_dict[bi_p1]['<unk>']
            else:
                no+=1
                prob = 0.000000003
        elif(bi_dict.has_key(bi_p1) == False): #no first
            if (bi_dict['<unk>'].has_key(bi_p2)):   #no first and second
                prob = bi_dict['<unk>'][bi_p2]
            elif (bi_dict['<unk>'].has_key(bi_p2) == False):
                no +=1
                prob = 0.000000003        
        sumofprobs += (-1.0 * math.log10(prob))
    print math.exp((sumofprobs)/N)
        
     