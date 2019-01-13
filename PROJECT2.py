from gensim.models import word2vec #pip install --upgrade gensim
import sys
import os
import logging
import codecs
import pdb
import scipy as sp
#import tsne as TSNE dont need it
import numpy as np
import pylab
from nltk.tokenize import WordPunctTokenizer

# The text gets split into sentences to use less RAM (required by gensim)
class FileToSentences(object):
    def __init__(self, fname, maxlen):
        self.fname = fname
        self.maxlen = maxlen

    def __iter__(self):
        with open(self.fname, "r") as ftext:
            text = ftext.read().split()
            words = []
            for word in text:
                if len(words) >= self.maxlen:
                    yield words
                    words = []
                words.append(word)
            yield words
            
file_path = "NewsQur.txt"
sentences = FileToSentences("NewsQur.txt", 100)
sentences1 = FileToSentences("NewsQur1.txt", 100)
sentences2 = FileToSentences("NewsQur2.txt", 100)
#np.savetxt('NewsQur1.txt', sentences1, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')
#np.savetxt('NewsQur2.txt', sentences2, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')
model = word2vec.Word2Vec(sentences, size=75, min_count=1, sg=0, iter=1, window=5)
print('loaded the first model')
model1 = word2vec.Word2Vec(sentences1, size=75, min_count=1, sg=0, iter=1, window=5)
print('loaded the second model')
model2 = word2vec.Word2Vec(sentences2, size=75, min_count=1, sg=0, iter=1, window=5)
print('loaded the third model')
print("Evaluating..")
related_pair_total = 0
related_pair_count = 0
related_pairs = [("شمال", "جنوب"), ("يمين", "يسار"), ("الرياض", "جدة"),
                 ("جامعة", "كلية"), ("بيت", "مسجد"), ("شجرة", "نخلة"), ("هاتف", "جوال"), ("الجمعة", "السبت"), 
("فرنسا", "باريس"), ("فرنسا", "اليابان"), ("ناحية", "جهة"), ("نوفمبر", "ديسمبر")]
related_pairs=[ ("الله", "العظيم"), ("المؤمنين", "المسلمين")]
for pair in related_pairs:
    related_pair_count += 1
    related_pair_total += model.similarity(pair[0], pair[1])
related_pair_measure = related_pair_total / related_pair_count
print("Similarity (SIM) = " + str(related_pair_measure))

unrelated_pair_total = 0
unrelated_pair_count = 0
#unrelated_pairs = [("شمال", "جامعة"), ("يمين", "المعلم"), ("الرياض", "طاولة"), ("جامعة", "شجرة"), ("بيت", "ثمر"), ("شجرة", "نور"), ("هاتف", "السبت"), ("الجمعة", "السيارة"), ("فرنسا", "ديسمبر"), ("فرنسا", "رقم"), ("ناحية", "شرطي"), ("نوفمبر", "الباب")]
unrelated_pairs=[ ("قالوا", "يوم"), ("نار", "يعبدون")]
for pair in unrelated_pairs:
    unrelated_pair_count += 1
    unrelated_pair_total += model.similarity(pair[0], pair[1])
unrelated_pair_measure = unrelated_pair_total / unrelated_pair_count
print("Dissimilarity (DIS) = " + str(unrelated_pair_measure))




# Getting the most similar words as a list
def most_similar_to_list(text):
    words = []
    for i in range(0, len(text)):
        w = str(text[i]).split("'")
        words.append(w[1])
    return words
fullText = open(file_path, "r").read().split()

while True:
    print("test")
    try:
        a = "7"#str(input("Which application you want to use? 1 = DEEP SEARCH , 2 = RELATIONSHIP GUESSER , 3 = PICK THE ODD, 4 = Print the vector of word x ,5 = x and y similarity value , 6 = words similar to x but not y: "))
        if a == "1":
            # Application 1 (DEEP SEARCH): Prints words similar to word x, and gets search results from the similar words
            x = str(input("Please enter a word: "))
            print(model.most_similar(x, topn=4))
            w = most_similar_to_list(model.most_similar(x, topn=4))
            for i in range(0, 4):
                y_index = fullText.index(w[i])
                y = fullText[y_index - 4] + " " + fullText[y_index - 3] + " " + fullText[y_index - 2] + " " + \
                    fullText[y_index - 1] + " " + fullText[y_index] + " " + fullText[y_index + 1] + " " + fullText[
                        y_index + 2] + " " + fullText[y_index + 3] + " " + fullText[y_index + 4]
                print("result " + str(i + 1) + ": " + y)
        elif a == "2":
            # Application 2 (RELATIONSHIP GUESSER): Brings list of words that are similar to y1 and x2 but far from x1
            print("x1(France) --> y1(Paris), x2(Japan) --> (Tokyo)")
            x1 = str(input("x1 = "))
            y1 = str(input("y1 = "))
            x2 = str(input("x2 = "))
            w = most_similar_to_list(model.most_similar(positive=[y1, x2], negative=[x1], topn=3))
            print("Top 3 guesses: ")
            for i in range(0, 3):
                print("guess " + str(i + 1) + ": " + w[i])
        elif a == "3":
            # Application 3 (PICK THE ODD): The user enters a list of words, then the application picks the odd one
            w = str(input("Please enter list of words: ")).split(" ")
            w_weights = []
            for i in range(0, len(w)):
                w_weight = 0.0
                for j in range(0, len(w)):
                    w_weight += model.similarity(w[i], w[j])
                w_weights.append(w_weight)
            w_odd = 0
            for i in range(0, len(w_weights)):
                if w_weights[i] < w_weights[w_odd]:
                    w_odd = i
            print("the odd word is: " + w[w_odd])
        elif a == "4":
            # Application 4: Prints the vector of the word x
            x = str(input("Please enter a word: "))
            print("Length of vector = " + str(len(model[x])))
            print(model[x])
        elif a == "5":
            # Application 5: Prints the value of similarity value between word x and y
            x = str(input("Please enter first word: "))
            y = str(input("Please enter second word: "))
            print("similarity value = " + str(model.similarity(x, y)))
        elif a == "6":
            # Application 6: Prints words similar to x but not y (this method is identical to RELATIONSHIP GUESSER but here we can enter multiple words)
            x = str(input("Please enter positive words: ")).split(" ")
            y = str(input("Please enter negative words: ")).split(" ")
            print(model.most_similar(positive=x, negative=y, topn=10))
        elif a == "7":
            # Application 7: DO MAGIC HERE
            x = str(input("Please enter a word: "))
            if x == '-1':
                break
#            print(model[x])
#            print(model1[x])
#            print(model2[x])
            norm1 = np.linalg.norm(model[x])
            print("norm1:" + norm1)
            norm2 = np.linalg.norm(model1[x])
            print("norm2:" + norm2)
            norm3 = np.linalg.norm(model2[x])
            print("norm3:" + norm3)
            avgNorm = np.average([norm1,norm2,norm3])
            print("avg norm:" + avgNorm)
            sumNorm = norm1+norm2+norm3
            print("sun norm:" + sumNorm)
            dot1 = np.dot(model[x],model1[x])
            print("dot1:" + dot1)
            dot2 = np.dot(model[x],model2[x])
            print("dot2:" + dot2)
            dot3 = np.dot(model1[x],model2[x])
            print("dot3:" + dot3)
            sumDot = dot1+dot2+dot3
            print("sum dot:" + sumDot)
            avgDot = np.average([dot1,dot2,dot3])
            print("avg dot:" + avgDot)
            #op2 = 
            print("first norm"+norm1)
#            print("2nd & 3rd norm avg"+(norm2+norm3)/2)
#            print("norm of first: ", norm1)
#            print("norm of 2nd: ", norm2)
#            print("norm of 3rd: ", norm3)
#            print("average norm: %3.9f " % (avgNorm))
#            print("average dot: %3.9f"% (avgDot))
#            print("avgNorm/avgDot: %3.9f" % (avgNorm/avgDot))
#            print("(norm1/norm2-norm3)/sumDot: %3.9f"%((norm1/(norm2-norm3))/sumDot))
#            print("(norm1/norm2+norm3)/sumDot: %3.9f"%((norm1/(norm2+norm3))/sumDot))
#            print("(norm1/norm2-norm3)/avgDot: %3.9f"%((norm1/(norm2-norm3))/avgDot))
#            print("(norm1/norm2+norm3)/avgDot: %3.9f"%((norm1/(norm2+norm3))/avgDot))
#            print("Norm subtraction",np.linalg.norm(model[x]) - np.linalg.norm(model1[x]) - np.linalg.norm(model2[x]))
#            print("subtract then take the norm: first and 2nd", np.linalg.norm(np.subtract(model[x],model1[x])))
#            print("subtract then take the norm: first and 3rd", np.linalg.norm(np.subtract(model[x],model2[x])))
#            print("subtract then take the norm: 3rd and 2nd", np.linalg.norm(np.subtract(model1[x],model2[x])))
#            print("subtract then take the norm: (first + 2nd) from first", (norm1 -(norm2+norm3))) 
#            print("dot product 1x2: ", dot1)
#            print("dot product 1x3: ", dot2)
#            print("dot product 2x3: ", dot3)
        else:
            print("wrong input.. please enter a number from 1 to 5..")
    except Exception as e:
        print("Oops!  A word was not found (maybe mentioned less than minimum count)")
        print("error message: " + repr(e))
        