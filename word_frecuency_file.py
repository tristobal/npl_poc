import nltk
from nltk.corpus import stopwords

file = open("speechs/20191117.txt", "rb")
filecontext = file.read().decode("utf-8")
tokens = [t for t in filecontext.split()]

sr = stopwords.words('spanish')
clean_tokens = [token for token in tokens if token not in sr]
freq = nltk.FreqDist(clean_tokens)

sorted_freq = sorted(freq.items(), key=lambda f: f[1], reverse=True)
print(dict(sorted_freq))

freq.plot(20)