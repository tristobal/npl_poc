from bs4 import BeautifulSoup
import urllib.request
import nltk
from nltk.corpus import stopwords

response = urllib.request.urlopen('http://php.net/')
html = response.read()
soup = BeautifulSoup(html, "html5lib")
text = soup.get_text(strip=True)

tokens = [t for t in text.split()]
sr = stopwords.words('english')
clean_tokens = [token for token in tokens if token not in sr]

freq = nltk.FreqDist(clean_tokens)
print(freq.items())

freq.plot(20)