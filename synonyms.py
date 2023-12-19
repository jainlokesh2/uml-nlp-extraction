# synonyms.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def get_synonyms(keyword):
    synsets = wordnet.synsets(keyword)
    return ', '.join(synsets[0].lemma_names()[:3]) if synsets else "No synonyms available."

def get_stopwords():
    return set(stopwords.words('english'))