from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize

ne_chunk(pos_tag(word_tokenize("George joined Udacity in Califonia")))