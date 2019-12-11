import re
import os
import sys

def read_file():
    with open(os.path.join(sys.path[0],'file.txt')) as f:
        text = f.read()
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
    print(sentences)

read_file()