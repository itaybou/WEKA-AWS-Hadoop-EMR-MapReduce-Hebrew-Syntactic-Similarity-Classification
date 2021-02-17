from pathlib import Path
from os.path import join, splitext
from glob import glob
from collections import defaultdict
from nltk.stem import PorterStemmer

ps = PorterStemmer()

folder = '1-file'
path = join(Path().resolve().parent, folder, 'similarity_vectors/output/co-occurrence_vectors')
input_files = glob(join(path, 'part*'))

word_pairs = ['alligator,crocodile']
stemmed_words = set()
lexeme_dict = defaultdict(lambda: set())

for word_pair in word_pairs:
   words = word_pair.split(',')
   for word in words:
      stemmed_words.add(ps.stem(word))


print(stemmed_words)
for file in input_files:
   with open(file, 'r') as vector_file:
      lines = vector_file.readlines()
      for line in lines:
         splitted = line.split('\t')
         if splitted[0] in stemmed_words:
            features = splitted[2].split(',')
            for f in features:
               splitted_f = f.strip().split(':')
               lexeme_dict[splitted[0]].add((splitted_f[0], splitted[1], splitted_f[1]))

print(lexeme_dict)