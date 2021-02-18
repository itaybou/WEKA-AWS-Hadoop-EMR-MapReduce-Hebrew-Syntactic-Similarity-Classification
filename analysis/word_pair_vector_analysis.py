from pathlib import Path
from os.path import join
from glob import glob
from collections import defaultdict
from nltk.stem import PorterStemmer
from pprint import pprint

ps = PorterStemmer()

folder = '15-files'

vector_path = join(Path().resolve().parent, folder, 'similarity_vectors/output/co-occurrence_vectors')
input_files = glob(join(vector_path, 'part*'))
print(vector_path)

false_positive_instances = ['barrel,revolver', 'food,stove', 'hospital,school']
false_negative_instances = ['carnivore,lizard', 'system,television', 'aeroplane,fighter']

def get_common_uncommon_features(word_pairs):
   stemmed_word_pairs = {pair:tuple(map(lambda word: ps.stem(word), pair.split(','))) for pair in word_pairs}

   stemmed_words = set()
   lexeme_dict = defaultdict(lambda: set())

   for word_pair in word_pairs:
      words = word_pair.split(',')
      for word in words:
         stemmed_words.add(ps.stem(word))

   for file in input_files:
      with open(file, 'r') as vector_file:
         lines = vector_file.readlines()
         for line in lines:
            splitted = line.split('\t')
            if splitted[0] in stemmed_words:
               features = splitted[2].split(',')
               for f in features:
                  splitted_f = f.strip().split(':')
                  lexeme_dict[splitted[0]].add(splitted_f[0])

   common_features = {}
   for pair, (lexeme1, lexeme2) in stemmed_word_pairs.items():
      common = lexeme_dict[lexeme1].intersection(lexeme_dict[lexeme2])
      un_common = lexeme_dict[lexeme1].symmetric_difference(lexeme_dict[lexeme2])
      common_features[pair] = (common, len(common), len(un_common))

   return common_features

fp_features = get_common_uncommon_features(false_positive_instances)
fn_features = get_common_uncommon_features(false_negative_instances)

print('False-Positive:')
for word_pair in fp_features.items():
   print(f'{word_pair}\n\n')

print('False-Negative:')
for word_pair in fn_features.items():
   print(f'{word_pair}\n\n')

