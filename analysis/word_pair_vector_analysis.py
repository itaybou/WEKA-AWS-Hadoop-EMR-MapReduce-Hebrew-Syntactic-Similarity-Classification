from pathlib import Path
from os.path import join, splitext
from glob import glob
from collections import defaultdict
from nltk.stem import PorterStemmer

ps = PorterStemmer()

folder = '15-files'
arff_file_name = 'word_pair_similarity.arff'

vector_path = join(Path().resolve().parent, folder, 'similarity_vectors/output/co-occurrence_vectors')
input_files = glob(join(vector_path, 'part*'))

arff_path = join(Path().resolve().parent, folder, 'classifier', arff_file_name)
classifier_instance_ids = {}

false_positive_instances = [8794, 12232, 12973]
false_negative_instances = [5, 6, 7]

instance_counter = 1
with open(arff_path, 'r') as arff_file:
   lines = arff_file.readlines()
   for line in lines:
      if line.startswith('% <'):
         line = line.replace("%", "").replace("<", "").replace(">", "")
         classifier_instance_ids[line.strip()] = instance_counter
         instance_counter += 1

inv_classifier_instance_ids = {v: k for k, v in classifier_instance_ids.items()}

def get_common_uncommon_features(instances):
   word_pairs = [inv_classifier_instance_ids[i] for i in false_positive_instances]
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
      un_common = lexeme_dict[lexeme1].difference(lexeme_dict[lexeme2])
      common_features[pair] = (common, len(common), len(un_common))

   return common_features

print(get_common_uncommon_features(false_positive_instances))
