from collections import Counter
import os
import zipfile

from six.moves import urllib
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time

# Define paramaters
VOCAB_SIZE = 10000
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
EXPECTED_BYTES = 31344016
DATA_FOLDER = '../data/'
FILE_NAME = 'text8.zip'
EMBED_SIZE = 128 # dimension of the word embedding vectors

def download(file_name, expected_bytes):
  """ Download the dataset text8 if it's not already downloaded """
  file_path = DATA_FOLDER + file_name
  if os.path.exists(file_path):
    print("Dataset ready")
    return file_path
  file_name, _ = urllib.request.urlretrieve(DOWNLOAD_URL + file_name, file_path)
  file_stat = os.stat(file_path)
  if file_stat.st_size == expected_bytes:
    print('Successfully downloaded the file', file_name)
  else:
    raise Exception('File ' + file_name +
                    ' might be corrupted. You should try downloading it with a browser.')
  return file_path

def read_data(file_path):
  """ Read data into a list of tokens 
  There should be 17,005,207 tokens
  """
  with zipfile.ZipFile(file_path) as f:
    words = tf.compat.as_str(f.read(f.namelist()[0])).split() 
    # tf.compat.as_str() converts the input into the string
  return words

def build_vocab(words, vocab_size):
  """ Build vocabulary of VOCAB_SIZE most frequent words excluding UNK """
  dictionary = dict()
  count = [('UNK', -1)]
  count.extend(Counter(words).most_common(vocab_size))
  count.remove(('UNK',-1))
  index = 0
  for word, _ in count:
    dictionary[word] = index
    index += 1
  index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return dictionary, index_dictionary

def build_cooccurence_mat(words, dictionary):

	print('')
	print('Building co-occurrence matrix...')
	start_time = time.time()
	# initialize A matrix
	A = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=int)

	# iterate words
	for i in range(len(words)-1):
		row_idx = dictionary.get(words[i], -1)
		col_idx = dictionary.get(words[i+1], -1)

		if row_idx >= 0 and col_idx >= 0:
			A[row_idx][col_idx] += 1

	print 'Total time: {0} seconds'.format(time.time() - start_time)

	return A

file_path = download(FILE_NAME, EXPECTED_BYTES)
words = read_data(file_path)
dictionary, _ = build_vocab(words, VOCAB_SIZE) 
A = build_cooccurence_mat(words, dictionary)

# create placeholders for co-occurrence matrix
X = tf.placeholder(tf.float32, [VOCAB_SIZE, VOCAB_SIZE], name='co_occurrence_matrix') 

# run SVD
s, u, v = tf.svd(X)

with tf.Session() as sess:

	print('')
	print('Running SVD...')
	# to visualize using TensorBoard
	writer = tf.summary.FileWriter('./graphs/a1', sess.graph)

	start_time = time.time()

	s, u, v = sess.run([s, u, v], feed_dict={X: A})

	print 'Total time: {0} seconds'.format(time.time() - start_time)

	print('SVD Finished!')

	writer.close()

# Find embedding of words in EMBED_SIZE dimension
embedding = u[:,:EMBED_SIZE]
print(embedding.shape)

