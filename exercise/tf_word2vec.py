import tensorflow as tf

# Step 0: define global variables
BATCH_SIZE  = 50
VOCAB_SIZE  = 100
EMBED_SIZE  = 300
NUM_SAMPLED = 100
LEARNING_RATE = 0.01

# Step 1: create a placeholder
center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

# Step 2: create a variable for embedding matrix
embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0))

# Step 3: compute the forward path of graph
embed = tf.nn.embedding_lookup(embed_matrix, center_words)

# Step 4: define the loss function
nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0/EMBED_SIZE**0.5))
nce_bias   = tf.Variable(tf.zeros([VOCAB_SIZE]))
loss       = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weight,
                                           biases  = nce_bias,
                                           labels  = target_words,
                                           inputs  = embed,
                                           num_sampled = NUM_SAMPLED,
                                           num_classes = VOCAB_SIZE))

# Step 5: define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# Step 6: execute the computation
with tf.Session() as sess:

  # initialize the variables
  sess.run(tf.global_variables_initializer())

  #writer = tf.summary.FileWriter('./graphs', sess.graph)
  total_loss = 0.0
  for index in xrange(NUM_TRAINING_STEPS): # run 100 epochs
    batch = batch_gen.next()
    _, loss_batch = sess.run([optimizer, loss], feed_dict={center_word:batch[0], target_words:batch[1]})
    total_loss += loss_batch
    if (index+1) % 2000 == 0:
      print('Average loss at step {}: {:5.1f}'.format(index+1, total_loss/(index+1)))
