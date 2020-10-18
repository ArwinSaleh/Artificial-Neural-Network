import tensorflow as tf

with tf.compat.v1.Session() as sess:
  a = tf.constant(3.0)  
  b = tf.constant(4.0)
  c = a+b
  print(sess.run(c))