import tensorflow as tf

x = tf.constant(10.0)
y = tf.placeholder(tf.float32)
z = tf.placeholder(tf.float32)
#a = tf.placeholder(tf.float32)

product = y + z
addition = product + x

sess = tf.Session()
print(sess.run(addition, {y: 5.0, z: 2.0}))

sess.close()
