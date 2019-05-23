import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, (2, 1))

w = tf.constant([[3, 4]], tf.float32)
y = tf.matmul(w, x)

F = tf.pow(y, 2)
grads = tf.gradients(F, x)

session = tf.Session()

print(session.run(grads, {x: np.array([[2], [3]])}))

