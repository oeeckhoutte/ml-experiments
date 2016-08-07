import numpy as np

# We generate random points
num_points = 1000
vectors_set = []
for i in xrange(num_points):
	x1 = np.random.normal(0.0, 0.55)
	y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
	vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# The 2 lines below are useful on OS X to prevent the following error:
# RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are Working with Matplotlib in a virtual enviroment see 'Working with Matplotlib in Virtual environments' in the Matplotlib FAQ

import matplotlib as mpl
mpl.use('TkAgg')

# Display Random points
import matplotlib.pyplot as plt

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
plt.show()

import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b


# Cost function calculation 
loss = tf.reduce_mean(tf.square(y - y_data))

# We want to minimize the cost function
# We train the Optimizer which is the gradient descent algo to the cost function defined
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(8):
	sess.run(train)
	print(step, sess.run(W), sess.run(b))
	print(step, sess.run(loss))
	# Display Graphic
	plt.plot(x_data, y_data, 'ro')
	plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
	plt.xlabel('x')
	plt.xlim(-2, 2)
	plt.ylabel('y')
	plt.ylim(0.1, 0.6)
	plt.legend()
	plt.show()

	