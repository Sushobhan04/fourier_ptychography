import tensorflow as tf

x = tf.linspace(0., 1., 10)
z = tf.complex(x, x)
expr = tf.reduce_sum(tf.abs(tf.fft(z)))

print expr

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

print sess.run(tf.gradients(expr, z))