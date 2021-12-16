import tensorflow as tf

g = tf.Graph()
fn = ""

with g.as_default() as g:
    tf.compat.v1.train.import_meta_graph(f'./{fn}/.meta')

with tf.compat.v1.Session(graph=g) as sess:
    file_writer = tf.compat.v1.summary.FileWriter(logdir=f'./{fn}/', graph=g)