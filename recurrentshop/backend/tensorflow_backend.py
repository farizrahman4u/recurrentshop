from keras.backend import tensorflow_backend as K
import tensorflow as tf


class learning_phase_scope(object):

	def __init__(self, value):
		self.value = value

	def __enter__(self):
		K.set_learning_phase(self.value)
		self.learning_phase_placeholder = K.learning_phase()

	def __exit__(self, *args):
		K._GRAPH_LEARNING_PHASES[tf.get_default_graph()] = self.learning_phase_placeholder
