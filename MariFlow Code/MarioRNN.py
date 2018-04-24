import tensorflow as tf
import math

class MarioRNN(object):
	def __init__(self, data, rnn_sizes, max_grad_norm, dropout_keep, variational_recurrent, train, loss_function):
		self.data = data
		self.dropout_keep = dropout_keep
		self.variational_recurrent = variational_recurrent
		self.loss_function = loss_function

		with tf.variable_scope("RNN"):
			self.layers = [tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True) for size in rnn_sizes]
	
		# Final layer of logistic activations for button presses
		self.logitsW = tf.Variable(tf.truncated_normal([rnn_sizes[-1], data.output_size], stddev=1.0 / math.sqrt(float(data.input_size))), name = "LogitsW")
		self.logitsB = tf.Variable(tf.zeros([data.output_size]), name = "LogitsB")
		self.max_grad_norm = max_grad_norm
		
		(self.initial_state, self.final_state, _, _, self.train_op) = self.build_graph(
			train=True,
			validate=False,
		)
			
		(_, _, _, self.cost, _) = self.build_graph(
			train=False,
			validate=True,
		)
		
		(self.single_initial_state, self.single_state, self.single_prediction, _, _) = self.build_graph(
			train=False,
			validate=False,
		)
		
	
	def build_graph(self, train, validate):
		data = self.data
		layers = self.layers
		max_grad_norm = self.max_grad_norm
		
		if train:
			layers = [tf.contrib.rnn.DropoutWrapper(
				layer,
				output_keep_prob=self.dropout_keep,
				variational_recurrent=self.variational_recurrent,
				dtype=data.dtype) for layer in layers]
			
		cell = tf.nn.rnn_cell.MultiRNNCell(layers, state_is_tuple=True)
			
		if train or validate:
			batch_size = data.batch_size
		else:
			batch_size = 1
		
		initial_state = state = cell.zero_state(batch_size, dtype = data.dtype)
		
		if train or validate:
			outputs = []
		
			with tf.variable_scope("RNN"):
				for time_step in range(data.num_steps):
					if time_step > 0:
						tf.get_variable_scope().reuse_variables()
					(cell_output, state) = cell(data.input[time_step, :, :], state)
					outputs.append(cell_output)
		else:
			self.single_input = tf.placeholder(shape=[1,data.input_size], dtype=data.dtype, name='single_input')
			initial_state = state = cell.zero_state(1, dtype=data.dtype)
			(cell_output, state) = cell(self.single_input, state)
				
		final_state = state
		
		if train or validate:
			logits = [tf.matmul(lstm_output, self.logitsW) + self.logitsB for lstm_output in outputs]
			predictions = tf.sigmoid(logits)
			
			if self.loss_function.lower() == "mean squared error":
				cost = tf.losses.mean_squared_error(
					data.output,
					predictions,
					weights=data.cost_weight
				)
			elif self.loss_function.lower() == "cross entropy":
				cost = tf.losses.sigmoid_cross_entropy(
					data.output,
					logits,
					weights=data.cost_weight
				)
			else:
				raise Exception('No such loss function: "{0}"'.format(self.loss_function))
		else:
			predictions = tf.sigmoid(tf.matmul(cell_output, self.logitsW) + self.logitsB, name='single_prediction')
			cost = None
		
		if train:	
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
			optimizer = tf.train.AdamOptimizer()
			train_op = optimizer.apply_gradients(
				zip(grads, tvars),
				global_step=tf.contrib.framework.get_or_create_global_step())
		else:
			train_op = None
		
		return initial_state, final_state, predictions, cost, train_op