from MarioRNN import MarioRNN
from TrainData import DataSet
import configparser
import sys

configFilename = "sample.cfg"

if len(sys.argv) >= 2:
	configFilename = sys.argv[1]
	
config = configparser.ConfigParser()
config.read(["defaults.cfg", configFilename])


def get_data(training):
	data = DataSet(
		filenames = config.get("Data", "Filename").strip().split('\n'),
		sequence_len = int(config.get("Data", "SequenceLength")),
		batch_size = int(config.get("Data", "BatchSize")),
		train = training,
		num_passes = int(config.get("Train", "NumPasses")),
		recur_buttons = config.get("Data", "RecurButtons") == "True"
	)
	
	return data
	
def get_model(data, training):
	rnn_sizes = []
	layer = 1
	while True:
		try:
			size = int(config.get("RNN", "Layer" + str(layer)))
			if size < 1:
				break
			rnn_sizes.append(size)
			layer = layer + 1
		except:
			break
			
	print("RNN Sizes: " + str(rnn_sizes))

	model = MarioRNN(
		data=data,
		rnn_sizes=rnn_sizes,
		max_grad_norm=float(config.get("Train", "MaxGradNorm")),
		dropout_keep=float(config.get("Train", "DropoutKeep")),
		variational_recurrent= config.get("Train", "VariationalRecurrent") == "True",
		train=training,
		loss_function=config.get("Train","LossFunction")
	)
	
	return model
	
def get_checkpoint_dir():
	return config.get("Checkpoint", "Dir")

def get_validation_period():
	return float(config.get("Train", "ValidationPeriod"))