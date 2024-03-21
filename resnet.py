# import the necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
class ResNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
		# load the ResNet50 network, ensuring the head FC layer sets are left
		# off
		baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(64, 64, 3)))
		# construct the head of the model that will be placed on top of the
		# the base model
		headModel = baseModel.output
		headModel = Flatten(name="flatten")(headModel)
		headModel = Dense(64, activation="relu")(headModel)
		headModel = Dropout(0.5)(headModel)
		headModel = Dense(2, activation="softmax")(headModel)
		# place the head FC model on top of the base model (this will become
		# the actual model we will train)
		model = Model(inputs=baseModel.input, outputs=headModel)
		# loop over all layers in the base model and freeze them so they will
		# *not* be updated during the first training process
		for layer in baseModel.layers:
			layer.trainable = False
		# return the constructed network architecture
		return model
