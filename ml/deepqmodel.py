import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam


class DeepQModel():
    """ its a nuenalnetwork which will be trained to calculate Q values """

    def __init__(self, input_shape=9, output_shape=9, learning_rate=0.1, hidden_layer=[16, 32, 16]):
        """ define nueral network """
        self.input = input_shape
        self.output = output_shape
        self.hidden_layer = hidden_layer
        self.learning_rate = learning_rate
        self.model = self.makeModel()
        
    def makeModel(self):
        """ create the deep model """
        model = Sequential()
        depth = len(self.hidden_layer)
        model.add(Dense(self.hidden_layer[0], input_shape=(self.input,), activation='relu'))
        for n in range(1, depth):
            model.add(Dense(self.hidden_layer[n], activation='relu'))
        model.add(Dense(self.output, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate), metrics=['mse', 'accuracy'])
        return model
    
    def trainModel(self, x_train, y_train, epoch=1, batch=1):
        """ train the model """
        model = self.model
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch, verbose=0)
        self.model = model
    
    def evaluateModel(self, x, y):
        """ test the model is correct or not """
        self.model.evaluate(x, y, verbose=1)
    
    def getQValue(self, state):
        """ get the Q value for given state """
        return self.model.predict(state)
    
    def saveQModel(self, name="model.h5"):
        self.model.save(name)
    
    def loadModel(self, name):
        self.model = load_model(name)
    
    def networkAnalysis(self):
        """ do network analysis """
