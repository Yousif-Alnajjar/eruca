Eruca - a program that uses a long short-term memory network, a subset of recursive neural networks, to predict the price of Bitcoin.
This prediction is used to algorithmically buy and sell bitcoin at price levels based on a user's selected risk level.

To use the program, run app.py.

Required libraries: numpy, cmu_graphics, PIL, yfinance, math, copy, datetime

app.py contains the UI elements of the application, as well as the function that trains the LSTM model
LSTM.py contains the functions requrired to train an LSTM
LSTMClass.py contains a node class that represents an individual layer in the network