"""
The predictor is an API for the prediction logic and exposes
1. Build
        The build will take in the TrainTimeSeries and will build a NN that is relevant for the TrainingTimeSeries
        i.e have the right number of input and op layers. Also will take as an input the number of hidden layer neurons
2. Train
        The train will then run a training on the data present in the TrainTimeSeries
3. Predict
        The predict will take an input that is similar to the input in the TrainingTimeSeries and will produce OPs

"""