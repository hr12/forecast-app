import os
from src.Forecaster import Forecaster
from src.ForecastEval import ForecastEval



if __name__ == "__main__":
    input_data_path = 'data/load_temperature_data.csv'
    
    ## instantiate a Forecaster class with input data
    fc = Forecaster(input_data_path)
    
    ## read and precprocess data
    fc.read_and_preprocess_data()
    
    ## train a forecasting model
    fc.train_model()
    
    ## forecast it for the next 24 hours; can be changed to longer duration by changing the n_periods param
    fc.forecast() ## forecasts exports the Forecasts to a directory
    
    
    
    #### Evaluation & Cross Validation
    ## instantiating a ForecastEval object
    fe = ForecastEval(fc)
    
    ## cross validation; prints forecast eval metrics by building model with different subsets of data 
    fe.cross_validate()
    
    ## in-time and out-of-evaluation to produce metrics RMSE and MAPE
    fe.eval_model()