import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
# from Forecaster import Forecaster
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
warnings.filterwarnings('ignore')

dashed_line = '='.join(['' for i in range(25)])

class ForecastEval:
    """_summary_
    Class to Take a Forecaster object and take upt cross validation and in-time and out-of-time evaluation
    
    """
    def __init__(self, forecaster)->None:
        self.forecaster = forecaster
        
    def cross_validate(self):
        """_summary_
        Method to do cross validation. It's currently set to include the initial 180 days in the modeling and evaluate over a horizon of 1 day after every 15 days on rolling basis.
        """
        
        print("Entering Cross Validation....")
        self.cv_df = cross_validation(self.forecaster.model,
                                 initial='180 days',
                                 period='15 days',
                                 horizon = '1 day'
                                 )
        self.performance_metrics = performance_metrics(self.cv_df, rolling_window=0, metrics=['rmse','mape'])
        print(dashed_line)
        print("Cross Validation metrics")
        print(self.performance_metrics.iloc[-2:,:])
        print(dashed_line)
        
        
        # tcv = TimeSeriesSplit(n_splits=n_splits)
        # mse_scores = []
        # mape_scores = []

        # for train_index, test_index in tcv.split(self.df):
        #     train, test = self.df.iloc[train_index], self.df.iloc[test_index]
        #     cv_model = Prophet()
        #     cv_model.fit(train)
        #     forecasts = cv_model.forecast(len(test))
        #     mse_scores.append(mean_squared_error(test,forecasts))
        #     mape_scores.append(mean_absolute_percentage_error(test,forecasts))

        # avg_mse = np.mean(mse_scores)
        # avg_mape = np.mean(mape_scores)
        # print(f'Average Cross-Validation MSE: {avg_mse}')
        # print(f'Average Cross-Validation MAPE: {avg_mape}')
        
    def eval_model(self):
        """_summary_
        Method to evalute model on in time and out of time data 
        builds model on the train split and evaluates on the out of time test data and print the metrics
        """
        
        print("Entering Model Validation....")
        
        
        eval_model = Prophet(
                        interval_width=0.95,
                        daily_seasonality=True,
                        weekly_seasonality=True,
                        yearly_seasonality=True,
                        changepoint_prior_scale=0.05
                        )
        
        eval_model.add_regressor('actual_temperature')

        eval_model.fit(self.forecaster.train)
        
        future = eval_model.make_future_dataframe(len(self.forecaster.test),freq='15min'
                # , include_history=False
                                      )
        
        future.loc[:,'actual_temperature'] = self.forecaster.df.loc[:,'actual_temperature']
        self.eval_forecast_df = eval_model.predict(future)
                
        ## in time evaluation
        # fitted_values = eval_model.
        
        y_pred_in_time = self.eval_forecast_df['yhat'][:len(self.forecaster.train)].values
        y_actual_in_time =  self.forecaster.train['y']
        self.rmse_in_time = np.sqrt(mean_squared_error(y_actual_in_time, y_pred_in_time))
        self.mape_in_time = mean_absolute_percentage_error(y_actual_in_time, y_pred_in_time)
        print(dashed_line)
        print("In time metrics:\nRMSE:{0}\nMAPE:{1}".format(self.rmse_in_time,self.mape_in_time))
        print(dashed_line)
        
        ## out of time evaluation
        
        y_pred_oot = self.eval_forecast_df['yhat'][-len(self.forecaster.test):].values
        y_actual_oot =  self.forecaster.test['y']
        print(y_pred_oot.shape,y_actual_oot.shape)
        self.rmse_oot = np.sqrt(mean_squared_error(y_actual_oot, y_pred_oot))
        self.mape_oot = mean_absolute_percentage_error(y_actual_oot, y_pred_oot)
        print(dashed_line)
        print("OOT metrics:\nRMSE:{0}\nMAPE:{1}".format(self.rmse_oot,self.mape_oot))
        print(dashed_line)