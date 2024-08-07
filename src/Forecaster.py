import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

dashed_line = '='.join(['' for i in range(25)])


class Forecaster:
    
    def __init__(self, filepath) -> None:
        """_summary_

        Args:
            filepath (_type_): _description_
        """
        
        self.filepath = filepath
        self.model = None
        self.start_date = '2012-11-01' ## hardcoded in this example, but can be parameterized
        self.end_date = '2013-12-01'## hardcoded in this example, but can be parameterized
        self.train_ratio = 0.98 ## keeping a high train ratio since we need to forecast only for the next 24 hours
        self.required_columns = ['actual_kwh','actual_temperature'] ## required columns in the input data
        self.fc_file_path = 'data/forecasts.csv' ## file path to store the forecast values
        # pass
    
    def assert_columns(self,df, required_columns):
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        assert not missing_columns, "Missing columns :{}".format(missing_columns)

    
    def read_and_preprocess_data(self):
        """_summary_
        Reads the data, checks whether the required columns exist. converts the date-times into the right format for Prophet model training.
        Fills in missing values with average in case of actual_kwh and last known temperature value in the case of 'actual_temperature'
        Does train test split. that can be used in evaluation
        
        """
        
        print(dashed_line)
        print("Reading in data....")
        df = pd.read_csv(self.filepath, sep=',' ,parse_dates=True, index_col=0, header=0,)
        print(df.dtypes)
        
        
        ## checking if the required columns are present or not
        try:
            self.assert_columns(df,self.required_columns)
        except AssertionError as e:
            print(e)  # Output: Missing columns: D
        
        ## the date-time column shoudl be name 'ds' as per prophet requirements
        # print(dashed_line)
        
        df.reset_index(inplace=True,drop=False,names='ds')
        df.ds = pd.to_datetime(df.ds, utc=True) ## converting to utc format
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None) ## another prophet requirement to remove localization
        df['y']=df['actual_kwh'] ## 'y' is the target column as per prophet
        df.drop('actual_kwh',axis=1,inplace=True)
        # print(dashed_line)        
        # print(df.shape)
        # print(df.head())
        
        
        ## filtering the data
        df = df[ (df['ds'] > self.start_date)&(df['ds'] < self.end_date)]
        # print(df[df.actual_kwh.isnull()]['ds'].describe())
        # df.head()
        df.sort_values(by='ds',inplace=True)
        # print(dashed_line)
        # print(df.head())
        
        ## filling in the null values in 'actual_temperature' with the last known value
        df['actual_temperature'].fillna(method='ffill', inplace=True)
        
        self.df = df
        print("df's shape :{}".format(self.df.shape))
        # print(self.df.shape)
        
        ## train-test split
        train_size = int(len(df) * self.train_ratio)
        print(train_size)
        self.train, self.test = df[0:train_size], df[train_size:len(df)]
        
        ## filling null values in train df with mean ## can be done in more smarter way
        self.train['y'] = self.train['y'].fillna(self.train['y'].mean()) 
        print('Train size:{} Test size:{}'.format(self.train.shape, self.test.shape))
        
                
    def train_model(self):
        """_summary_
        A simple Prophet model training with an additional regressor
        """
        
        # self.read_and_preprocess_data()
        
        print(dashed_line)
        print("Initiating training...")
        
        self.model = Prophet(
            interval_width=0.95,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
            # growth='logistic'
            # changepoint_prior_scale=0.05
        )
        self.model.add_regressor('actual_temperature') ## adding regressor
        self.model.fit(self.df)
        print("Completed training the model on the entire input data")

    def forecast(self, n_periods=96):
        """_summary_
        
        Method to forecast into the future

        Args:
            n_periods (int, optional): _description_. Defaults to 96.
        """
        
        if self.model == None:
            print("Model not trained")
            print("Training the model....")
            self.train_model()
        
        print(" Forecasting for the next {} periods...".format(n_periods))
        future = self.model.make_future_dataframe(n_periods,freq='15min'
                , include_history=False
                                      )
        
        ## import actual future forecast temperatures
        ## in the absence of future temperatures, we take last n_periods temperatures as a placeholder
        future.loc[:,'actual_temperature'] = self.df.loc[:,'actual_temperature'].iloc[-n_periods:].values
        self.forecast_df = self.model.predict(future)
        self.forecast_df.index=self.forecast_df.ds
                
        self.forecast_df.loc[:,'yhat'].to_csv(self.fc_file_path, index=True)
        
        print(dashed_line)
        print('Forecasts head:')
        print(self.forecast_df.head(10))
        print(dashed_line)