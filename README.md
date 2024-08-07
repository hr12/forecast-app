# forecast-app
A simple forecasting application to train, evaluate, and forecast on time series data.

## Steps to Execute This End-to-End

1. **Create a New Conda or Pip Environment:**
   - Using Conda:
     ```sh
     conda create --name forecast-app
     conda activate forecast-app
     ```

2. **Install Packages Mentioned in `requirements.txt`:**
   - Make sure you have a `requirements.txt` file in the root folder.
   - Install the required packages:
     ```sh
     conda install --file requirements.txt
     ```

3. **Add a `data` Folder in the Root Folder:**
   - Create a folder named `data` in the root directory of your project. This folder will be used to store your time series data files.
     ```sh
     mkdir data
     ```

4. **Execute the Code:**
   - Run the application by executing:
     ```sh
     python app.py
     ```
