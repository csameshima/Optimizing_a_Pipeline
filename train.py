#----------
# Import Libraries
#----------

#----- These libraries were provided in the original code
#----- They were sorted by type

#- General libraries
import argparse
import os
import joblib
import numpy as np
import pandas as pd

#- sklearn libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#- Azure ML libraries
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

#- This library was included to load the dataset
from azureml.core import Dataset





#----------
# Code
#----------

#----- The original code of this section was relocated to the last part of the script,
#----- since the clean_data function was called before it was defined


#- This function was provided in the original code

def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)
    
    #- This code was added because the original function was not returning expected values
    return x_df, y_df





#- This function was provided in the original code

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    #- Added creation of outputs folder
    os.makedirs('./outputs/model_h', exist_ok=True)
    joblib.dump(model, './outputs/model_h/model_h.joblib')



if __name__ == '__main__':

    #----- Begin relocated code
    
    # TODO: Create TabularDataset using TabularDatasetFactory
    # Data is located at:
    # "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

    #- Define path to data
    web_path ='https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv'

    #- Create tabular dataset
    ds = Dataset.Tabular.from_delimited_files(path=web_path)

    #- Call function to clean dataset
    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.

    ### YOUR CODE HERE ###
    #- Split into 70-30 proportion, since is the general recommended value in the field
    #- Set random_state to 0, to ensure that the same random combination is used between runs
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

    #- Retrieve the current service context for logging metrics and uploading files
    run = Run.get_context()

    #----- End relocated code
        
    main()