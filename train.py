from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import argparse
import os
import numpy as np
import joblib
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"


### YOUR CODE HERE ###a



def clean_data(data):
    # Dict for cleaning data
    x_df = data.drop(['Revenue','Month'],axis=1)
    y_df = data.iloc[:,-1].values
    y_df = np.reshape(y_df,(12330,1))
    label_encoder_x_df = LabelEncoder()
    x_df['VisitorType'] = label_encoder_x_df.fit_transform(x_df['VisitorType'])
    label_encoder_y_df = LabelEncoder()
    y_df = label_encoder_y_df.fit_transform(y_df)
    y_df = np.reshape(y_df,(12330,1))
    imputer = SimpleImputer(strategy='median')
    imputer = imputer.fit(x_df)
    x_df=imputer.transform(x_df)
    scaler = StandardScaler()
    x_df = scaler.fit_transform(x_df)

    # Clean and one hot encode data

    return x_df, y_df



def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--penalty', type=str, default='l2', help="Used to specify the norm used in the penalization")
    args = parser.parse_args()
    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    hype_loc = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
    ds = TabularDatasetFactory.from_delimited_files(path=hype_loc)
    x, y = clean_data(ds)
    # TODO: Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=10)
    model = LogisticRegression(C=args.C, max_iter=args.max_iter,penalty=args.penalty).fit(x_train, y_train)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model,filename='./outputs/model.joblib')

if __name__ == '__main__':
    main()
