# Udacity Azure Machine Learning Capstone Project

This is 3rd project part of the Udacity Azure ML Nanodegree. In this project, we will use the external dataset which is not present in azure machine learning studio.  
We will be using HyperDrive to tune the hyperparameters of scikit learn algorithm to create best model and Azure AutoML to find the best model. We will compare the both the models and deploy the best model to Azure Container Service as a REST endpoint with key based authentication. We will send the POST request to check Endpoint is working.  
In this project we will be using Jupyter Notebook from traing to deployment instead of Azure ML Studio as it gives more flexibility.   
  
  
## Project Set Up and Installation  
  
For setup and installing dependencies please see [Enviornment file details](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/tree/main/Enviorment_Dependencies)       
   
**Compute Cluster**    
    
![Compute Cluster](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/compute_cluster.png)        
    
  
**Note :** I was having trouble in retrieving the trained AutoMl model due to differences in SDK version. So, before running the notebooks please run the script 
[update_azure_SDK](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/scripts/update_azure_SDK.ksh) to make the SDK version `1.20.0`. Please restart the kernel after running the script.   
  
## Dataset  
  
Data is taken from **Kaggle** 
  
### Overview  
  
I have uploaded the data to the GitHub repository. link to the [data](https://raw.githubusercontent.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/main/online_shoppers_intention.csv).  
Data consists of various Information related to customer behavior in online shopping websites. It helps us to perform Marketing Analytics.  
We will try to predict if shopper will generate Revenue or not from the data.  
    
**Registered Dataset**     
    
![Registered Dataset](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/registered_dataset.png)       
      
### Attribute Information   
  
The dataset consists of 10 numerical and 8 categorical attributes.  
The `Revenue` attribute is used as the class label.  
  
**1.** `"Administrative", "Administrative Duration", "Informational", "Informational Duration", "Product Related" and "Product Related Duration"` represent the number  of different types of pages visited by the visitor in that session and total time spent in each of these page categories.The values of these features are derived from the URL information of the pages visited by the user and updated in real time when a user takes an action, e.g. moving from one page to another.   
   
**2.** The `"Bounce Rate", "Exit Rate" and "Page Value"` features represent the metrics measured by "Google Analytics" for each page in the e-commerce site.      
          The value of `Bounce Rate` feature for a web page refers to the percentage of visitors who enter the site from that page and then leave ("bounce") without triggering any other requests to the analytics server during that session.       
          The value of `Exit Rate` feature for a specific web page is calculated as for all pageviews to the page, the percentage that were the last in the session.    
          The `Page Value` feature represents the average value for a web page that a user visited before completing an e-commerce transaction.      
               
**3.** The `"Special Day"` feature indicates the closeness of the site visiting time to a specific special day (e.g. Mother’s Day, Valentine's Day) in which the sessions are more likely to be finalized with transaction. The value of this attribute is determined by considering the dynamics of e-commerce such as the duration between the order date and delivery date. For example, for Valentina’s day, this value takes a nonzero value between February 2 and February 12, zero before and after this date unless it is close to another special day, and its maximum value of 1 on February 8.     
   
**4.** The dataset also includes operating system, browser, region, traffic type, visitor type as returning or new visitor, a Boolean value indicating whether the date of the visit is weekend, and month of the year.    
  
TODO: Get data.   
We used method `from_delimited_files('URL')` from `TabularDatasetFactory` Class to retreive data from the csv file.[csv_file_link](https://raw.githubusercontent.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/main/online_shoppers_intention.csv).    
  


### Task
*TODO*: Explain the task you are going to be solving with this dataset .  
Objective of this project to create two ML models using HyperDrive and AutoMl and deploy the best model as REST API endpoint.  
Fort his data set we are predicting if the shopper will generate the Revenue or not. So, for this task we will be doing Classification. For HyperDrive we will be using Logistic regression from scikit-learn.   
Columns used in training [Attribute information](#attribute-information)

### Access
*TODO*: Explain how you are accessing the data in your workspace.  
We have uploaded the data to the GitHub repository. link to the [data](https://raw.githubusercontent.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/main/online_shoppers_intention.csv). 
We used method `from_delimited_files('URL')` from `TabularDatasetFactory` Class to retreive data from the csv file.     
`URL : (https://raw.githubusercontent.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/main/online_shoppers_intention.csv).    `    
We uploaded this data to the workspace and accessed using the `dataset = Dataset.get_by_name(workspace=ws, name='online_shoppers_intention')` in automl.ipynb.  
  
![Registered Dataset](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/registered_dataset.png)    
  
## Automated ML
     
[Automl Jupyter Notebook](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/automl.ipynb)     
     
Completed the AutoML experiment.    
    
![Automlexpcmp](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/automl_cmplt_exp.png)    
  
**AutoML Settings**  
  
|Name|Description|Value
|----|-----------|-----
|experiment_timeout_minutes|It defines as how long experement will run|20
|max_concurrent_iterations|maximum number of iterations will run in parallel|5
|primary_metric|metric will be used to optimize and select the best model|accuracy
  
**AutoML Config**   
  
|Name|Description|Value
|----|-----------|-----
|compute_target|Used for training(to run the experiment on)|compute_target=project3
|task|Task we will be performing like 'Classification,'Regression' etc.|classification
|training_data|dataset to be trained on|dataset
|label_column_name|Coumn to be predicted|Revenue
|enable_early_stopping|early stopping if we find the best model|True
|featurization|featurization setting|auto
|debug_log|name of the file to store the debug log|automl_errors.log
  
  
### Results
The algorithm with the best performance during the tests was "VotingEnsemble" with a score of 0.90697.       
    
**RunDetails**      
     
![RunDetailesWidget](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/automl_rundetails.png)      
![Rundetails2](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/automl_rundetails2.png)        
              
Screenshot of the best model trained with it's parameters.               
         
![fittedmodel](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/automl_bestmodel_1.png)    
     
    
    
  
## Hyperparameter Tuning      
    
[HyperDrive Jupyter Notebook](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/hyperparameter_tuning.ipynb)        
    
Completed HyperDrive Experiment.    
    
![hyperdrive exp cmp](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/hyperdrive_experiment_cmp.png)     
        
We used scikit learn Logistic Regrssion algorithm for the predicion as this task is about Classification. We created train.py script to do that.  
[train.py](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/scripts/train.py)    
This script accepts three arguments.  
   1. C : Inverse of regularization `choice(1,2,3,4,5)`[For More Details](https://stackoverflow.com/questions/22851316/what-is-the-inverse-of-regularization-strength-in-logistic-regression-how-shoul)   
     
   2. max_iter : Maximum number of iterations taken for the solvers to converge. `choice(100,150,200,250)`   
   
   3. penalty(regularization) :l1 and l2 regularization `choice('l1', 'l2')`[For more Details](https://medium.com/analytics-vidhya/l1-vs-l2-regularization-which-is-better-d01068e6658c#:~:text=The%20main%20intuitive%20difference%20between,the%20data%20to%20avoid%20overfitting.&text=That%20value%20will%20also%20be%20the%20median%20of%20the%20data%20distribution%20mathematically.)  

***Tuning the Hyperparameters***    
 If we tune the HyperParameters manually it will take log time. So, we used the Azure HyperDrive tool to find and tune the best Hyperparameters. We followed the below steps.
   
   1.  Used RandomParameterSampler as it is fast as compared to grid search and saves the budget to exhaustively search over the search space.  
       The parameter search spcae used for *C* and *max_iter* is choice(1,2,3,4,5) and choice(100,150,200,250) resp.  
       [For more Details on RandomSampling](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)  
   2.  Specified the Early stopping policy as *BanditPolicy with slack_factor as 0.1, evaluation_interval as 1 and delay_evaluation as 5*   
       Bandit is an early termination policy based on slack factor/slack amount and evaluation interval.   
       The policy early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing     training run.  
       [For more Details on Bandit Policy](https://azure.github.io/azureml-sdk-for-r/reference/bandit_policy.html)


### Results  
The best Logistic Regrestion HyperDrive model with **accuracy 0.878345**  was found with below parametrs.     
For more details regarding the [parameters](#hyperparameter-tuning)     
|Parameter|Value
|---------|-----
|--C|4
|--max_iter|150
|--penalty|l2   

  
![Rundetails](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/hyperdrive_rundetails.png)  
   
![hyperdrive exp completed](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/hyperdrive_experiment_cmp.png)  
     
**Registering the Hyper Drive Model**       
     
![HyperDrivereg](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/hyperdrive_register_model.png)   

    
## Model Deployment   
**Registered the best models from Hyperdrive and AutoML**     
     
![Models](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/registerd_models.png)         
    
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.    
We can deploy a model using ACI(Azure container Instance),AKS(Azure Kubernatics Service) and local webservice with CPU/GPu based on our needs of processing the requests.   
We will be deploying the model created by AutoMl as the accuracy for automl model is greater than Hyperdrive model which is **3%** more i.e. **0.90**.  
    
For Deploying we will need below files.   
* Enviornment dependecies file. [link](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/Enviorment_Dependencies/envFile.yml)    
* Scoring script. [score.py](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/scripts/score.py)    
  
We can download the above files using below code.   
```
# Download scoring file 
automl_best_run.download_file('outputs/scoring_file_v_1_0_0.py', 'score.py')
# Download environment file
automl_best_run.download_file('outputs/conda_env_v_1_0_0.yml', 'envFile.yml')
automl_best_run.download_file('outputs/env_dependencies.json', 'envDependencies.json')
```
  
**Note :**  In `score.py` please add the `outputs` as shown in path as shown in below code after downloading.  
  `model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'outputs','model.pkl')`     
      
For this project I have used ACI with `cpu_cores = 1, memory_gb = 4`.Also, enabled `app insights and key based authorization.`      
    
![notebook deploy](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/deploying_notebook.png)  
      
**Deployed Model Status**       
     
![Model status](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/deployed_model_status.png)    
    
**Deployed Model**         
     
![Deployed model](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/deployed_model.png)     
      
**Enabled Application Insights**      
     
![App insights](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/deployed_model_application_insights_true.png)    
     
**Enabled Key Based Authentication**          
      
![Authentication](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/deployed_model_key_based.png)     
    
**To Query the Endpoint**    
    
We need to create the sample data. I created sample data with 10 records as jason file. As shown in below code block.  `df` is dataframe created from the dataset.  
```
import json
test_df = df.sample(10) 
test_df_label = test_df.pop('Revenue')
test_data = json.dumps({'data': test_df.to_dict(orient='records')})
```
   
We have enabled key based authentication. We will get the key fom the model and send the POST request as shown in below code block.  
```
import requests                           # to send the POST request
key = 'ujlbv7heiX5PQE2ksMl1lE7uIaPsyhHW'  # key of the model for authorization
headers = {'Content-type': 'application/json'}  # we are using json type content
headers['Authorization'] = f'Bearer {key}'      # adding key
response = requests.post(service.scoring_uri, test_data, headers=headers)  #sending the responsse to the endpoint
print(response)  # printing the received response
```
**      
  
**
      
## deleting the Service and Cluster   
   
![Delete](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/deleting_service_cluster.png)      
      
## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions   
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.    
I have enabled the application insights for logging. Find the screenshots below.     
  
**Enabled Application Insights**      
     
![App insights](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/deployed_model_application_insights_true.png)   
    
**Application Insights link working**   
![appliacation insight link](https://github.com/Bhosalenileshn/ML-Azure-Udacity-Capstone-Project/blob/main/screenshots/app_insight_link.png)  
   
