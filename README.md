# Image Classification using AWS SageMaker

This project will be using AWS Sagemaker to finetune a pretrained model that can perform image classification. We will have to 
use Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices to finish this project. 
The choice of dataset is the dog breed classication dataset to classify between different breeds of dogs in images.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

## Files
1. train_and_deploy.ipynb
Fetch, Debug filing report, deploy best training job, query model endpoint
2. train.py
The train_model.py script is what will train the model,
3. hpo.py (for hyperparameter optimization)
Used for hyperparameter tuning
4. PDF/HTML of the Profiling Report
con
5. README.md
File that describes the project, explains how to set up and run the code, and describes your results.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
