# Image Classification using AWS SageMaker

This project will be using AWS Sagemaker to finetune a pretrained model that can perform image classification. We will have to 
use Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices to finish this project. 
The choice of dataset is the dog breed classication dataset to classify between different breeds of dogs in images.

## Project Set Up and Installation
Enter AWS and open SageMaker. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant we can use it to complete the project.

## Files
1. train_and_deploy.ipynb
Fetch, Debug filing report, deploy best training job, query model endpoint
2. train.py
The train_model.py script is what will train the model,
3. hpo.py (for hyperparameter optimization)
Used for hyperparameter tuning
4. PDF/HTML of the Profiling Report
The Debugger ProfilerReport rule invokes all of the monitoring and profiling rules and aggregates the rule analysis into a comprehensive report
5. README.md
File that describes the project, explains how to set up and run the code, and describes your results.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

**Screenshot** 
![alt text](https://github.com/beauvilerobed/image-classification-AWS-sagemaker/blob/main/Screen%20Shot%202021-12-23%20at%207.08.25%20PM.png?raw=true)


## Profiling
Add rules in rules list.
Create the profilier and debugger configurations.
Create the estimator to train your model.

## Debugging
Create Hook
Set hook to track the loss
Set hook to train mode
Set hook to eval mode

### Results and insights

One key insight would be to checks how many data loaders are running in parallel and whether the total number is equal the number
of available CPU cores. The rule triggers if number is much smaller or larger than the number of available cores.
If too small, it might lead to low GPU utilization. If too large, it might impact other compute intensive operations on CPU

One result would be The StepOutlier rule measures step durations and checks for outliers. The rule 
returns True if duration is larger than stddev times the standard deviation. The rule 
also takes the parameter mode, that specifies whether steps from training or validation phase 
should be checked.

## Model Querying
```
import torch
import torchvision
from torchvision import transforms

testing_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


testset = torchvision.datasets.ImageFolder(root="sample_image", 
            transform=testing_transform)
image = torch.utils.data.DataLoader(testset)
for input, label in image:
    print(predictor.predict(input))
```

**Screenshot** of the deployed active endpoint in Sagemaker.
![alt text](https://github.com/beauvilerobed/image-classification-AWS-sagemaker/blob/main/Screen%20Shot%202021-12-25%20at%2010.54.25%20PM.png?raw=true)

