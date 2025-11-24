import boto3
import time
import os

def trigger_training():
    sagemaker = boto3.client('sagemaker')
    
    # Configuration
    job_name = f"mvtec-ad-training-{int(time.time())}"
    role_name = "mvtec_sagemaker_execution_role"
    image_uri = os.environ.get("TRAINING_IMAGE_URI")
    
    if not image_uri:
        raise ValueError("TRAINING_IMAGE_URI environment variable not set")
        
    # Get Role ARN
    iam = boto3.client('iam')
    role = iam.get_role(RoleName=role_name)
    role_arn = role['Role']['Arn']
    
    print(f"Starting training job: {job_name}")
    print(f"Image: {image_uri}")
    print(f"Role: {role_arn}")
    
    response = sagemaker.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            'TrainingImage': image_uri,
            'TrainingInputMode': 'File',
        },
        RoleArn=role_arn,
        InputDataConfig=[
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': 's3://mvtec-ad-project-data/raw', # Adjust path as needed
                        'S3DataDistributionType': 'FullyReplicated',
                    }
                },
                'ContentType': 'application/x-image',
                'InputMode': 'File'
            }
        ],
        OutputDataConfig={
            'S3OutputPath': 's3://mvtec-ad-project-models/output'
        },
        ResourceConfig={
            'InstanceType': 'ml.g4dn.xlarge', # GPU instance
            'InstanceCount': 1,
            'VolumeSizeInGB': 50,
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 86400
        }
    )
    
    print(f"Training job started. ARN: {response['TrainingJobArn']}")

if __name__ == "__main__":
    trigger_training()
