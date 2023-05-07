# amazon-ctr

## Access the API ## 

Use the curl command below:

```
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "Does iris grow best outdoors or indoors?"}' \
  http://alexa-ctr-load-balancer-1949114071.us-east-2.elb.amazonaws.com/predict
```
 
## Model Development and Training ##
The dataset contains conversations on different real-world tasks: Cooking and Home Improvement. Each conversation has multiple texts shared back and forth, each text is assigned one of 11 intents from: ['ask_question_ingredients_tools' 'return_list_ingredients_tools', 'request_next_step', 'return_next_step', 'ask_question_recipe_steps', 'answer_question_recipe_steps', 'answer_question_external_fact', 'stop', 'ask_student_question', 'chitchat', 'misc']

On a high-level, a model that takes into account each previous text in a conversation into consideration while making predictions for the follow up text should be targeted for this use-case. However, given the overall task, deliverables, timeline, compute resources at-hand and other commitments, I wanted to take an approach that would take minimal time to build and train (and be small enough to be loaded into a 1GB RAM VM for real-time inference) a model on what looks like a Multi-class classification task. 

One interesting part was that the intent labels contain valuable information if split into separate words. Doing so will make the task to be a semantic textual similarity task such that given an input text, we would calculate similarity scores between each of the labels (split into different words) and the label with the highest score would be the final prediction.


### Base Model and Training Strategy ###
Model Type: [Sentence Transformer Bi-Encoder](https://www.sbert.net/docs/pretrained_models.html)

Base Model: [Sentence-Transformer all-minilm-l12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) (134MB)

Although, the above model ranks 4th on the Sentence Embeddings Performance, it was the smallest model that could be served on a low-memory VM with still decent comparable performance (~1.2% difference than the first).

Fine-Tune on: Train conversations where each text (non-empty) is used as a separate example. Same process is used to make the validation set too. 

Training Approach: Training Bi-Encoders with in-batch hard negatives (giving a score of 0 to other intents to a text) on a contrastive loss objective can be very beneficial for the training process. This also served as the motivation to use this in comparison to a cross-encoder that only outputs a continuous score instead of giving meaningful embeddings which can be useful for semantic search type tasks in the future if we are planning on improving a newer model using this.  

So, I experimented by using a [ContrastiveLoss](https://www.sbert.net/docs/package_reference/losses.html?highlight=contrastive#contrastiveloss) and [OnlineContrastiveLoss](https://www.sbert.net/docs/package_reference/losses.html?highlight=contrastive#onlinecontrastiveloss) provided by the Sentence Transformer library. Out of the two, OnlineContrastive Loss gave the best performance. Online Contrastive Loss selects hard positives and hard negatives in order to update the loss function values which would justify achieving a higher performance. 

I also tried out two validation set evaluators: [Embedding Similarity Evaluator](https://www.sbert.net/docs/package_reference/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator) and [Binary Classification Evaluator](https://www.sbert.net/docs/package_reference/evaluation.html#sentence_transformers.evaluation.BinaryClassificationEvaluator); out of which Embedding Similarity Evaluator gave the best performance. 

Hyperparameter Search: I ran experiments by varying values of following parameters on the given ranges: 
1. Batch Size: [64, 128, 256] | Best Value: 64
2. Epochs: [2, 4, 8] | Best Value: 2
3. Warmup: [0.01, 0.1, 0.25, 0.5] | Best Value: 1
4. Weight Decay: 0.01 (Used default value for this)

You can track exact metrics for each runs here on [Wandb](https://wandb.ai/jaysinha/alexa-ctr?workspace=user-jaysinha). 

Test Accuracy: 79.356%

## Model Deployent ##

### Base Setup ###
My first thought was to use auto-scaling enabled [SageMaker Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html) with the model stored in S3 bucket and a [custom Docker Image](https://github.com/aws/amazon-sagemaker-examples/tree/main/advanced_functionality/scikit_bring_your_own/container) deployed to ECR with the inference logic which would make the future maintenance of these easier in the future. Unfortunately, SageMaker free-tier does not include ml.t2.micro instances which would have made the process a whole lot easier. 

So, my approach was to make a flask API which would keep the model in-memory and then return the label with the highest similarity score to the end-user. 

The inference logic looks something like below:

```python
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(request.json)

        # Generate model outputs
        with torch.no_grad():
            sentences1 = request.json['text']
            embedding1 = model.encode(sentences1)
            embedding2 = model.encode(sentences2)
            score = util.cos_sim(embedding1, embedding2).tolist()[0]
            predicted = targets[np.argmax(score)]

        # Extract the model output and return as JSON
        return jsonify({'statusCode': 200, 
                        "body": {
                                    "label": predicted,
                                    "score": score[np.argmax(score)]
                                }
        })
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'statusCode': 500, 
                        "body": tb
        })
```

The above flask application is exposed to port 5001. 

### Deploying the application to AWS ###

Now, we will deploy a t2.micro EC2 instance using the AWS Console:

1. Login to your AWS account and navigate to the EC2 dashboard.
2. Click on the "Launch Instance" button to start the process of launching a new EC2 instance.
3. Choose the Amazon Machine Image (AMI) that you want to use for your instance. The t2.micro instance supports Ubuntu 22.04LTS which we will be using for our experiment.
4. Choose the instance type as t2.micro from the list of available instance types. This instance type has 1 vCPU and 1 GB of memory by default.
5. Choose the appropriate configuration settings for your instance, such as VPC, subnet, security group, IAM role, etc. Add port 80 and 5001 inbound from 0.0.0.0/0 in your security group which would make it accessible from anywhere on the Internet.
6. On the "Add Storage" page, use the default storage size from 8GB . You can do this by modifying the "Size (GiB)" field.
7. Choose the appropriate storage type (e.g. General Purpose SSD or Magnetic) and configure any additional storage settings that you need.
8. Ignore all other fields as we will be using the defaults. 
9. Review your settings on the "Review Instance Launch" page, and then click on the "Launch" button.
10. You will be prompted to create or select a key pair for your instance. This key pair will be used to SSH into your instance. Follow the instructions to create or select a key pair, and then click on the "Launch Instances" button.
11. Your instance will now be launched and will be visible on the EC2 dashboard. Wait for a few minutes for the instance to become available.
12. Once the instance is available, you can connect to it using SSH. Use the public IP address or DNS name of the instance to connect to it.

First, we will have to setup the dependencies in order to run pip and the Sentence Transformers library (sentence-transformers, torch, transformers, flask) etc. Use the commands below:

```
sudo ssh -i <.pem key file> ubuntu@<IP_ADDRESS>
```

Installing full version of transformers requires the full version of torch which would too intensive for our 1GB RAM VM, so we install CPU versions of both and only use the required dependencies in the Sentence Transformer library (nltk, scikit-learn, scipy) in order for this to run efficiently.

```
sudo apt update
sudo apt install python3-pip3 -y
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu --quiet
pip3 install 'transformers[torch]' --quiet
pip3 install nltk scikit-learn scipy --quiet
pip3 install sentence-transformers --no-dependencies --quiet
```

Now, we will transfer the model files, app.py file (containing the flask application code) to the EC2 instance by using the command below:

```
sudo scp -i alexa-ctr.pem -r model/ ubuntu@<IP_ADDRESS>:/home/ubuntu
sudo scp -i alexa-ctr.pem app.py ubuntu@<IP_ADDRESS>:/home/ubuntu
```

Test your app once by running the below command in /home/ubuntu directory:

```
python3 app.py
```

Test your service from within the server or outside:

```
curl http://localhost:5001
curl http://<SERVER_IP_ADDRESS>:5001
```

which should return:

```
{
    "body": "Use command: curl -X POST                     -H \"Content-Type: application/json\"                     -d '{\"text\": \"The first thing you should do is make sure you open the zipper as much as possible.\"}'                     http://localhost:5001/predict",
    "statusCode": 200
}
```

We will have to make the flask application as a system service such that the application is started even after the server restarts etc. We will use the below script to do so:

```
sudo vi /etc/systemd/system/autoscale.service
```

Paste the below in the file:
```
[Unit]
Description=Autoscale Flask project
After=network.target
[Service]
User=ubuntu
Group=www-data
ExecStart=/usr/bin/python3 /home/ubuntu/app.py
[Install]
WantedBy=multi-user.target
```

Use the below commands to register and start the application as a service in linux:
```
sudo systemctl start autoscale.service
sudo systemctl enable autoscale.service
```

Check the status of the above created service which should be running: 

```
sudo systemctl status autoscale.service
```

Upon fixing (if needed), the above setup:
```
sudo systemctl restart autoscale.service
```

### Deploying the application as a scalable service ###
Now, we are ready to deploy this as a scalable EC2 service on a Elastic Load Balancer. Only proceed with the following if everything is working okay before this.

1. We will use the EC2 instance deployed above as a deployment configuration for all other instances, to do this goto Instances under EC2 and select the instance. Click on Action, scroll to Image and Templates => Create Image. This would create an AMI which would be used by all the instances added during Autoscaling.
![Image Creation](/images/image_creation.png)
2. Next, we will create a Launch Template configuration using the above AMI Image. 
    
    a. Click on Launch Templates in the left hand bar and click on Create Launch Template. Assign the AMI a name. 
        ![Create Launch Template](/images/launch-template-1.png)

    b. Scroll down and select the AMI you created in the above step. 
        ![Select the AMI](/images/launch-template-2.png)

    c. Select the instance type (t2.micro), key (used to deploy the above EC2 instance) and the security group that was created in the above EC2 instance. Finally click on Create Launch Template on right. 
        ![More config](/images/launch-template-3.png)

3. Now, we will create an Elastic Load Balancer. To do so, click on Load Balancer in the left navbar and click on Create Load Balancer button.

    a. Select Application Load Balancer. 
        ![Create ELB](/images/load-balancer-1.png)

    b. Assign a name to the ELB, select Internet-facing, the default VPC (which would be already selected).
        ![ELB Step 2](/images/load-balancer-2.png)

    c. Select two subnets and the security group which we used to create the first EC2 instance.
        ![ELB Step 3](/images/load-balancer-3.png)

    d. We will use the deault port 80 but need to create a Target group before we can finally create our Load Balancer. Click on Create Target Group hyperlink below the Target Group dropdown. (Follow Step 4 first)

    e. (Do this after finishing Step 4) Use the refresh the button on the right of Target Group, select the Target group you created from Step 4 and then click on Create Load Balancer. 

4. Create Target Group by using a name and selecting: Instances Target type and leave all other fields as defaults and create the target group. Go back to step 3 (e).

5. Now, we will create our Autoscaling Group which will use everything created above:

    a. Click on Autoscaling Group at the bottom of the left navigation bar, click on Create Auto Scaling Group and then assign it a name and use the Launch Template in the dropdown that we created in Step 2. Click Next.
        ![asg-1](/images/asg-1.png)

    b. Select the default VPC and two subnets from the drop down that we used in Step 3(c). Click Next.
        ![asg-2](/images/asg-2.png)

    c. Select Attach to an Existing Load Balancer and then select the one we created in Step 3. Use default values for the rest configurations in this page. Click Next.
        ![asg-3](/images/asg-3.png)
        ![asg-4](/images/asg-4.png)

    d.  Use Desired and Minimum Capacity = 1 and Maximum Capacity = 2. Use Target tracking scaling policy, Application Load Balancer Request count per target as your metric with threshold 10 and select the target group we created in Step 4. Click on Skip to Review.  
        ![asg-5](/images/asg-5.png)
        ![asg-6](/images/asg-6.png)

    e. Click on Create Auto Scaling group.
        ![asg-7](/images/asg-7.png)

Copy the Load Balancer URL by selecting the Load Balancer by copying the DNS Name created in Step 3: 

![lb-url](/images/load-balancer-url.png)

After executing all the above steps, your Autoscaling group will deploy a new EC2 instance using the AMI we created in Step 1 above. Now, we can delete the first EC2 instance we created in this setup such that we do not cross the 750hr/month free tier limit. 

Your Model is now deployed on a Autoscaling enabled EC2 instance. 

Some Justifications:
1. In Step 5 (d), we use Target Tracking Policy = Application Load Balancer Request per second to replicate the invocations per second setting that we configure in SageMaker real-time inference endpoints.
2. As of now, to keep the end-user onboarding friction to a minimum we do not have any authentication setup in the API. 

Future Improvements / Shortcomings:

1. As of now, the conversation texts with empty 'text' field were ignored while the formation of the training and validation sets.
2. The current deployment setup does not take into account a model refresh, ideally the model folder should be replaced with a new model that is being tracked on a S3 bucket. 
3. We do not have any monitoring metric setup for this model which would help us detect data drift, prediction quality, etc. 
4. As of now, we keep of log of incoming requests as application log but ideally we should store incoming requests in a data base.
5. We do not have any DDoS or authentication protection setup on the end API. 
6. There are not test cases setup for the minimal application logic we have. 


Time taken to complete each step:
1. Model Development and Training: ~6 hours.
2. Model Deployment & Autoscaling Setup (including dry runs - 2): ~4 hours
3. Documentation: ~2 hours
4. Code Cleanup & Repository Setup: ~1 hour

References for AWS Setup:
1. [Create an EC2 Instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)
2. [Setup Launch Template, Load Balancer, Autoscaling Group](https://docs.aws.amazon.com/autoscaling/ec2/userguide/tutorial-ec2-auto-scaling-load-balancer.html)
3. [Setup Launch Template Custom AMI](https://medium.com/analytics-vidhya/autoscale-on-aws-with-ec2-python-flask-and-nginx-part3-9ab0abeeeea6)