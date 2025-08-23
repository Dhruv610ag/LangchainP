import boto3
import json

#we need to go to the model access and request for thr model we will be using and always select us-east-1 or us-west-2 as they have many models just request them and take the access
#in terminal me need to write the aws configure and it will ask for the ids we need to provide that id's to it  also area and jason ka bhi puchenge vo bhi batana hai 
 
prompt_data="""
Act as a Shakespeare and write a poem on Generative AI
"""

bedrock=boto3.client(service_name="bedrock-runtime")

payload={
    "prompt":"[INST]"+ prompt_data +"[/INST]",
    "max_gen_len":512,
    "temperature":0.5,
    "top_p":0.9
}
body=json.dumps(payload)
model_id="meta.llama2-70b-chat-v1"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body=json.loads(response.get("body").read())
repsonse_text=response_body['generation']
print(repsonse_text)