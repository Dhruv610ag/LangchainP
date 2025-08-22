# Introducing Code Llama, a state-of-the-art large language model for coding
"""
We are releasing Code Llama 70B, the largest and best-performing model in the Code Llama family
Code Llama 70B is available in the same three versions as previously released Code Llama models, all free for research and commercial use:
CodeLlama - 70B, the foundational code model;
CodeLlama - 70B - Python, 70B specialized for Python;
and Code Llama - 70B - Instruct 70B, which is fine-tuned for understanding natural language instructions.
Code Llama is a state-of-the-art LLM capable of generating code, and natural language about code, from both code and natural language prompts.
Code Llama is free for research and commercial use.
Code Llama is built on top of Llama 2 and is available in three models:
Code Llama, the foundational code model;
Code Llama - Python specialized for Python;
and Code Llama - Instruct, which is fine-tuned for understanding natural language instructions.
In our own benchmark testing, Code Llama outperformed state-of-the-art publicly available LLMs on code tasks
code llama is the fine tune version of the of llama-2 fine tuned for the need of the code generation
"""
import requests
import json
import gradio as gr
#setting the temperature to be 1 so that the model becomes very creative default value is almost around 0.7 and 0.8 


url="http://localhost:11434/api/generate"

headers={

    'Content-Type':'application/json'
}

history=[]

def generate_response(prompt):
    history.append(prompt)
    final_prompt="\n".join(history)

    data={
        "model":"codeguru",
        "prompt":final_prompt,
        "stream":False# if we create the stream history to be True than it would a lot of unnecessary answer along with the main answer so that's why we have assigned it to false
    }

    response=requests.post(url,headers=headers,data=json.dumps(data))

    if response.status_code==200:
        response=response.text
        data=json.loads(response)
        actual_response=data['response']
        return actual_response
    else:
        print("error:",response.text)

interface=gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=4,placeholder="Enter your Prompt"),
    outputs="text"
)
interface.launch()
