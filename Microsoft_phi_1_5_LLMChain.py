#use t4 gpu in colab

#!pip -q install langchain huggingface_hub transformers sentence_transformers einops

import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'your_token'

from langchain import PromptTemplate, HuggingFaceHub, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

#-----------

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

#-----------

torch.set_default_device('cuda')

#-----------

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")

class LLMPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run(self, question):
        inputs = self.tokenizer(question, return_tensors="pt")
        input_ids = inputs["input_ids"]
        outputs = self.model.generate(input_ids=input_ids, max_length=300)
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text

llm_chain = LLMPipeline(model=model, tokenizer=tokenizer)

#-----------

question = "write a landing page on general dentistry?"

print(llm_chain.run(question))