!pip -q install git+https://github.com/huggingface/transformers # need to install from github
!pip install -q datasets loralib sentencepiece
!pip -q install bitsandbytes accelerate xformers einops
!pip -q install langchain

!nvidia-smi

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_default_device('cuda')



model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5",
                                             trust_remote_code=True,
                                             torch_dtype="auto")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5",
                                          trust_remote_code=True,
                                          torch_dtype="auto")

def generate(input_text, max_new_tokens=500):
    inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.batch_decode(outputs)[0]

    print(text)
    

from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain


from transformers import AutoModel
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import json
import textwrap


llm = HuggingFacePipeline(pipeline=generate)


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain with user input
user_input = "anime umbrellas"
output = chain.run(user_input)

print(output)