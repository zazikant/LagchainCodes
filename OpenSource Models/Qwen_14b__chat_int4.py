!pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
!pip install auto-gptq optimum

#Long-Context Understanding (L-CU) is a technique that extends the context length of a language model, such as Qwen-7B-Chat. This is achieved through the use of NTK-aware interpolation and LogN attention scaling. To enable these techniques, set the use_dynamic_ntk and use_logn_attn flags in the config.json file to true. On the VCSUM long-text summary dataset, Qwen-7B-Chat achieved impressive Rouge-L results when using these techniques.

!git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention
%cd flash-attention
!pip install .


from transformers import AutoTokenizer, AutoModelForCausalLM

!pip install -q langchain

import os
import torch
import transformers

from dotenv import load_dotenv, find_dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM


from transformers import AutoModel

from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv(find_dotenv())

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU device index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B-Chat-Int4", trust_remote_code=True)

# model = AutoModelForCausalLM.from_pretrained(
#      "Qwen/Qwen-14B-Chat-Int4",
#      device_map="auto",
#      trust_remote_code=True, torch_dtype=torch.bfloat16,use_cache=False,
#  ).eval()

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-14B-Chat-Int4", device_map=device,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_cache=False,
).eval()


from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map=device,
    max_new_tokens=1256,
    do_sample=True,
    top_k=30,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

from langchain.chains import LLMChain, SequentialChain

from langchain.memory import ConversationBufferMemory

from langchain import HuggingFacePipeline

from langchain import PromptTemplate,  LLMChain

llm = HuggingFacePipeline(pipeline = pipe)

llm('write a just a one word answer: which is bigger in size an giraff or dinosaur? \n\n')


