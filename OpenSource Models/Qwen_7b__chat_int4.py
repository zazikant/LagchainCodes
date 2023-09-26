!pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
!pip install auto-gptq optimum

#Long-Context Understanding (L-CU) is a technique that extends the context length of a language model, such as Qwen-7B-Chat. This is achieved through the use of NTK-aware interpolation and LogN attention scaling. To enable these techniques, set the use_dynamic_ntk and use_logn_attn flags in the config.json file to true. On the VCSUM long-text summary dataset, Qwen-7B-Chat achieved impressive Rouge-L results when using these techniques.

!git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention
%cd flash-attention
!pip install .

from transformers import AutoTokenizer, AutoModelForCausalLM

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat-Int4", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
).eval()

!pip install -q langchain

from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain


from transformers import pipeline

pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer
                )


from transformers import pipeline

pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                device_map="auto",
                max_new_tokens = 256,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )


llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0.7,'max_length': 256, 'top_k' :30})


llm('write a email for commemoration on founder of company')


