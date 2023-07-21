# Load environment
import os
import pandas as pd
from langchain import PromptTemplate
from pandasai import PandasAI
from pandasai.llm.falcon import Falcon
from langchain.chains import LLMChain
from langchain import HuggingFaceHub
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Load LLM model
llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"max_length":64, "max_new_tokens":500, "temperature":0.6})

# Load data with pd.read_csv and use pandasai to create a new column named "status"
df = pd.read_csv("Loan2.csv")
llmt = Falcon(HUGGINGFACEHUB_API_TOKEN)
pandas_ai = PandasAI(llmt, verbose=True, conversational=True)
response = pandas_ai.run(df, prompt="create a new df column named status, checks if loan status is paid or not paid and in DF status write as Ok or Not OK")


# Define prompt template
template = """You are a helpful assistant that that writes thanks  or regret personalised emails. Only use the factual information to answer the question. If you feel like you don't have enough information to answer the question, say "I don't know". Your answers should be verbose and detailed.

context:

status: Status tells us that if the loan is paid or not paid. If status is ok, then the loan is paid. If status is not ok, then the loan is not paid.
Loan_ID: This is the id of the load that was given to the customer
Loan_status: This is the current status of the loan (Paid or Not Paid)
Principal: This is the amount of the loan that was given to the customer
terms: This is the number of days that the customer has to pay back the loan
due_date: This is the date that the customer has to pay back the loan
age: This is the age of the customer
education: This is the education level of the customer
name: This is the name of the customer

status: {status}
Loan_ID: {Loan_ID}
Loan_status: {Loan_status}
number_of_loans: {number_of_loans}
Principal: {Principal}
terms: {terms}
due_date: {due_date}
age: {age}
purpose_of_loan: {purpose_of_loan}
name: {name}

Email to customer considering all attributes:"""

df

prompt = PromptTemplate(input_variables=["Loan_ID", "Loan_status", "number_of_loans", "Principal", "terms", "due_date", "age", "purpose_of_loan", "name", "status"], template=template)

responses = []
# Step 3: Iterate over the DataFrame and generate responses
for index, row in df.iterrows():
    
    row_values = row.to_dict()
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(**row_values)    
    
    # Append response to list
    responses.append(response)

# Create new dataframe from responses list
emailing = pd.DataFrame(responses, columns=["emailing"])

# Concatenate the original dataframe with the new dataframe
df = pd.concat([df, emailing], axis=1)

# Export dataframe to CSV
df.to_csv("emailing.csv", index=False)