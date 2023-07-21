
import pandas as pd
import requests
import pandasai
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

def GPT(prompt, df, column, maxTokens, temperature, model):
    finalPrompt = prompt
    for col in df.columns:
        value = str(df.loc[i, col])
        if value: # only include column name and value if value is not empty
            finalPrompt += f" {col} {value}" # add column name and value to prompt

    apiKey = "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"
    url = "https://api.openai.com/v1/completions"
    data = {
        "prompt": finalPrompt,
        "max_tokens": maxTokens,
        "temperature": temperature,
        "model": "text-davinci-003",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + apiKey,
    }
    response = requests.post(url, json=data, headers=headers)
    result = response.json()["choices"][0]["text"]
    return result

# Read the Loan.csv file
df = pd.read_csv("Loan2.csv")

maxTokens = 100
temperature = 0.5
model = "text-davinci-003"

# Use pandasai to create a new column named "status"
llm = OpenAI(api_token="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")
pandas_ai = PandasAI(llm, verbose=True, conversational=True)
response = pandas_ai.run(df, prompt="create a new df column named status, checks if loan status is paid or not paid and in DF status write as Ok or Not OK")

# Check the status column and generate appropriate text

if "status" in df.columns:
    for i in range(len(df)):
        prompt_without_row = "write a personalised text referring to loan status column:"
        for col in df.columns:
            value = str(df.loc[i, col])
            if value: # only include column name and value if value is not empty
                prompt_without_row += f" {col} {value}" # add column name and value to prompt
        generated_text = GPT("write a personalised text for loan status", df.iloc[[i]], "status", maxTokens, temperature, model)
        if generated_text:
            df.loc[i, "emailing"] = generated_text
            print(generated_text)
else:
    text = "Status column not found in DataFrame."
    
    #export df to csv
df.to_csv("Loan5.csv", index=False)