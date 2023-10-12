def summary(content):
    map_prompt = """
    Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:
    """
    
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    
    combine_prompt = """
    You are a summarisation expert. Focus on maintaining a coherent flow and using proper grammar and language. Write a detailed summary of the following text:
    "{text}"
    SUMMARY:
    """
    
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    
    summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template, verbose=True
                                    )

    response = summary_chain.run({"input_documents": docs})

    return response