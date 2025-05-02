from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llm.llama2_llm import llm

template = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""

pt = PromptTemplate(input_variables=["context"], template=template)
prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)
