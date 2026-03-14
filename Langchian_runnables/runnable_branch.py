from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation"
    )

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = 'write a report about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text'])

chain1 = prompt1 | model | parser

branch = RunnableBranch(
    (lambda x : len(x.split()) > 100, prompt2 | model | parser),
    RunnablePassthrough()
)
final_chain = chain1 | branch

print(final_chain.invoke({'topic':'cricket'}))