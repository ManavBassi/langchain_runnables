from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation"
    )

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = 'write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'explain the joke in a few words{joke}',
    input_variables=['joke']
)

chain  = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

print(chain.invoke({'topic':'cricket'}))