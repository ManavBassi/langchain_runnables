from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation"
    )

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = 'write a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'write a instagram post about {topic}',
    input_variables=['topic']
)


parallel_chain = RunnableParallel({
    'tweet': prompt1 | model | parser,
    'instagram_post': prompt2 | model | parser
})

result = parallel_chain.invoke({'topic':'cricket'})

print(result['tweet'])
print(result['instagram_post'])

