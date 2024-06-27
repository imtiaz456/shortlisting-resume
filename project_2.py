
import os
import openai
import pinecone
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone, Chroma
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import SequentialChain
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.agents import initialize_agent
from langchain.agents import AgentType
# from langchain.python import PythonREPL
from langchain.agents import tool
from langchain_community.agent_toolkits.load_tools import load_tools

import streamlit as st

# Load documents
directory = r"D:\new_nlp\dataset"
loader = DirectoryLoader(directory, show_progress=True)
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()
# # Set API key
os.environ["OPENAI_API_KEY"] = "sk-proj-ycGs0pq5z2EBrDesA5w2T3BlbkFJBT2TkT0GCV3MNOtDM8O0"

# Embeddings and vector store setup
embedding = OpenAIEmbeddings()
persist_directory = 'db'
vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()

# Reload the database
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

#################
#practice
docs = retriever.invoke("which candidate is good fit for Machine learning engineer roles.")

docs = retriever.invoke("Give name of a candidate who is good fit for Data visualization and data analysts.")

#end practice
###############

# QA Chain
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

def process_llm_response(llm_response):
    result = llm_response['result']
    sources = [source.metadata['source'] for source in llm_response["source_documents"]]
    
    print(result)
    print('\n\nSources:')
    for source in sources:
        print(source)
    
    return result, sources

# job_description = "MS or PhD in computer science or a related technical field,5+ years of industry work experience. Good sense of product with a focus on shipping user-facing data-driven features, Expertise in Python and Python based ML/DL and Data Science frameworks. Excellent coding, analysis, and problem-solving skills. Proven knowledge of data structure and algorithms. Familiarity in relevant machine learning frameworks and packages such as Tensorflow, PyTorch and HuggingFace. Experience working with Product Management and decomposing feature requirements into technical work items to ship products. Experience with generative AI, knowledge of ML Ops and ML services is a plus. This includes Pinecone, LangChain, Weights and Biases etc. Familiarity with deployment technologies such as Docker, Kubernetes and Triton are a plus. Strong communication and collaboration skills"
# question = job_description + " Based on the given job description, short list resumes which are a good fit based on skills, education, and work experience mentioned in it? Also provide the candidate name which will be mentioned in the first line of the PDF without subheading."

###################################################################
# for practice

# full example
# warning = "If you don't know the answer, just say that you don't know, don't try to make up an answer"
job_description = "MS or PhD in computer science or a related technical field,5+ years of industry work experience. Good sense of product with a focus on shipping user-facing data-driven features, Expertise in Python and Python based ML/DL and Data Science frameworks. \
Excellent coding, analysis, and problem-solving skills. Proven knowledge of data structure and algorithms. \
Familiarity in relevant machine learning frameworks and packages such as Tensorflow, PyTorch and HuggingFace\
Experience working with Product Management and decomposing feature requirements into technical work items to ship products\
Experience with generative AI, knowledge of ML Ops and ML services is a plus. This includes Pinecone, LangChain, Weights and Biases etc. \
Familiarity with deployment technologies such as Docker, Kubernetes and Triton are a plus\
Strong communication and collaboration skills"
# job_description = "MS or PhD in computer science or a related technical field,5+ years of industry work experience."
# question = warning+job_description + " Based on the given job description"
question = job_description + " Based on the given job description"
query = question + "short list resumes which is good fit based on skills,education and work experience mwntioned in it? also provide the candidate name which will be mentioned in first line of pdf without subheading"
# query = "short list resumes which is good fit for Data analysis roles based on skills,education and work experience mwntioned in it?"

llm_response = qa_chain(query)
process_llm_response(llm_response)
#@@@@@@
###########################

# query = question
# llm_response = qa_chain(query)
# process_llm_response(llm_response)

# # Example of another query
# another_query = "Based on the resumes provided, give the name of the candidate who is a good fit for Data Analyst roles. Please ensure to mention the candidate's name explicitly."
# # another_query = "Give name of candidate who is a good fit for Data Analyst roles."
# llm_response = qa_chain(another_query)
# process_llm_response(llm_response)

#################################
# for practice purposes

job_description = "MS or PhD in computer science or a related technical field,5+ years of industry work experience. Good sense of product with a focus on shipping user-facing data-driven features, Expertise in Python and Python based ML/DL and Data Science frameworks. \
Excellent coding, analysis, and problem-solving skills. Proven knowledge of data structure and algorithms. \
Familiarity in relevant machine learning frameworks and packages such as Tensorflow, PyTorch and HuggingFace\
Experience working with Product Management and decomposing feature requirements into technical work items to ship products\
Experience with generative AI, knowledge of ML Ops and ML services is a plus. This includes Pinecone, LangChain, Weights and Biases etc. \
Familiarity with deployment technologies such as Docker, Kubernetes and Triton are a plus\
Strong communication and collaboration skills"
question = job_description + " Based on the given job description"
query = question + "short list resumes which is good fit based on skills,education and work experience mwntioned in it? also provide the candidate name which will be mentioned in first line of pdf without subheading"


llm_response = qa_chain(query)
process_llm_response(llm_response)


job_description = "MS or PhD in computer science or a related technical field,5+ years of industry work experience. Good sense of product with a focus on shipping user-facing data-driven features, Expertise in Python and Python based ML/DL and Data Science frameworks. \
Excellent coding, analysis, and problem-solving skills. Proven knowledge of data structure and algorithms. \
Familiarity in relevant machine learning frameworks and packages such as Tensorflow, PyTorch and HuggingFace\
Experience working with Product Management and decomposing feature requirements into technical work items to ship products\
Experience with generative AI, knowledge of ML Ops and ML services is a plus. This includes Pinecone, LangChain, Weights and Biases etc. \
Familiarity with deployment technologies such as Docker, Kubernetes and Triton are a plus\
Strong communication and collaboration skills"
question = job_description + " Based on the given job description"
query = question + "retrive the full document information of a resume which is good fit based on skills,education and work experience mwntioned in it? "

resume_doc = retriever.invoke(query)

print(resume_doc)
resume_doc = resume_doc[0].page_content
print(resume_doc)



#@@@@@@@@@@

################################################################
# Structured output
review_template = """\
For the following text, extract the following information:

Skills: what are the technical and non technical skills? \
Answer output them as a comma separated Python list.

Education: What is the highest education of the candidate and what is the GPA as mentioned in the text?\
Answer Output should be the university/college name and GPA if given in text, output them as a comma separated Python list.

Projects: Extract all project titles mentioned in a text\
and output them as a comma separated Python list.

Publications: Extract all publication titles mentioned in a text\
and output them as a comma separated Python list.

Work experience: Extract all organization names where he/she has worked along with the number of years or months worked there and also extract designation\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
Skills
Education
Projects
Publications
Work experience

text: {text}
"""

prompt_template = ChatPromptTemplate.from_template(review_template)
print(prompt_template)


# Define the conversation chain with memory
resume_doc = "resumes"
memory = ConversationBufferWindowMemory(k=1)
memory_llm_conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
messages = prompt_template.format_messages(text=resume_doc)
response = memory_llm_conversation(messages)

print(response['response'])

# Structured output parsing
skills_schema = ResponseSchema(name="Skills", description="what are the technical and non technical skills? Answer output them as a comma separated Python list.")
projects_schema = ResponseSchema(name="Projects", description="Extract all project titles mentioned in a text and output them as a comma separated Python list.")
work_experience_schema = ResponseSchema(name="Work experience", description="Extract all organization names where he/she has worked along with the number of years or months worked there and also extract designation and output them as a comma separated Python list.")
response_schemas = [skills_schema, projects_schema, work_experience_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

review_template_2 = """\
For the following text, extract the following information:

Skills: what are the technical and non technical skills? \
Answer output them as a comma separated Python list.

Projects: Extract all project titles mentioned in a text\
and output them as a comma separated Python list.

Work experience: Extract all organization names where he/she has worked along with the number of years or months worked there and also extract designation\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
Skills
Projects
Work experience

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)
messages = prompt.format_messages(text=resume_doc, format_instructions=format_instructions)
response2 = llm(messages)
output_dict = output_parser.parse(response2.content)


print(output_dict.get('Skills'))
print(output_dict.get('Projects'))
print(output_dict.get('Work experience'))


# Sequential Chains Example
first_prompt = ChatPromptTemplate.from_template("Skills: what are the technical and non technical skills? Answer output them as a comma separated Python list.\n\n{resume_doc}")
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="skills")

second_prompt = ChatPromptTemplate.from_template("Can you name what the job roles among Data Scientist, Machine learning Engineer, Software Engineer, Data Engineer, Devops Engineer, Cloud Architect are suited based on the given skill sets\n\n{skills}")
chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="job_titles")

third_prompt = ChatPromptTemplate.from_template("Explain each skill as for what kind of projects are these useful:\n\n{skills}")
chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="skills_explanation")

overall_chain = SequentialChain(chains=[chain_one, chain_two, chain_three], input_variables=["resume_doc"], output_variables=["skills", "job_titles", "skills"], verbose=True)
seqchain_output = overall_chain(resume_doc)

################################
#practice

# @tool
# def job_desription(text: str)-> str:
#  """Returns job disriptions mentioned below, use this for any \
#  questions related to knowing the job disription. \
#  The input should always be an empty string, \
#  and this function will always return a string containing job disriptions.\ """

#  return "Job discriptions:\
#  1)Machine learning Engineer:Machine Learning Engineer with expertise in designing and developing robust models and algorithms to solve complex business problems. Experienced in end-to-end machine learning pipelines, from data preprocessing to deployment. Proficient in Python, TensorFlow, and PyTorch. Skilled in data preprocessing, feature engineering, and cloud platforms (AWS, Azure, GCP). Strong communicator with a collaborative approach and a proven ability to drive projects to completion.\
#  2)Computer Vision Engineer:Computer Vision Engineer specializing in 3D scan structure extraction and model development. Collaborates with product and research teams to enhance current products and enable new ones. Experienced with massive datasets, 2D Deep Learning, and Computer Vision using PyTorch and/or TensorFlow. Balances generalist and researcher roles, ensuring ML models transition into meaningful production. Works closely with product owners to deliver value efficiently to customers.\
#  3)Data Analyst:Data analyst with expertise in designing dashboards and have experience on tableau, power bi, sas, spss."


# tools = load_tools(["llm-math","wikipedia"], llm=llm)

# agent= initialize_agent(
#     tools+ [job_desription],
#     llm,  #turbo_llm, qa_chain,
#     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     handle_parsing_errors=True,
#     verbose = True)

# agent_template = """\
# The following is the resume and query:

# resume: {resume}

# query: {query}
# """

# prompt = ChatPromptTemplate.from_template(template=agent_template)
# query_human = 'Skills: what are the technical and non technical skills? \Answer output them as a comma separated Python list.'
# messages = prompt.format_messages(resume=resume_doc, query=query_human)
# result = agent(messages)

# agent_template = """\
# The following is the resume and query:

# resume: {resume}

# query: {query}
# """


# prompt = ChatPromptTemplate.from_template(template=agent_template)
# query_human = 'Give me the available job discriptions?'
# messages = prompt.format_messages(resume=resume_doc, 
#                                 query=query_human)

# result = agent(messages) 

### working fine up to this

# #####

# job_description_template = """
# You are good at matching available job description with resume.\
# Steps:\
# 1.Retreive job discriptions from givel tool attached with agent \
# 2.Compare if resume can be selected based on any job discription, if yes then retuen that specific job discription
# 3.If no job discription matches the return None

# Here is a resume:
# {input}"""


# portfolio_finder_template = """

# You are good at finding portfolio link from the given resume and return that link to the user.If link not found return None.

# Here is a question:
# {input}"""

# summary_template = """
# You are good at summerising the given resume. You will include skills, professional experience, education in the summary.

# Here is a question:
# {input}"""

# prompt_infos = [
#     {
#         "name": "job_description",
#         "description": "Good for providing job discription that is matched",
#         "prompt_template": job_description_template
#     },
#     {
#         "name": "portfolio",
#         "description": "Good for returning portfolio link from resume",
#         "prompt_template": portfolio_finder_template
#     },
#     {
#         "name": "summary",
#         "description": "Good for providing summary of resume",
#         "prompt_template": summary_template
#     }
# ]

# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# destination_chains = {}
# for p_info in prompt_infos:
#     name = p_info["name"]
#     prompt_template = p_info["prompt_template"]
#     prompt = ChatPromptTemplate.from_template(template=prompt_template)
#     if name == "job_description":
#         chain = agent
#     elif name == "portfolio" :
#         chain = LLMChain(llm=llm, prompt=prompt)
#     else:
#         chain = LLMChain(llm=llm, prompt=prompt)

#     destination_chains[name] = chain

# destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
# destinations_str = "\n".join(destinations)


# default_prompt = ChatPromptTemplate.from_template("{input}")
# default_chain = LLMChain(llm=llm, prompt=default_prompt)

# MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
# language model select the model prompt best suited for the input. \
# You will be given the names of the available prompts and a \
# description of what the prompt is best suited for. \
# You may also revise the original input if you think that revising\
# it will ultimately lead to a better response from the language model.

# << FORMATTING >>
# Return a markdown code snippet with a JSON object formatted to look like:
# ```json
# {{{{
#     "destination": string \ name of the prompt to use or "DEFAULT"
#     "next_inputs": string \ a potentially modified version of the original input
# }}}}
# ```

# REMEMBER: "destination" MUST be one of the candidate prompt \
# names specified below OR it can be "DEFAULT" if the input is not\
# well suited for any of the candidate prompts.
# REMEMBER: "next_inputs" can just be the original input \
# if you don't think any modifications are needed.

# << CANDIDATE PROMPTS >>
# {destination}

# << INPUT >>
# {{input}}

# << OUTPUT (remember to include the ```json)>>"""

# router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format( destination=destinations_str)
# router_prompt = PromptTemplate(template=router_template, input_variables=["input"], output_parser=RouterOutputParser())

# router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# chain = MultiPromptChain(router_chain=router_chain,
#                          destination_chains=destination_chains,
#                          default_chain=default_chain, verbose=True
#                         )


# #####



#end practice
################################



import streamlit as st

st.title("Resume Shortlisting App")

job_description = st.text_area("Job Description", height=200)

if st.button("Shortlist Resumes"):
    query = job_description + " Based on the given job description, shortlist resumes which are a good fit based on skills, education, and work experience mentioned in it? Also provide the candidate name which will be mentioned in the first line of the PDF without subheading."
    llm_response = qa_chain(query)
    result, sources = process_llm_response(llm_response)

    st.write("### Result")
    st.write(result)
    
    st.write("### Sources")
    for source in sources:
        st.write(source)