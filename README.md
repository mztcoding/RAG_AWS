# Langchain and AWS:

So you have learnt Langchain completely and now you want to learn how to work with Langchain applications and cloud using AWS. I have got you covered.

## What is AWS?
Amazon Web Services (AWS) is a subsidiary of Amazon that provides on-demand cloud computing platforms and APIs to individuals, companies, and governments, on a metered, pay-as-you-go basis.

First we have to setup an AWS account, then we will continue further:

## Setting Up an AWS account: 

Go to https://aws.amazon.com and click Create a new account.

Give all the required information. You will also have to attach a payment gateway to create your account. AWS offers 12 months free tier.

### Creating an IAM user:
When you create an account, you are the root user. But you cannot always login as a root user as it is very dangerous. You can change a setting or delete something as a root user and it is gone forever. So, you have to create an **IAM** user.

In order to create an IAM user, search for “IAM” in the search bar at the top of the console and click on “IAM” to open the IAM dashboard.

* In the left-hand navigation pane, click on “Users.”
* Click the “Add users” button at the top of the screen.
* Set User Details. Enter a unique username for the new IAM user and choose the type of access this user should have.
* Attach existing policies directly: Choose this option to assign predefined policies to the user. For example, you might select the "AdministratorAccess" policy to give full permissions or a specific service policy for limited access.
* Carefully review the user details, permissions, and policies you’ve assigned.
* If everything looks good, click on the “Create user” button.
* Download Credentials: After the user is created, AWS will show you the user’s credentials (Access Key ID and Secret Access Key). Download the .csv file containing these credentials, or copy them securely. This will be your only opportunity to download or view these credentials.

### Setting up Budget

Set up a budget for your AWS account. This will automatically notify you when you try to exceed your pre-defined budget.

## What is AWS Bedrock?

Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI, and Amazon through a single API, along with a broad set of capabilities you need to build generative AI applications with security, privacy, and responsible AI. 

Using Amazon Bedrock, you can easily experiment with and evaluate top FMs for your use case, privately customize them with your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG), and build agents that execute tasks using your enterprise systems and data sources. Since Amazon Bedrock is serverless, you don't have to manage any infrastructure.
## AWS_RAG

Now we are going to create a RAG application using Langchain and AWS Bedrock.

First you have to configure aws cli intok your machine.

### Configure aws cli
Install awscli 
```python
pip install awscli
```
Now configure.

Type the following into terminal:
```bash
$ aws configure
AWS Access Key ID [None]: YOUR_ACCESS_KEY_ID
AWS Secret Access Key [None]: YOUR_SECRET_ACCESS_KEY
Default region name [None]: us-east-1
Default output format [None]: json
```
Enter the credentials you got when creating an IAM user. 

### Installing required packages
Now let's start making the app.
First add these libraries into a requirements.txt

* boto3
* awscli
* streamlit
* langchain
* langchain_community
* faiss-cpu
* pypdf
* numpy

Then run:
```bash
pip install requirements.txt
```

### Start writing code
Now create a file named app.py and import required packages on top.
```python
import boto3
import streamlit as st

## We will be using Titan Embeddings Model To generate Embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store
from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
```
### Setting up Bedrock client and Embeddings
```python
## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)
```
### Load data and split it into docs:

```python
## Data ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs
```
### Create a vector store:
```python
def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

```
### Get the llm from Bedrock
```python

def get_llama3_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="meta.llama3-8b-instruct-v1:0",client=bedrock,
                model_kwargs={'max_gen_len':512})
    
    return llm
```
### Creating a prompt template
```python
prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

```
### Getting the response
```python

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']
```
Using streamlit to create the UI

```python

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_llama3_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
```

## Congratulations 🎉🎉
You have successfully create a RAG application using AWS Bedrock.