import os
import streamlit as st
from deta import Deta
from dotenv import load_dotenv
from uuid import uuid4
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
import hashlib

load_dotenv()

# Your Deta project key and base name
DETA_PROJECT_KEY = os.getenv("DETA_PROJECT_KEY")  # Replace with your Deta project key
DETA_BASE_NAME = "articles"  # Set the name for your Deta Base

deta = Deta(st.secrets["data_key"])
db = deta.Base(DETA_BASE_NAME)

# Your API KEY must be saved securely!!
api_key: str | None = os.getenv(key="OPENAI_API_KEY")

# A prompt with defined templates
title_template = PromptTemplate(
    input_variables=["topic"],
    template="Give me a medium article title on {topic}",
)

article_template = PromptTemplate(
    input_variables=["article_title"],
    template="Write a medium article for the title {article_title}.",
)

# OpenAI llm instances
gpt4_turbo_instruct_llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)
gpt4_turbo_llm = ChatOpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)

# Chains for generating the title and article
title_chain = LLMChain(
    llm=gpt4_turbo_instruct_llm, 
    prompt=title_template,
    output_key="article_title", 
    verbose=True
)
article_chain = LLMChain(
    llm=gpt4_turbo_llm, 
    prompt=article_template, 
    verbose=True
)

st.set_page_config(page_title='Medium Article Generator', page_icon='✒️', initial_sidebar_state="auto", menu_items=None)

st.title("Medium Article Generator ✒️")

# History Sidebar
st.sidebar.title("History")

# Fetch all history items from Deta Base
history_items = db.fetch().items
history_items.sort(key=lambda x: x["created"], reverse=True)  # Sort by creation time

# Display clickable entries
for entry in history_items:
    if st.sidebar.button(entry["title"], key=f"history_{entry['key']}"):
        st.markdown(f"## {entry['title']}")
        st.write(entry['article'])

# Text input
topic: str = st.text_input("Enter the article's topic:", placeholder="<Insert Article Topic Here>")

if st.button('Generate Article'):
    with st.spinner('Generating your article...'):
        title_response = title_chain.run(topic)

        if title_response:
            generated_title = title_response

            # Run the article generation chain with the generated title
            article_response = article_chain.run(generated_title)

            if isinstance(article_response, str):
                generated_content = article_response

                # Display Generated Article with Title
                st.markdown(f"## {generated_title}")
                st.write(generated_content)

                # Add to Deta Base history
                entry_key = hashlib.sha256(generated_content.encode('utf-8')).hexdigest()
                db.put({
                    "key": entry_key,
                    "title": generated_title,
                    "article": generated_content,
                    "created": int(uuid4().time)  # Timestamp based on UUID generation time
                })
            else:
                st.error("Failed to generate the content of the article.")
        else:
            st.error("Failed to generate a title for the article.")
else:
    st.write("Type something into the text input above and press the 'Generate Article' button to start the article generation process.")
