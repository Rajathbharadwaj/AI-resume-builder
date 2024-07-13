import csv

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import fitz
import ast
from jobspy import scrape_jobs
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
import sentence_transformers

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
    add_vertical_space(5)
    st.write('AI Resume Builder [GitHub](https://github.com/Rajathbharadwaj/AI-resume-builder)')


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


def main():
    load_dotenv()
    llm = Ollama(model="llama3")
    structured_llm = llm.with_structured_output(Joke)
    structured_llm.invoke("Tell me a joke about cats")
    st.header("Let's simplify job search, Upload your resume 💬")
    location_city = st.text_input('Enter the City', placeholder="Toronto")
    location_country = st.text_input('Enter the Country', placeholder="CANADA")

    # upload a PDF file
    pdf = None
    if location_city:
        pdf = st.file_uploader("Load pdf: ", type=['pdf'])

    # st.write(pdf)
    text = ""

    # @st.cache_data(experimental_allow_widgets=True)
    def parse_resume():

        text = ""
        if pdf is not None:
            # st.info('Reading your resume')
            doc = fitz.open(stream=pdf.read(), filetype="pdf")
            # progress = st.progress(0, text='Reading and extracting skills from your resume')
            for page in doc:
                text += page.get_text()

                # progress.progress(counter, )
            doc.close()
            print(text)

            output_parser = StrOutputParser()
            # prompt = ChatPromptTemplate.from_messages([
            #     ("system", "A text processed from a PDF file will be given to you. Your job is to extract skills from this text. \
            # These skills are strictly to be technical, don't extract other skills like soft skills or leadership or etc. Only extract technical skills. \
            # Now when you extract for skills, return to me in the decreasing order of occurrences of the skills. For example if pytorch is a skill, return to me pytorch as the 1st element if the number of times that skill has been used in the text is greater than other skills and so on.\
            #  Here's the text"),
            #     ("user", "{text}")
            # ])
            # chain = prompt | llm | output_parser
            # skills = chain.invoke({"text": text})
            #
            # prompt2 = ChatPromptTemplate.from_messages([
            #     ("system", "I will give you number of technical skills as text. Your job is to return to the top Domain of the skill/framework. For eg, if the skill is react, then it's web dev. This is what i will use to perform job search. \
            #  Return as a dictionary. Here's the skills"),
            #     ("user", "{skills}")
            # ])
            # chain2 = prompt2 | llm | output_parser
            # job_search_skills = chain2.invoke({"skills": skills})
            #
            # prompt3 = ChatPromptTemplate.from_messages([
            #     ("system", "I will give you number of technical skills as string but it is actually a dict. Your job is to return to the top 3 Domain of the skill/framework. For eg, if the skill is react, then it's web dev. This is what i will use to perform job search. \
            #  Return as a dictionary. This is an example output expected dict->'Name of skill': 'domain' Here's are the skills"),
            #     ("user", "{job_search_clean}")
            # ])
            # chain3 = prompt3 | llm | output_parser
            # cleaned_job_search = chain3.invoke({"job_search_clean": job_search_skills})
            # st.write(cleaned_job_search)
            #
            prompt4 = ChatPromptTemplate.from_messages([
                ("system", "I will give you a text extracted from a person's resume. This text contains the person's resume details. Your job is to tell me which domain is the candidate best suited for based on the skill sets such that we can perform a job match based on the resume.\
                           Make sure to use all the information on the text to decide the best job for that candidate. There maybe multiple skillsets, but the most appropriate one is were the candidate has spent a lot of time on. It could by the virtue of building project or learning\
                           a particular framework. Here is the text {text}"),
                ("user", "{text}")
            ])
            chain4 = prompt4 | llm | output_parser
            job_matcher = chain4.invoke({"text": text})
            st.write(f'{job_matcher}')
            #
            prompt5 = ChatPromptTemplate.from_messages([
                ("system",
                 "I will give you a text which is a summary of a person's resume, now your job is to just give me the top 4 job roles based on this text as a python's list. Here's the text {text}"),
                ("user", "{text}")
            ])
            chain5 = prompt5 | llm
            job_roles = chain5.invoke({"text": job_matcher})
            st.write(job_roles.content)
            print(ast.literal_eval(job_roles.content)[0])

            job_list = ast.literal_eval(job_roles.content)

            return job_list, job_roles

    if pdf:
        job_list, job_roles = parse_resume()
        option = st.selectbox(
            "Based on your resume, these jobs are a good match. Select which you're looking for now",
            options=job_list,
        )

        if not location_country and not location_city:
            st.warning('Enter the City and Country')
        else:
            with st.spinner(f"Searching for Jobs in {job_roles}"):
                jobs = scrape_jobs(
                    site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor"],
                    search_term=f"{option}",
                    location=f"{location_city.lower()}",
                    results_wanted=40,
                    hours_old=72,  # (only Linkedin/Indeed is hour specific, others round up to days old)
                    country_indeed=f'{location_country.lower()}'  # only needed for indeed / glassdoor
                )
                jobs.to_csv("jobs.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)  # to_excel
            st.success("Found Jobs")
            st.dataframe(jobs)

        # chain.invoke({"input": text})

        # chain = chat_model
        # chain.invoke({"text": text})
        # print(chain)

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000,
        #     chunk_overlap=200,
        #     length_function=len
        # )
        # chunks = text_splitter.split_text(text=text)
        #
        # # # embeddings
        # store_name = pdf.name[:-4]
        # st.write(f'{store_name}')
        # # st.write(chunks)
        #
        # if os.path.exists(f"{store_name}.pkl"):
        #     with open(f"{store_name}.pkl", "rb") as f:
        #         VectorStore = pickle.load(f)
        #     # st.write('Embeddings Loaded from the Disk')s
        # else:
        #     embeddings = OpenAIEmbeddings()
        #     VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        #     with open(f"{store_name}.pkl", "wb") as f:
        #         pickle.dump(VectorStore, f)

        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        # query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)

        # if query:
        #     docs = VectorStore.similarity_search(query=query, k=3)
        #
        #     llm = OpenAI()
        #     chain = load_qa_chain(llm=llm, chain_type="stuff")
        #     with get_openai_callback() as cb:
        #         response = chain.run(input_documents=docs, question=query)
        #         print(cb)
        #     st.write(response)


if __name__ == '__main__':
    main()
