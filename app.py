import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import fitz

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


def main():
    load_dotenv()
    llm = ChatOpenAI(model_name='gpt-4')
    st.header("Let's simplify job search, Upload your resume 💬")

    # upload a PDF file
    pdf = st.file_uploader("Load pdf: ", type=['pdf'])

    # st.write(pdf)
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
        prompt = ChatPromptTemplate.from_messages([
            ("system", "A text processed from a PDF file will be given to you. Your job is to extract skills from this text. \
        These skills are strictly to be technical, don't extract other skills like soft skills or leadership or etc. Only extract technical skills. \
        Now when you extract for skills, return to me in the decreasing order of occurrences of the skills. For example if pytorch is a skill, return to me pytorch as the 1st element if the number of times that skill has been used in the text is greater than other skills and so on.\
         Here's the text"),
            ("user", "{text}")
        ])
        chain = prompt | llm | output_parser
        skills = chain.invoke({"text": text})

        prompt2 = ChatPromptTemplate.from_messages([
            ("system", "I will give you number of technical skills as text. Your job is to return to the top Domain of the skill/framework. For eg, if the skill is react, then it's web dev. This is what i will use to perform job search. \
         Return as a dictionary. Here's the skills"),
            ("user", "{skills}")
        ])
        chain2 = prompt2 | llm | output_parser
        job_search_skills = chain2.invoke({"skills": skills})

        prompt3 = ChatPromptTemplate.from_messages([
            ("system", "I will give you number of technical skills as string but it is actually a dict. Your job is to return to the top 3 Domain of the skill/framework. For eg, if the skill is react, then it's web dev. This is what i will use to perform job search. \
         Return as a dictionary. This is an example output expected dict->'Name of skill': 'domain' Here's are the skills"),
            ("user", "{job_search_clean}")
        ])
        chain3 = prompt3 | llm | output_parser
        cleaned_job_search = chain3.invoke({"job_search_clean": job_search_skills})
        st.write(cleaned_job_search)



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
