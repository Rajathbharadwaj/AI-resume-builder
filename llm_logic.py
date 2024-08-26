from typing import Literal
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from jobspy import scrape_jobs
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from pprint import pprint
from dotenv import load_dotenv, dotenv_values
from langgraph.checkpoint.memory import MemorySaver
import os
import re


load_dotenv(override=True, dotenv_path=".env")
latex_template = r"""

\documentclass{resume} % Use the custom resume.cls style

\usepackage[left=0.4 in,top=0.4in,right=0.4 in,bottom=0.4in]{geometry} % Document margins
\newcommand{\tab}[1]{\hspace{.2667\textwidth}\rlap{#1}} 
\newcommand{\itab}[1]{\hspace{0em}\rlap{#1}}
\name{Firstname Lastname} % Your name
% You can merge both of these into a single line, if you do not have a website.
\address{+1(123) 456-7890 \\ San Francisco, CA} 
\address{\href{mailto:contact@faangpath.com}{contact@faangpath.com} \\ \href{https://linkedin.com/company/faangpath}{linkedin.com/company/faangpath} \\ \href{www.faangpath.com}{www.faangpath.com}}  %

\begin{document}

%----------------------------------------------------------------------------------------
%	OBJECTIVE
%----------------------------------------------------------------------------------------

\begin{rSection}{OBJECTIVE}

{Software Engineer with 2+ years of experience in XXX, seeking full-time XXX roles.}


\end{rSection}
%----------------------------------------------------------------------------------------
%	EDUCATION SECTION
%----------------------------------------------------------------------------------------

\begin{rSection}{Education}

{\bf Master of Computer Science}, Stanford University \hfill {Expected 2020}\\
Relevant Coursework: A, B, C, and D.

{\bf Bachelor of Computer Science}, Stanford University \hfill {2014 - 2017}
%Minor in Linguistics \smallskip \\
%Member of Eta Kappa Nu \\
%Member of Upsilon Pi Epsilon \\


\end{rSection}

%----------------------------------------------------------------------------------------
% TECHINICAL STRENGTHS	
%----------------------------------------------------------------------------------------
\begin{rSection}{SKILLS}

\begin{tabular}{ @{} >{\bfseries}l @{\hspace{6ex}} l }
Technical Skills & A, B, C, D
\\
Soft Skills & A, B, C, D\\
XYZ & A, B, C, D\\
\end{tabular}\\
\end{rSection}

\begin{rSection}{EXPERIENCE}

\textbf{Role Name} \hfill Jan 2017 - Jan 2019\\
Company Name \hfill \textit{San Francisco, CA}
 \begin{itemize}
    \itemsep -3pt {} 
     \item Achieved X\% growth for XYZ using A, B, and C skills.
     \item Led XYZ which led to X\% of improvement in ABC
    \item Developed XYZ that did A, B, and C using X, Y, and Z. 
 \end{itemize}
 
\textbf{Role Name} \hfill Jan 2017 - Jan 2019\\
Company Name \hfill \textit{San Francisco, CA}
 \begin{itemize}
    \itemsep -3pt {} 
     \item Achieved X\% growth for XYZ using A, B, and C skills.
     \item Led XYZ which led to X\% of improvement in ABC
    \item Developed XYZ that did A, B, and C using X, Y, and Z. 
 \end{itemize}

\end{rSection} 

%----------------------------------------------------------------------------------------
%	WORK EXPERIENCE SECTION
%----------------------------------------------------------------------------------------

\begin{rSection}{PROJECTS}
\vspace{-1.25em}
\item \textbf{Hiring Search Tool.} {Built a tool to search for Hiring Managers and Recruiters by using ReactJS, NodeJS, Firebase and boolean queries. Over 25000 people have used it so far, with 5000+ queries being saved and shared, and search results even better than LinkedIn! \href{https://hiring-search.careerflow.ai/}{(Try it here)}}
\item \textbf{Short Project Title.} {Build a project that does something and had quantified success using A, B, and C. This project's description spans two lines and also won an award.}
\item \textbf{Short Project Title.} {Build a project that does something and had quantified success using A, B, and C. This project's description spans two lines and also won an award.}
\end{rSection} 

%----------------------------------------------------------------------------------------
\begin{rSection}{Extra-Curricular Activities} 
\begin{itemize}
    \item 	Actively write \href{https://www.faangpath.com/blog/}{blog posts} and social media posts (\href{https://www.tiktok.com/@faangpath}{TikTok}, \href{https://www.instagram.com/faangpath/?hl=en}{Instagram}) viewed by over 20K+ job seekers per week to help people with best practices to land their dream jobs. 
    \item	Sample bullet point.
\end{itemize}


\end{rSection}

%----------------------------------------------------------------------------------------
\begin{rSection}{Leadership} 
\begin{itemize}
    \item Admin for the \href{https://discord.com/invite/WWbjEaZ}{FAANGPath Discord community} with over 6000+ job seekers and industry mentors. Actively involved in facilitating online events, career conversations, and more alongside other admins and a team of volunteer moderators! 
\end{itemize}


\end{rSection}


\end{document}

"""



resume_yaml_schema = """
	•	Personal Information:
	•	name
	•	surname
	•	date_of_birth
	•	country
	•	city
	•	address
	•	phone_prefix
	•	phone
	•	email
	•	github
	•	linkedin
	•	Education Details:
	•	degree
	•	university
	•	gpa
	•	graduation_year
	•	field_of_study
	•	exam
	•	Experience Details:
	•	position
	•	company
	•	employment_period
	•	location
	•	industry
	•	key_responsibilities
	•	skills_acquired
	•	Projects:
	•	name
	•	description
	•	link
	•	Achievements:
	•	name
	•	description
	•	Certifications:
	•	certification_name
	•	Languages:
	•	language
	•	proficiency
	•	Interests:
	•	interest

"""
class CreateResume(BaseModel):
    """
    Given a resume text, undestand the candidate's skills, experience, education, projects, certifications, languages, interests, and achievements.
    And generate an optimized resume given new job description.
    """
    optimized_resume: str = Field(
        ...,
        description="Updated Resume optimzed with ATS and job description aligned"
    )


llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_create_resume = llm.with_structured_output(CreateResume)

system = """
You are an expert resume writer with a deep understanding of Applicant Tracking Systems (ATS) and the key elements required to craft a resume that aligns perfectly with a given job description. 
Your task is to generate an optimized version of a candidate’s resume. 
Given the job description provided, extract and incorporate relevant keywords, skills, and experiences that the employer is looking for.
Ensure that the resume is tailored to highlight the candidate’s strengths in alignment with the job requirements and is structured in a way that maximizes the chances of passing through ATS filters successfully.
Also, I'll provide an example of the resume schema which you can use to extract details from the candidate's resume.
"""

create_resume_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Resume Schema:\n\n {resume_yaml_schema} \n\n Candidate's Resume:\n\n {resume_text} \n\n Job Description:\n\n {job_description}"),
])

create_resume_chain = create_resume_prompt | structured_llm_create_resume

class ResumeLatex(BaseModel):
    """
    Given a resume text, generate a resume in latex format.
    """
    resume_latex: str = Field(
        ...,
        description="Resume in latex format without new line characters"
    )

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_resume_latex = llm.with_structured_output(ResumeLatex)

system = """
You are an expert resume writer with a deep understanding of Applicant Tracking Systems (ATS) and the key elements required to craft a resume in latex format. 
Your task is to generate a resume in latex format given the resume text provided.
When generating LaTeX code within a Python script, it’s important to avoid the unintended addition of newline characters (\n). This can be achieved by using triple quotes (\""") to define the LaTeX code as a block string. 
Block strings in Python preserve the original formatting of the LaTeX code, including line breaks and spacing, without introducing additional newlines. 
When processing or manipulating LaTeX code, ensure that triple quotes are used to encapsulate the entire LaTeX content, preventing any unintended formatting issues.
I will provide you with an latex template which you can follow to generate the resume in latex format.
"""

resume_latex_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Resume Text:\n\n {resume_text} \n\n Latex Template:\n\n {latex_template}"),
])  

resume_latex_chain = resume_latex_prompt | structured_llm_resume_latex

class CreateCoverLetter(BaseModel):
    """
    Given a resume text, generate a cover letter in latex format.
    """
    cover_letter: str = Field(
        ...,
        description="Cover letter in latex format"
    )   

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_create_cover_letter = llm.with_structured_output(CreateCoverLetter)

system = """
You are an expert cover letter writer with a deep understanding of Applicant Tracking Systems (ATS) and the key elements required to craft a cover letter in latex format. 
Your task is to generate a cover letter in latex format given the resume text provided.

"""

cover_letter_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Resume Text:\n\n {resume_text} \n\n Job Description:\n\n {job_description}"),
])

cover_letter_chain = cover_letter_prompt | structured_llm_create_cover_letter

class FixLatexNewlines(BaseModel):
    """
    Given a latex code, fix the newlines and all the other formatting issues in the latex code.
    """
    fixed_latex_code: str = Field(
        ...,
        description="Latex code without new line characters"
    )

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_fix_latex_newlines = llm.with_structured_output(FixLatexNewlines)

system = """
You are an expert latex code fixer with a deep understanding of latex code and the key elements required to fix the newlines and all the other formatting issues in the latex code. 
Your task is to fix the newlines and all the other formatting issues in the latex code given the latex code provided. Also remove the comments in the latex code that might affect the rendering of the latex code.
"""

fix_latex_newlines_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Latex Code:\n\n {latex_code}"),
])

fix_latex_newlines_chain = fix_latex_newlines_prompt | structured_llm_fix_latex_newlines

class JobTitle(BaseModel):
    """
    Given a candidate's resume, return the top 3 domains of the job title.
    """
    job_domains: List[str] = Field(
        ...,
        description="Top 4 domains of the job title"
    )

    experience: int = Field(
        ...,
        description="Experience of the candidate in years"
    )

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_job_title = llm.with_structured_output(JobTitle) 

system = """
You are an expert job title extractor with a deep understanding of job titles and the key elements required to extract the job title from the resume. 
Your task is to extract the job title from the resume given the resume text provided. Given the following resume text, provide the top 4 job titles this person is suitable for.
Please ensure the job titles are standard and can be used for job searches on job portals.
The job titles should be concise and relevant to the person's skills and experience.
"""

job_title_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Resume Text:\n\n {resume_text}"),
])

job_title_chain = job_title_prompt | structured_llm_job_title



def create_resume_latex(resume_text: str, job_description: str):
    optimized_resume = create_resume_chain.invoke({"resume_text": resume_text, "job_description": job_description, "resume_yaml_schema": resume_yaml_schema})
    optimized_latex = resume_latex_chain.invoke({"resume_text": optimized_resume.optimized_resume, "latex_template": latex_template})
    fixed_latex = fix_latex_newlines_chain.invoke({"latex_code": optimized_latex.resume_latex})
    def fix_latex_newlines(latex_code):
    # Regex to remove newlines that are not preceded by \\ and not followed by another newline
        fixed_latex_code = re.sub(r'(?<!\\\\)\n(?!\n)', ' ', latex_code)
        return fixed_latex_code
    optimized_latex = fix_latex_newlines(fixed_latex.fixed_latex_code)
    return optimized_resume.optimized_resume, optimized_latex

def create_cover_letter(resume_text: str, job_description: str):
    cover_letter = cover_letter_chain.invoke({"resume_text": resume_text, "job_description": job_description})  
    return cover_letter.cover_letter

def get_top_4_jobs(resume_text: str,):
    job_title_domains = job_title_chain.invoke({"resume_text": resume_text})
    jobs_titles = job_title_domains.job_domains
    experience = job_title_domains.experience
    return jobs_titles, experience

def get_jobs_from_api(job_titles: str, city: str, country: str, total_jobs: int):
    jobs = scrape_jobs(
                        site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor"],
                        search_term=job_titles,
                        location=f"{city.lower()}",
                        results_wanted=total_jobs,
                        hours_old=72,  # (only Linkedin/Indeed is hour specific, others round up to days old)
                        country_indeed=f'{country.lower()}',
                        is_remote=True, # only needed for indeed / glassdoor
                    )
    return jobs







