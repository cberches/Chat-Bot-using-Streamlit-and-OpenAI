import openai
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader
import docx
import PyPDF2
import markdown
import tempfile
import os
from PIL import Image
import numpy as np
import easyocr

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chatbot","JD Generator", "Job Matching", "AI Screening", "CV Parsing"])

openai.api_key = st.secrets['OPENAI_API_KEY']

with tab1:
    st.title("NXT Chatbot")
    isV2 = st.toggle("Use v2 with index")
    if isV2: 
        st.header("Chatbot v2")
        st.subheader("Using Llama Indexing for document file-based knowledge base")

        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
        ]

        @st.cache_resource(show_spinner=False)
        def load_data():
            with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
                reader = SimpleDirectoryReader(input_dir="./kb", recursive=True)
                docs = reader.load_data()
                service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", 
                                                                        temperature=0.5, 
                                                                        system_prompt="You are a chat assistant for NXT. Keep your answers technical and based on facts – do not hallucinate features. . Don’t give information not mentioned in the CONTEXT INFORMATION."))
                index = VectorStoreIndex.from_documents(docs, service_context=service_context)
                return index

        index = load_data()

        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        if st.session_state.messages["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_engine.chat(prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response }
                    st.session_state.messages.append(message) # Add response to message history
    else: 
        st.header("Chatbot v1")
        st.subheader("Using system prompt as knowledge base")
        try:
            if "openai_model" not in st.session_state:
                st.session_state["openai_model"] = "gpt-3.5-turbo"

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("What is up?"):
                system_prompt = str("Context: question	Answer"
                +"How do I post a job listing on this platform?	To post a job and connect with potential candidates:\n\n1. Set up your company profile if you haven't set it up.\n2. Navigate to â€˜Manage Jobâ€™.\n3. Click on â€˜Add Jobâ€™.\n4. Provide essential details:\n   - Name, Email, and Contact Number.\n   - Job Information: Title, Type, Arrangement, Hours, Address, City, Salary, Overview, Qualification, and Benefits.\n   - Job Requirements: Skill Set and Dialect.\n   - Posting Date: Start and End Date.\n5. Save as a Draft.\n6. When ready, click â€˜Publishâ€™ and confirm."
                +"How can I edit my client profile?	To edit your company profile:\n\n1. Log in with your email.\n2. Click 'Profile' > 'Company Profile.'\n3. Complete company details.\n4. Click 'Update Setting.'\n\nIf you need further assistance, feel free to ask!"
                + "Is there an option to schedule interviews or calls within the platform?	No, currently, our platform does not have the option to schedule interviews. If you have any other questions or need assistance, feel free to ask!"
                + "Where can I get help if I encounter technical issues?	If you encounter technical issues, feel free to ask me! I'm here to help. For specific questions that require assistance from a real person, you can reach out to our support team at NXT Contact Us (https://www.nxt.com.ph/contact-us)."
                + "How do I report a problem or seek assistance from platform support?	If you encounter technical issues, feel free to ask me! I'm here to help. For specific questions that require assistance from a real person, you can reach out to our support team at NXT Contact Us (https://www.nxt.com.ph/contact-us)."
                + "How do I manage and edit my posted jobs?	Easy! Go to 'Manage Job' in your profile, click 'Edit' on the job post. Need more assistance or have specific changes in mind?"
                + "How can I view my subscription status?	Simply log in, go to 'Profile,' then click 'Subscription.' It'll show your subscription type and validity. Need further clarification?"
                + "What types of job postings can I create on NXT?	You can post jobs for skilled professionals, fresh graduates, or experienced candidates. The platform caters to diverse hiring needs. What type of position are you looking to fill?"
                + "Can I search for specific skills or qualifications in candidates?	Absolutely! Utilize our search filters to find candidates based on skills, experience, and more. Need help with a specific search?"
                + "How can I track the status of applications for my posted jobs?	Go to 'Manage Job,' and you'll see a summary of your posted jobs, including application status. Need assistance with anything specific?"
                + "What do I do if I encounter technical issues on the platform?	Feel free to ask me! For technical issues, you can also check our support page at https://www.nxt.com.ph/contact-us. How can I assist you further?"
                + "How do I withdraw a job posting if it's no longer available?	In 'Manage Job,' find the post, and click 'Delete.' Confirm, and it's withdrawn. Need help with this or anything else?"
                + "Can I customize the application status for candidates in the hiring process?	Certainly! You can set custom statuses for applications. Let me know if you need guidance on customizing these statuses."
                + "Are there any tips for creating an attractive job posting?	Absolutely! Ensure a detailed job overview, attractive benefits, and clear qualifications. Need more tips or assistance in creating a post?"
                + "How do I provide feedback to NXT about my experience or suggestions?	We'd love to hear from you! Feel free to share your thoughts or suggestions by visiting https://www.nxt.com.ph/contact-us. Anything specific you'd like to discuss?"
                + "Can I edit a job posting after it's been published?	Yes, you can edit your job postings. Go to 'Manage Job,' find the post, click the edit icon, and make your changes. Need assistance with editing?"
                + "How can I view the resumes of applicants for a specific job posting?	In 'Manage Job,' click 'View Applicants' for the desired job. You'll see a list of applicants. Need help navigating or viewing resumes?"
                + "What types of benefits information should I include in a job posting?	Common benefits include health insurance, retirement plans, and flexible schedules. Would you like guidance on crafting an effective benefits section?"
                + "Is there a way to share job postings on social media directly from the NXT platform?	Currently, we recommend copying the job link and sharing it on your social media. Anything specific you'd like help with?"
                + "Can I save drafts of job postings and come back to finish them later?	Absolutely! While creating a job post, click 'Save as A Draft.' Need guidance on managing drafts or anything else?"
                + "As an employer, can I post job opportunities without any charges on NXT?	Yes, good news! Employers can post job opportunities on NXT for free. We aim to provide a seamless experience for both job seekers and employers without any associated charges."
                + "How can I reset my password on NXT if I forget it?	No worries! If you forget your password, simply click on the 'Forgot Password' link on the login page. You'll receive an email with instructions to reset your password."
                + "What is 'Manage Jobs' in NXT?	'Manage Jobs' is a module in NXT where a Client can oversee and manage job postings, including creating, viewing, updating, and activating job posts in the Applicant Tracking System (ATS)."
                + "Can you explain the key features of 'Manage Jobs' that benefit Clients?	The 'Manage Jobs' module allows Clients to efficiently handle job postings, providing features like job creation, viewing, updating, and activation in ATS. Clients can manage their hiring processes effectively."
                + "Who can access the 'Manage Jobs' module?	All individuals registered as Clients in NXT will have access to the 'Manage Jobs' module, visible on the left panel of the NXT Client landing page after login."
                + "Is there any specific requirement for Clients to access the 'Manage Jobs' module?	No, any individual registered as a Client in NXT can access the 'Manage Jobs' module after logging in. There are no additional requirements."
                + "What is ATS in NXT?	ATS, or Applicant Tracking System, is a feature in NXT that allows Clients to manage and track the progress of applicants on the job posts they have enrolled or activated."
                + "How does ATS streamline the recruitment process for Clients?	ATS streamlines the recruitment process by providing Clients with tools to manage and track applicant progress efficiently, enhancing the overall hiring experience."
                + "How does one post a job in NXT?	To post a job, a Client needs to go to 'Manage Jobs,' click the \Create Job\" button"
                + "What are the advantages of completing all required fields when posting a job?	Completing all required fields ensures a clear and comprehensive job posting, saving time and effort for both clients and talents by providing detailed information about the job opening."
                + "Can the details or status of a posted job be updated?	Yes, Clients can update the details or status of posted jobs. For a status update from Draft to Publish or correction, they can click the pencil icon in the Manage Jobs table and make the necessary updates. For Open to Close status update, similar steps apply."
                + "Are there any restrictions on the frequency of job updates in NXT?	No, Clients can update job details or status as needed. There are no restrictions on the frequency of updates, allowing flexibility in managing job postings."
                + "How can an expired job post be reopened?	To reopen an expired job post, the Client needs to click the pencil icon in Manage Jobs, update the Posting Start and End Dates, and click the Publish button for the changes to take effect."
                + "Are there any limitations on the number of times a job post can be reopened in NXT?	There are no specific limitations on the number of times a job post can be reopened. Clients can reopen job posts as needed by updating the Posting Start and End Dates."
                + "What types of job posts can be activated in ATS?	'Open' job posts and 'Expired' job posts can be activated in ATS."
                + "Is there a specific reason to activate 'Expired' job posts in ATS?	Yes, activating 'Expired' job posts in ATS allows Clients to manage applicants for positions that might become available again in the future."
                + "How can a job post be added to ATS in NXT?	The Client, with an active ATS subscription, can tick the checkbox for the desired job post in Manage Jobs and click the 'Add to ATS' button. Alternatively, they can activate a job in ATS via the job details form by toggling the 'ATS Status' switch."
                + "How does having an active ATS subscription impact the job activation process?	An active ATS subscription is necessary for adding jobs to ATS. It determines the number of job posts the Client can activate in the ATS."
                + "How can a job post be removed from ATS in NXT?	The Client can remove a job post from ATS by ticking the checkbox for the desired job in Manage Jobs and clicking the 'Remove in ATS' button. Alternatively, they can deactivate a job in ATS via the job details form by toggling the 'ATS Status' switch."
                + "Can a job be removed from ATS if there are ongoing applicants?	No, jobs with ongoing applicants cannot be removed from ATS to ensure a seamless application process for those candidates."
                + "What is the primary purpose of the ATS in NXT?	The primary purpose of the ATS is to allow Clients to manage the applicant selection process for job posts."
                + "How does the ATS contribute to a more organized and efficient hiring process?	The ATS streamlines the applicant selection process, providing tools for organized management of applicants, ensuring efficiency and ease in the hiring process."
                + "How can a Client confirm the closure of a job in NXT?	To confirm the closure of a job, the Client needs to click the Pencil icon in the Manage Jobs table, open the job details form, and click the Close button. A confirmation prompt will appear."
                + "Are there any implications for closing a job in NXT?	Closing a job makes it no longer searchable to talents, but Clients can still view the details in read-only mode for reference purposes."
                + "What information is required to post a job in NXT?	To post a job, a Client needs to provide information in the Hiring Manager Information, Job Information, Job Requirements, and Posting Dates sections, ensuring completion of all required fields."
                + "How can the completion of all required fields enhance the quality of job postings?	Completing all required fields ensures a comprehensive job posting, providing a clear picture of the job opening and enhancing the quality of information available to potential applicants."
                + "What happens when a job is closed in NXT?	Once a job is closed, it will no longer be searchable to talents, but Clients can still view the details in read-only mode for reference purposes."
                + "Can a closed job be reopened without any changes to the details?	Yes, a closed job can be reopened without changes by updating the Posting Start and End Dates and clicking the Publish button."
                + "What does the 'Save as Draft' option do in NXT?	The 'Save as Draft' option in NXT saves and shelves the job opening for future use without making it searchable to job seekers or Talents."
                + "In what situations might a Client choose to save a job as a draft?	Clients may choose to save a job as a draft if they want to work on the details later or if the job opening is not yet ready to be searchable to job seekers."
                + "Can 'Expired' job posts be activated in ATS in NXT?	Yes, 'Expired' job posts can be activated in ATS, as they may possibly be set to 'Open' eventually."
                + "Is there any difference in the activation process for 'Expired' and 'Open' job posts in ATS?	The activation process for 'Expired' and 'Open' job posts is the same, allowing flexibility for Clients to manage applicants for positions that may become available again."
                + "What is the significance of completing job details in NXT?	Completing job details in NXT, including requirements, roles, responsibilities, and salary, saves time and effort for both clients and talents."
                + "How can detailed job information benefit both Clients and talents in the hiring process?	Detailed job information helps Clients attract suitable talents and helps talents make informed decisions, contributing to a more successful and efficient hiring process."
                + "How does a Client access the 'Manage Jobs' module in NXT?	After logging in, a Client can access the 'Manage Jobs' module"
                + "Is there any alternative method for Clients to access the 'Manage Jobs' module?	No, accessing the 'Manage Jobs' module is exclusive to the left panel of the NXT Client landing page after login. There are no alternative access methods."
                + "How does a Client view the ATS Candidate List in NXT?	A Client can click the Job Title of an activated job in ATS to open the ATS Candidate List page in NXT."
                + "What information is available on the ATS Candidate List page, and how can it assist Clients in the hiring process?	The ATS Candidate List page provides a comprehensive view of applicants for a specific job, helping Clients make informed decisions and manage the applicant selection process more effectively."
                + "What is the purpose of the Hired List page in the ATS web application?	The Hired List page is a crucial component of the ATS web application designed to assist Hiring Managers in managing and understanding applicants who have been successfully hired for specific job posts."
                + "What does the Hired List feature inside the ATS focus on?	The Hired List feature inside the ATS is dedicated to each specific job post, serving as a curated list of applicants who have been successfully hired for a particular job opening."
                + "How are applicants added to the Hired List?	Applicants become visible in the Hired List feature once they have successfully completed all stages of the sourcing and job offer process. These stages include New Applicant, Screening, Exams, Initial Interview, Final Interview, Offer Letter, Background Investigation, Pre-Employment, Employment Contract, Pre-Employment Requirements, Hired, and submission by the recruiter as hired."
                + "What information does the Hired List provide about applicants?	The Hired List provides valuable insights and information about applicants who have accepted job offers and are ready to be onboarded. This includes details such as their name, contact information, application status, and any additional notes provided by the recruiter."
                + 'How can one navigate to the Hired List for a specific job post?	To access the Hired List for a specific job post, follow these steps: Log in to the ATS web application using your client credentials, navigate to the \Manage Jobs" section'
                + "What stages are involved in the sourcing and job offer process before applicants appear on the Hired List?	Applicants go through stages such as New Applicant, Screening, Exams, Initial Interview, Final Interview, Offer Letter, Background Investigation, Pre-Employment, Employment Contract, and Pre-Employment Requirements before reaching the Hired stage and being submitted by the recruiter as hired."
                + "What actions can Hiring Managers perform from the Hired List page?	From the Hired List page, Hiring Managers can perform various actions related to hired applicants, including viewing detailed applicant profiles, initiating the onboarding process, and communicating with hired applicants regarding next steps and onboarding procedures."
                + "How does the Hired List contribute to the recruitment process in the ATS?	The Hired List contributes to the recruitment process by providing a centralized space for Hiring Managers to manage and understand applicants who have successfully been hired. It streamlines the onboarding process and facilitates communication with hired applicants."
                + "What does the Hired List showcase for each specific job post?	For each specific job post, the Hired List showcases a curated list of applicants who have successfully completed the hiring process, accepted job offers, and are ready for onboarding."
                + "What stages does an applicant go through before being added to the Hired List?	Before being added to the Hired List, applicants go through stages such as New Applicant, Screening, Exams, Initial Interview, Final Interview, Offer Letter, Background Investigation, Pre-Employment, Employment Contract, and Pre-Employment Requirements, culminating in the Hired stage and submission by the recruiter."
                + "What details are typically included in each entry on the Hired List?	Each entry on the Hired List may include details such as the applicant's name, contact information, application status, and any additional notes provided by the recruiter."
                + "How can Hiring Managers initiate the onboarding process from the Hired List?	Hiring Managers can initiate the onboarding process for hired applicants directly from the Hired List page, streamlining the transition from successful hiring to onboarding procedures."
                + "In the ATS, when are applicants considered officially hired and added to the Hired List?	Applicants are considered officially hired and added to the Hired List in the ATS once they have successfully completed all stages of the sourcing and job offer process and have been submitted by the recruiter as hired."
                + "What is the purpose of the Failed/Withdraw List page in the ATS web application?	The Failed/Withdraw List page is a critical component of the ATS web application, providing Hiring Managers with powerful tools to view the list of applicants who were unsuccessful, failed, or withdrew their application for specific job posts."
                + "What does the Failed/Withdraw List feature within the ATS focus on?	The Failed/Withdraw List feature within the ATS is dedicated to each specific job post, serving as a curated list of applicants who were unsuccessful, failed, or voluntarily withdrew their application for a particular job opening."
                + "How are applicants added to the Failed/Withdraw List?	Applicants become visible in the Failed/Withdraw List feature under circumstances where they have undergone the stages of the sourcing and job offer process, did not meet the qualifications, failed in some aspect of the application process, or were tagged as 'failed' or 'withdrawn' by the recruiter based on their assessment."
                + "What information does the Failed/Withdraw List provide about applicants?	The Failed/Withdraw List allows Hiring Managers to gain insights into why applicants were not successful in their application process. It includes details such as the applicant's name, contact information, application status, and any additional notes provided by the recruiter."
                + "How can one navigate to the Failed/Withdraw List for a specific job post?	To access the Failed/Withdraw List for a specific job post, follow these steps: Log in to the ATS web application using your credentials, navigate to the 'Manage Jobs' section, select the desired job post, and click on the 'Failed/Withdraw' tab or button to view the list of applicants who were unsuccessful or withdrew their application for that job post."
                + "Under what circumstances do applicants become visible in the Failed/Withdraw List?	Applicants become visible in the Failed/Withdraw List when they have undergone the stages of the sourcing and job offer process, did not meet the qualifications or failed in some aspect, or were tagged as 'failed' or 'withdrawn' by the recruiter based on their assessment."
                + "What details can be found in each entry on the Failed/Withdraw List?	Each entry on the Failed/Withdraw List may include details such as the applicant's name, contact information, application status, and any additional notes provided by the recruiter."
                + "What actions can Hiring Managers perform from the Failed/Withdraw List page?	From the Failed/Withdraw List page, Hiring Managers can perform various actions related to failed or withdrawn applicants, including reviewing reasons for application failure or withdrawal, analyzing trends and patterns in application outcomes, and communicating with applicants to provide feedback or support."
                + "How does the Failed/Withdraw List contribute to the recruitment process in the ATS?	The Failed/Withdraw List contributes to the recruitment process by offering tools for Hiring Managers to analyze and understand why applicants were unsuccessful or withdrew their applications. It provides valuable insights for continuous improvement in the hiring process."
                + "What is the primary focus of the Failed/Withdraw List for each specific job post?	For each specific job post, the Failed/Withdraw List focuses on curating a list of applicants who were unsuccessful, failed, or voluntarily withdrew their application. It aids Hiring Managers in gaining insights into the outcomes of the application process."
                + "In what situations might a recruiter tag an applicant as 'failed' or 'withdrawn'?	A recruiter might tag an applicant as 'failed' or 'withdrawn' based on their assessment of the applicant's status, which could be due to the applicant not meeting the qualifications or experiencing failure in some aspect of the application process."
                + "How can Hiring Managers review reasons for application failure or withdrawal from the Failed/Withdraw List?	Hiring Managers can review reasons for application failure or withdrawal directly from the Failed/Withdraw List page, providing them with insights into the specific circumstances that led to an applicant's unsuccessful outcome."
                + "What benefits does the Failed/Withdraw List provide for Hiring Managers?	The Failed/Withdraw List provides Hiring Managers with powerful tools to analyze trends and patterns in application outcomes, review reasons for failure or withdrawal, and communicate with applicants for feedback or support. It supports continuous improvement in the hiring process."
                + "How does the ATS determine whether an applicant should be added to the Failed/Withdraw List?	The ATS adds an applicant to the Failed/Withdraw List based on circumstances where the applicant has undergone the stages of the sourcing and job offer process, did not meet the qualifications or failed in some aspect, or was tagged as 'failed' or 'withdrawn' by the recruiter."
                + "What is the purpose of the Talent List Page in the ATS web application?	The Talent List Page is a fundamental feature of the ATS web application, designed to provide Hiring Managers with efficient tools to manage and organize the talent pool effectively."
                + "What sort and filter options are available on the Talent List Page?	The Talent List Page offers Ascending and Descending Order sorting options by Name and a Date Applied filter. These options provide flexibility in organizing the talent pool based on applicant names and application timelines."
                + "How can the table list on the Talent List Page be sorted?	The table list on the Talent List Page can be sorted in both ascending and descending orders based on the Name column, allowing Hiring Managers to organize the talent pool efficiently."
                + "What does the Date Applied filter on the Talent List Page allow you to do?	The Date Applied filter on the Talent List Page enables users to filter the list based on the application date of the applicant. Selecting a specific date on the filter allows narrowing down the list based on application timelines."
                + "What functionality does the Keyword Search in the Search Field provide on the Talent List Page?	The Keyword Search in the Search Field allows users to filter the list on the table by providing keywords. These keywords should match the Employee name and Job Posting, enabling quick and effective search functionality."
                + "What information is displayed in the Applicant List Table on the Talent List Page?	The Applicant List Table displays all applicants regardless of the job applied. It includes Applicant Name, Job Posting, and Date Applied. This information provides a quick reference to individual candidates and their application timelines."
                + "How does the Applicant List Table help in understanding the specific roles candidates are interested in?	The Job Posting column in the Applicant List Table shows the specific roles each applicant applied for, helping Hiring Managers understand the roles candidates are interested in."
                + "What actions can be performed using the Eye Icon in the Action column of the Talent List Page?	Clicking on the Eye Icon in the Action column allows users to view the CV of a specific applicant. This provides quick access to applicant documents for assessment."
                + "How can Hiring Managers access the detailed profile of an applicant from the Talent List Page?	Hiring Managers can access the detailed profile of an applicant by clicking on the applicant's name in the table on the Talent List Page. This comprehensive profile includes relevant details about the applicant, facilitating a deeper understanding of their qualifications and experience."
                + "What details are included in the comprehensive profile of an applicant on the Talent List Page?	The comprehensive profile of an applicant on the Talent List Page includes relevant details about the applicant, providing information about their qualifications and experience."
                + "Why is the Date Applied information important on the Talent List Page?	The Date Applied information on the Talent List Page is crucial as it helps Hiring Managers understand the timeline for applicant engagement. Knowing when each applicant submitted their application provides insights into the recruitment process's timing and efficiency."
                + "What is the purpose of the Hired List page in the Applicant Tracking System (ATS)?	The Hired List page is a crucial feature of the ATS designed to assist Hiring Managers in managing and understanding applicants who have been successfully hired for specific job posts."
                + "What sort and filter options are available on the Hired List page?	The Hired List page offers Ascending and Descending Order sorting options by Name and a Date Applied filter. These options provide flexibility in organizing the list of successfully hired applicants based on names and application timelines."
                + "How can the table list on the Hired List page be sorted?	The table list on the Hired List page can be sorted in both ascending and descending orders based on the Name column, allowing Hiring Managers to organize the list of successfully hired applicants efficiently."
                + "What does the Date Applied filter on the Hired List page allow you to do?	The Date Applied filter on the Hired List page enables users to filter the list based on the application date of the hired applicant. Selecting a specific date on the filter allows narrowing down the list based on application timelines for successful hires."
                + "What functionality does the Keyword Search in the Search Field provide on the Hired List page?	The Keyword Search in the Search Field allows users to filter the Hired List table by providing keywords. These keywords should match the Employee name and Job Posting, enabling quick and effective search functionality."
                + "What information is displayed in the Applicant List Table on the Hired List page?	The Applicant List Table on the Hired List page displays all applicants who have been successfully hired in their applied jobs across all job postings created by the company. It includes Applicant Name, Job Posting, and Date of Application."
                + "How does the Applicant List Table help in understanding the specific roles successfully hired applicants are now part of?	The Job Posting column in the Applicant List Table on the Hired List page shows the specific roles each successfully hired applicant applied for, helping Hiring Managers understand the roles they are now part of within the company."
                + "Why is the Date of Application information important on the Hired List page?	The Date of Application information on the Hired List page is crucial as it provides insights into the timeline of successful hires. Reviewing when each hired applicant submitted their application helps understand the efficiency and timing of the recruitment process."
                + "What actions can be performed using the Eye Icon in the Action column of the Hired List page?	Clicking on the Eye Icon in the Action column allows users to view the CV of a specific hired applicant on the Hired List page. This provides quick access to applicant documents for further reference."
                + "How can Hiring Managers access the detailed profile of a hired applicant from the Hired List page?	Hiring Managers can access the detailed profile of a hired applicant by clicking on the applicant's name in the table on the Hired List page. This comprehensive profile includes relevant details about the successfully hired individual, aiding in understanding their qualifications and experience."
                + "What details are included in the comprehensive profile of a hired applicant on the Hired List page?	The comprehensive profile of a hired applicant on the Hired List page includes relevant details about the individual, providing information about their qualifications and experience as a successfully hired employee."
                + "What benefits does the Hired List page offer for Hiring Managers in managing successful hires?	The Hired List page offers benefits for Hiring Managers by providing efficient tools to manage and understand applicants who have been successfully hired. It aids in organizing successful hires, reviewing application timelines, and accessing detailed profiles for further reference."
                + "How does the ATS determine whether an applicant should be added to the Hired List?	The ATS adds an applicant to the Hired List once they have been successfully hired for a specific job post. The list includes applicants from all job postings created by the company who have completed the hiring process and accepted job offers."
                + "Who is allowed to see the ATS dashboard?	Access to the ATS dashboard is exclusively granted to our esteemed clients and dedicated recruiters. It's the central hub where they orchestrate the recruitment symphony. If you're part of this distinguished group, welcome to the dashboard, where efficiency meets excellence."
                + "Can super admin view different company dashboards?	The super admin has the superpower to effortlessly view and analyze multiple company dashboards. It's like having a bird's-eye view of the entire ATS landscape."
                + "Can users view the total number of jobs?	Curious about opportunities? Users can easily spot the total number of available jobs right on the dashboard. It's your go-to spot for job hunting within the system."
                + "Can users view total jobs posted?	Keepin' track! The dashboard proudly showcases the grand total of jobs posted. It's a real-time snapshot of the job market within your ATS domain."
                + "Can users view the total number of active job posts?	Active and kicking! Users get the lowdown on the total number of jobs currently in play. It's your dashboard's way of highlighting what's hot in the job market."
                + "Can users view the total number of expired/closed job posts?	Yes, Closure details, anyone? The dashboard spills the beans on the total number of expired or closed job posts. It's like tidying up the job board to make room for new opportunities."
                + "Can users view the total number of draft jobs?	In the drafting room! The dashboard keeps you in the loop with the total number of draft jobs. It's where ideas take shape before hitting the spotlight."
                + "Can users view the total number of talent pool lists?	Talent central! Users can easily check out the total number of talents in the talent pool right from the dashboard. It's like having your talent reservoir on display."
                + "Can users view the total number of Hired talents?	Success stories! The dashboard celebrates victories by showcasing the total number of talents that have successfully landed jobs. It's your personal hall of hired fame."
                + "What metrics and analytics are displayed on the dashboard?	The metrics and analytics displayed on the dashboard are To Be Determined (TBH), as the development team is actively assessing the most relevant and valuable data points to present."
                + "Is the dashboard customizable for individual users?	The dashboard's customization for individual users is currently TBD. The development team is exploring options to provide users with the flexibility to tailor their dashboard based on personal preferences and job responsibilities."
                + "How frequently is the dashboard data updated?	The frequency of dashboard data updates is currently TBD. The development team is evaluating the optimal update intervals to balance real-time insights with system performance."
                + "Can users set up notifications for specific events or metrics?	The ability for users to set up notifications for specific events or metrics on the dashboard is currently To Be Determined (TBD). We are working on providing more information about this soon."
                + "Is there an export feature for dashboard data?	The export feature for dashboard data is To Be Determined (TBD). The development team is considering the implementation of this functionality to allow users to export relevant data for further analysis or reporting purposes in future updates."
                + "What do I do if I encounter technical issues on the platform?	Feel free to ask me! For technical issues, you can also check our support page at https://www.nxt.com.ph/contact-us. How can I assist you further?"
                + "How can I update my contact information on NXT?	To update your contact information, log in to your NXT account, go to 'Profile,' and click 'Update Profile.' There, you can modify your contact details. Need further assistance?"
                + "Is there a limit to the number of job applications I can submit?	No, there's no limit to the number of job applications. Feel free to apply to as many opportunities as you want. Need guidance on effective job applications?"
                + "Can I delete my NXT account?	Certainly. If you wish to delete your account, please contact NXT Support through our website or submit a request via the provided form. How can I assist you further?"
                + "What should I do if I encounter technical issues while using the platform?	If you face technical issues, try clearing your browser cache or disabling pop-up blockers. For persistent problems, reach out to NXT Support through our website. Need more help?"
                + "How can I customize my job search preferences?	To customize your job search, go to 'Joblist,' use keywords, role titles, or company names in the search field, and click 'Search Job.' You can also set filters for a more tailored search. Need assistance with that?"
                + "What are the benefits of using Single Sign-On (SSO) with my Google or Facebook account?	SSO simplifies access. By linking your Google or Facebook account, you can log in seamlessly without remembering multiple passwords. Interested in setting up SSO or have other questions?"
                + "How do I stand out in my job applications on NXT?	To stand out, complete your profile, including skills and work history. Tailor your applications to match job requirements. Want tips on creating an impressive profile?"
                + "What is MeNXT Career Tips, and how can it help me in my job search?	MeNXT Career Tips provides insights into the latest trends and tips. Explore it to enhance your job search strategy. Anything specific you're looking for in career tips?"
                + "How secure is my personal information on NXT, especially during the SSO process?	Security is a top priority. During SSO, only essential information is accessed. We employ high-security measures to protect your data. Any specific concerns or questions about security?"
                + "Can I connect my LinkedIn profile to my NXT account?	As of now, NXT primarily supports SSO with Google or Facebook. However, you can manually add details from your LinkedIn profile to enhance your NXT profile. Need help with that?"
                + "Is there a feature to track the progress of my job applications?	Certainly! Go to 'Applied Jobs' in your profile to track the progress of your applications. You'll receive real-time updates on your application status. Anything specific you'd like to know?"
                + "Can I update my skills and work history after creating my NXT profile?	Absolutely! You can update your skills and work history at any time. Go to 'My Resume' in your profile to make necessary edits. Need guidance on updating specific details?"
                + "How can I provide feedback or suggestions about the NXT platform?	We appreciate your feedback! You can provide suggestions or comments by submitting the form available on our website. Your input is valuable in enhancing the NXT experience. Any particular feedback you'd like to share?"
                + "How can I reset my password on NXT if I forget it?	No worries! If you forget your password, simply click on the 'Forgot Password' link on the login page. You'll receive an email with instructions to reset your password."
                + "Who is NXT, and what services do you provide?	NXT is a leading platform connecting jobseekers and businesses. We offer end-to-end workforce solutions. How can I assist you further?"
                + "What are the benefits of registering as a client on nxt.com.ph?	Registering as a client offers various benefits, including seamless talent connection and highest data security. Ready to sign up, or do you have specific questions?"
                + "Can I use my Google or Facebook account to sign in?	Absolutely! Click on 'Get Started,' choose Google or Facebook, and follow the prompts. Need more guidance on this?"
                + "How can I reset my password if I forget it?	No worries! Click 'Forgot Password' during login, and follow the steps sent to your email. Need assistance with anything else?"
                + "What details do I need to complete my company profile?	To complete your company profile, provide details like company name, size, industry, etc. Need a step-by-step guide, or do you have specific questions?"
                + "Is there a fee for posting jobs on NXT?	Good news! Our clients enjoy a FREE subscription. You can publish unlimited job postings. Any other questions or ready to post a job?"
                + "What types of job postings can I create on NXT?	You can post jobs for skilled professionals, fresh graduates, or experienced candidates. The platform caters to diverse hiring needs."
                + "Can I search for specific skills or qualifications in candidates?	Absolutely! Utilize our search filters to find candidates based on skills, experience, and more."
                + "Is there a limit to the number of job postings I can have?	With your FREEmium account, you can publish unlimited job postings for free. Ready to post a new job, or do you have specific requirements?"
                + "Is there a mobile app for accessing NXT services on the go?	Currently, we recommend accessing NXT through your browser for the best experience. Any other questions or mobile-related concerns?"
                + "What industries does NXT cater to in terms of job placements?	NXT caters to a wide range of industries. You can find opportunities in IT, healthcare, finance, and more. Need assistance in finding jobs in a specific industry?"
                + "What sets NXT apart from other job platforms?	NXT stands out for its user-friendly interface, SSO capabilities, and real-time application updates. The platform is designed for a seamless and rewarding job search experience. Anything specific you're interested in?"
                + "Can you tell me more about Acquiro, the company behind NXT?	Acquiro is the workforce solutions company behind NXT. We specialize in connecting top talents and businesses through our advanced tech platform. Any specific details you're interested in?"
                + "What types of job opportunities are available on NXT, and in which industries?	NXT offers a wide range of job opportunities across various industries and career levels. Whether you're seeking opportunities in IT, healthcare, finance, or others, NXT has you covered. Any specific industry you're interested in?"
                + "Tell me more about the talent enablement resources for job seekers on NXT.	NXT provides resources, tools, and guidance for skill enhancement and career success. You can explore MeNXT Career Tips for the latest trends. Need assistance in accessing these resources?"
                + "Can job seekers customize their job search based on work arrangements, like remote or part-time?	Absolutely! Job seekers on NXT have the flexibility to customize their job search based on preferences such as full-time, part-time, remote, and flexible work arrangements."
                + "What measures does NXT have in place to prioritize user security and privacy?	NXT prioritizes user security and privacy with robust data protection measures. Your information is treated with the utmost care. Any specific concerns or questions about security?"
                + "Are there any fees associated with using NXT for employers or job seekers?	Good news! NXT offers a seamless experience for both employers and job seekers with no associated fees. You can enjoy the benefits without any cost. Anything else you'd like to clarify?"
                + "What types of personal information does Acquiro Solutions and Tech Inc. collect from users?	Acquiro collects various types of personal information, including full name, contact information, date of birth, nationality, gender, social media profiles, resume/CV details, job preferences, cover letter, references, additional documents, user account information, and more."
                + "How does Acquiro use and protect my personal information?	Acquiro uses personal information for purposes like contacting users, providing information about products/services, email marketing (with consent), record maintenance, legal compliance, customer support, and other transaction-related services. The company has implemented physical, technical, and organizational security measures to protect the data."
                + "What rights do I have as a data subject under the Data Privacy Act?	As a data subject, you have rights such as being informed, accessing your personal information, correcting inaccuracies, objecting to processing, and more. You can contact Acquiro's Data Protection Officer at privacy@acquiro.com.ph for further information."
                + "How long is my personal information retained by Acquiro?	Personal information is retained as long as necessary to fulfill its purposes, comply with legal obligations, resolve disputes, and enforce agreements. Clear retention periods are established for different types of data."
                + "Who has access to my personal information?	Personal information is shared with affiliates directly involved in fulfilling services, third parties providing data reporting services, and may be disclosed under applicable laws or for legal compliance."
                + "How often does Acquiro update its Privacy Policy, and how am I notified of changes?	Acquiro reserves the right to modify its Privacy Policy at any time. Users are notified of changes by posting updated contents on its websites. Continued use after changes implies consent."
                + "Can I review and update the information I provided?	Yes, users can typically review and update their information by logging into their accounts on the websites."
                + "What security measures are in place to protect my personal information during transmission?	Acquiro uses Security Sockets Layer (SSL) protocol to encrypt personal information as it travels over the internet. Various security techniques, including secure servers and firewalls, are employed."
                + "What are the terms and conditions for using the NXT website?	The terms and conditions cover aspects like user accounts, job listings, the application process, privacy, intellectual property, dispute resolution, limitation of liability, in-line ads, data collection, and the free use of NXT services for employers/hirers."
                + "How does Acquiro handle data from users who are just visitors to the company websites?	Acquiro collects information from website visitors, including IP address, device information, browsing behavior, referral source, cookies, and tracking technologies. This data is processed in line with the Privacy Policy."
                + "Can I opt out of targeted ads and data collection on the NXT website?	Yes, users can usually manage their ad preferences and opt-out of certain data collection practices. Check the Privacy Policy for details on how to exercise these preferences."
                + "How can I contact Acquiro's Data Protection Officer with privacy-related concerns?	You can reach out to Acquiro's Data Protection Officer at privacy@acquiro.com.ph or send comments or issues in writing to the provided address at 15F ASTICOM, W5th Building, 5th Ave. Bonifacio Global City, Taguig, Philippines."
                + "What happens if I disagree with certain aspects of the Terms and Conditions?	If you disagree with any aspect of the Terms and Conditions, it is advised to refrain from using or utilizing the NXT Website. Continued access and use of the Website signify your acceptance of those terms."
                + "Can you explain more about the limitations of liability mentioned in the Terms and Conditions?	While Acquiro strives to provide accurate information, it does not guarantee the accuracy, completeness, or reliability of content. Acquiro shall not be held responsible for any loss, damage, or inconvenience arising from the use of the Website or the information presented therein."
                + "Are there any age restrictions for creating an account on the NXT website?	Users must be at least 18 years old to create an account on the NXT website, as mentioned in the Terms and Conditions."
                + "What steps should I take if I want to report a job listing that seems to violate guidelines or applicable laws?	Users can typically report job listings that violate guidelines or laws through the platform. The platform may have a reporting feature or contact information for reporting such issues."
                + "How are disputes related to the use of the NXT website resolved?	Disputes related to the use of the NXT website are governed by Philippine laws, and both parties agree to submit to the exclusive jurisdiction of Philippine courts for resolution."
                + "Can I get more information about the free use of NXT services for employers/hirers?	Employers/hirers who have created accounts through the NXT website can freely utilize products and services, including posting jobs and utilizing talent search, without incurring service fees, as long as they comply with the terms and conditions outlined in the Terms of Service."
                + "As an employer, what information does Acquiro collect when I use the NXT website for hiring purposes?	Acquiro collects information such as company profile details, job postings, employer account details, links to social media and the company website, employer branding content, contact information for hiring teams, analytics and usage data, communication data, reviews and ratings, and integration with the Applicant Tracking System (ATS) if applicable."
                + "Can I browse job opportunities on NXT without creating an account?	While some features are available without an account, creating one offers a more personalized experience. You can explore a wider range of opportunities, save preferences, and receive real-time updates on your applications."
                + "As an employer, can I post job opportunities without any charges on NXT?	Yes, good news! Employers can post job opportunities on NXT for free. We aim to provide a seamless experience for both job seekers and employers without any associated charges."
                + "What's the Applicant Tracking System (ATS) of NXT?	It is a NXT feature that enables businesses and organizations to manage their recruitment process more effectively. It serves as a centralized hub for handling job applications, resumes, candidate data, and hiring-related communications."
                + "Are there subscriptions inside the platform?	Indeed, the ATS is here to supercharge your recruitment game with our subscription plans. It's like having a secret weapon to streamline your process and find the best talent out there."
                + "How can you subscribe to ATS?	To dive into the world of ATS excellence, the first step is for our valued clients to subscribe to NXT. Once you're in, you can easily add the ATS subscription as an extra feature. It's all about making your recruitment journey seamlessly powerful."
                + "What are the pros of having an ATS subscription?	1. Centralized Applicant Management\n2. Job Post Progress Tracking\n3. Efficient Workflow\n4. Customizable Status Updates\n5. User-Friendly Interface"
                + "Are there different packages for Subscriptions?	Absolutely! The ATS takes customization seriously. We offer a range of subscription packages, each packed with distinct features and limits. It's about tailoring your experience to fit your unique recruitment needs. Explore the options and find the perfect match for your hiring journey."
                + "What are the perks of a Basic Subscription?	5 Job Placement\n30 Validity (from date of subscription)\nCompany Branding (eMail)\nVirtual Account Manager\nJob Posting Insights and Benchmark Comparison\nAll job postings are on Top Searches"
                + "What are the perks of a Premium Subscription?	10 Job Placement\n45 Validity (from date of subscription)\nCompany Branding\nCustomer Support (eMail, Call) Analytics and Insights \nTop Searches (5X a month per Job for 3 hours of preferred time)"
                + "What are the perks of a Pro Subscription?	20 Job Placement\n60 Validity (from date of subscription)\nCompany Branding\nVirtual Account Manager\nJob Posting Insights and Benchmark Comparison\nAll job postings are on Top Searches"
                + "Can you customize your subscriptions?	Yes, clients have the flexibility to customize subscription plans based on their specific needs. We believe in giving clients the power to tailor their subscription plans. It's all about making it work just right for you and your hiring goals."
                + "Can you upgrade subscription?	Yes, Our clients can level up their game by upgrading their subscription plans. It's like unlocking a new set of tools and features for an even more robust experience."
                + "Can you downgrade Subscription?	Yup, it's in the cards. If you ever feel the need to adjust, downgrading is on the table. Just keep in mind, the previous subscription needs to take a bow before the downgrade dance begins."
                + "Can you cancel Subscription?	Yes, Need a breather? No problem! Clients have the flexibility to cancel their subscription when it suits them. It's all about making choices that align with your current needs."
                + "Can you Renew Subscription?	Yes,Good news! If you're loving the ATS perks, renewing your subscription is a breeze. Keep the momentum going and continue reaping the benefits."
                + "does ATS allow subscription trial?	Yes, Dive in without any commitments! We offer a 30-day free trial period for our clients to explore the platform, test the waters, and see firsthand how ATS can elevate their recruitment game."
                + "What payment methods are accepted for ATS subscriptions?	As of the moment, Mode of payments are currently To Be Determined (TBD). We are working on providing more information about this soon."
                + "Is there an auto-renewal feature for subscriptions?	For now, no worries about auto-renewals. We've got you covered. You have full control over when and how you want to continue with your subscription."
                + "Can users switch between subscription packages at any time?	Absolutely! We get that needs change. That's why users have the freedom to switch between subscription packages. It's like having a wardrobe of options to pick from based on your evolving requirements."
                + "Are there any discounts or promotions available for subscriptions?	Yes, periodic discounts or promotions may be available but for now its currently in a TBD (To be Determined) status. We are working on providing more information about this soon."
                + "Is there a trial period for each subscription package?	Sure thing! Each subscription package comes with a sweet 30-day free trial period. It's like a test drive for you to explore and experience all the awesome features before committing."
                + "Can u register with an existing Client?	No, registration with an existing client is not supported. Each client needs a unique registration. But don't worry, it's easy-peasy to get your unique registration done."
                + "Can u choose to register without an existing Client?	Yup, absolutely! If you're new to the scene, you can totally sign up without needing to buddy up with an existing client. It's your ticket to joining the NXT crew hassle-free."
                + "What information is required for client registration?	Your Email Address: So we can stay in touch and keep you updated.\nPassword: Choose a secure one to keep your account safe.\nAccepting Privacy Policy and Terms and Conditions: Just a heads-up, we take privacy seriously. Take a moment to check those off.\nHow did you Learn about NXT: We're curious! Let us know how you found out about us. Your feedback helps us improve."
                + "Are there different levels of access for clients within the ATS?	As of now, the details regarding different levels of access for clients within the ATS (Applicant Tracking System) of NXT are yet to be determined (TBD). The system is currently in the development or planning stage, and the specific roles and permissions for clients are under consideration. The access levels for clients will be carefully designed to align with the diverse needs and responsibilities of users within the recruitment process."
                + "Can a user be associated with multiple clients?	Just to keep things streamlined, each user is linked with a single client. It's like having your designated spot in the NXT family."
                + "How is client information managed and updated in the system?	Easy peasy! Clients hold the reins. They can effortlessly manage and update their information by heading to the company profile within the system. It's your company's space, after all!"
                + "If the user's question is not directly related to the given contxt, politely reject it.")
            
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    for response in openai.ChatCompletion.create(
                        model=st.session_state["openai_model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": "prompt" + ". Don’t give information not mentioned in the CONTEXT INFORMATION."}
                        ],
                        stream=True,
                    ):
                        full_response += response.choices[0].delta.get("content", "")
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        except: 
            st.subheader('Refresh Page')

with tab2:
    prompt = st.text_input("Describe the position/job role")
    system_prompt = """
    Context:
    The job listing is being generated for a company looking to attract qualified candidates for a vacant position. The company aims to provide a comprehensive and appealing overview of the job to entice potential applicants.

    Objective:
    Craft a detailed yet concise job description that highlights the key aspects of the position, including responsibilities, qualifications, benefits, skillset, and dialect. The goal is to attract suitable candidates and encourage them to apply.

    Style:
    The writing style should be professional, informative, and engaging. It should strike a balance between formality and approachability, making the job listing easy to read while maintaining a sense of importance and relevance.

    Tone:
    Maintain a positive and inclusive tone throughout the job listing. Emphasize the value the company places on its employees and the exciting opportunities the position offers. The tone should be encouraging and inviting, prompting potential candidates to envision themselves thriving in the role.

    Audience:
    The primary audience is prospective job seekers, including individuals actively seeking employment or considering a career change. Tailor the language to resonate with a diverse audience, encompassing various demographics and professional backgrounds.

    Response Style:
    The skillset should be the list of skill only (ex Python, tensorlfow, etc.), without explanation and the dialect should be the language (ex english) to be used in communication without explanation. 
    **Job Title:** 
    **Job Description:**
    **Job Overview:**
    **Qualifications:**
    **Benefits:**
    **Skillset:**
    **Dialect:**"""

    if st.button('Generate'):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

with tab3:
    
    system_prompt = """# CONTEXT # 
    I am a machine learning engineer at a startup company in the staffing solutions industry. In this field, I create machine learning models which will help the company to achieve their goals. Now, I am focused on talent recruitment by helping them to develop AI models specifically for the Talent Recruitment platform as well as the application tracking system.

    # OBJECTIVE #
    Your task is to help me in scoring an applicant if they are match to the job they applied for. This involves getting the AI match scores with the average of three important factors work, skills, and education. The aim is to provide a score and breakdown with its analysis for helping talent recruiters in pre-screening.

    # STYLE #
    Write in a format which will be displayed in the pre-screening process. Ensure that the score is in percentage format, the AI score as the average of the Work, education, and skills, and the insight should be short but concise.

    # Tone #
    Maintain a formal tone in the insights. Make sure to retain the tone as an HR Expert.

    # AUDIENCE #
    The target audience is individuals who are the clients which are the HR recruiter or hiring managers looking for talents to improve their team.

    # RESPONSE FORMAT #
    Provide the AI Score and its breakdown on how much work, skills, and education impacted AI score in percentages and the insight or your analysis in short but concise manner. The AI score should be the average of the three breakdown criteria.

    example:
    AI Score: 96%
    Drivers of matching:
    - Work: 96%
    - Skills: 96%
    - Education: 96%

    Provide also in listed format

    """
    jd = st.empty()
    app_info = st.empty()

    jd_in = jd.text_area("Input Job Description")
    app_info_in = app_info.text_area('Input Applicant info')

    jd_gen_btn = st.radio("Use Job Matching Sample Info", ['Yes', 'No'])

    job_info = f"""
    Position: AI/ML Engineer
    Location: Taguig
    Job Description:
    "Develop and implement machine learning algorithms and models to address business challenges, such as predictive analytics, natural language processing, computer vision, recommendation systems, or anomaly detection.
    Collaborate with data scientists and domain experts to understand business requirements and translate them into technical specifications.
    Preprocess and analyze large datasets to extract relevant features and insights, ensuring data quality and integrity throughout the process.
    Design, train, and evaluate machine learning models using state-of-the-art techniques and frameworks.
    Optimize and fine-tune models for performance, accuracy, and scalability.
    Deploy models into production environments and monitor their performance and effectiveness over time.
    Collaborate with software engineers to integrate AI/ML solutions into existing systems or develop new applications.
    Stay up-to-date with the latest advancements in AI/ML research and technologies, and actively contribute to the organization's knowledge base.
    Conduct experiments, run A/B tests, and provide data-driven insights to support decision-making processes.
    Document and communicate methodologies, results, and findings effectively to both technical and non-technical stakeholders."

    Job Requirements:
    "Bachelor's degree in computer science, data science, or a related field
    Solid understanding of machine learning algorithms, statistical models, and data structures.
    Proficiency in programming languages such as Python, R, or Java, and experience with ML libraries and frameworks (e.g., TensorFlow, PyTorch, scikit-learn).
    Knowledgeable in MongoDb and DynamoDb.
    Hands-on experience with data preprocessing, feature engineering, and exploratory data analysis.
    Demonstrated experience in developing and deploying machine learning models in production environments.
    Strong analytical and problem-solving skills, with the ability to think critically and creatively to tackle complex challenges.
    Excellent communication and collaboration skills, with the ability to explain technical concepts to both technical and non-technical stakeholders.
    Experience with cloud platforms (e.g., AWS, Azure, Google Cloud) and knowledge of distributed computing frameworks (e.g., Apache Spark) is a plus.
    Familiarity with software development best practices, version control systems, and agile methodologies.
    A strong passion for staying up-to-date with the latest AI/ML trends, research papers, and technologies.
    Should have a good understanding and foundational knowledge of NodeJS, ReactJs, and/or AngularJS"

    Open to work in BGC on a hybrid arrangement
    Job Type: Full-time
    Salary: Php40,000.00 - Php60,000.00 per month
    Benefits: Health insurance
    Schedule: 8-hour shift
    Supplemental pay types: 13th month salary
    """

    info_string = "- Gender: Female\n- Age: 21\n- Location: San Mateo, Rizal\n- Work History: Data Science Intern for the past 3 months\n- Skills: data science, machine learning, pandas, python, streamlit, numpy, seaborn, matplotlib, scikit-learn, shap, scipy.\n- Education: Bachelor of Science in Computer Science"

    gen_ai_score = st.button('Analyze Applicant Score')

    if gen_ai_score:
        prompt = ""
        if jd_gen_btn=='Yes':
            jd.text_area("Input Job Description", value=job_info)
            app_info.text_area('Input Applicant info', value=info_string)

            jd_in = job_info
            app_info_in = info_string
            prompt= str(jd_in) + " " + str(app_info_in)

        else:
            prompt= str(jd_in) + " " + str(app_info_in)

        ai_score_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            ai_score_placeholder.markdown(full_response)
        ai_score_placeholder.markdown(full_response)

with tab4:
    system_prompt = """# CONTEXT # 
        I am a machine learning engineer at a startup company in the staffing solutions industry. In this field, I create machine learning models which will help the company to achieve their goals. Now, I am focused on talent recruitment by helping them to develop AI models specifically for the Talent Recruitment platform as well as the application tracking system.

        # OBJECTIVE #
        Your task is to help me in scoring an applicant assessment answer. This involves processing the question and sample or suggested answer by HR or client, Applicant answer, creating a scoring for each question (20 out of 20 scoring), and your analysis and insight.

        # STYLE #
        Write in a format which will be displayed in the pre-screening process. Ensure that each question does not show the sample or suggested answer by HR or client, but include the applicant answer, the score, and your insight or analysis.

        # Tone #
        Maintain a formal tone in the insights. Make sure to retain the tone as an HR Expert.

        # AUDIENCE #
        The target audience is individuals who are the clients which are the HR recruiter or hiring managers looking for talents to improve their team.

        # RESPONSE FORMAT #
        Provide first the question, then each question will show the applicant answer, your scoring (20 out of 20), and your analysis or insight.

        Example:
        Assessment Analysis:

        1. [ Insert Question]:
        - Applicant Answer:
        - AI Score: 17 out of 20
        - Insights: 

        2. [ Insert Question]:
        - Applicant Answer:
        - AI Score: 17 out of 20
        - Insights: 

        3. [ Insert Question]:
        - Applicant Answer:
        - AI Score: 17 out of 20
        - Insights: 

        4. [ Insert Question]:
        - Applicant Answer:
        - AI Score: 17 out of 20
        - Insights: 

        5. [ Insert Question]:
        - Applicant Answer:
        - AI Score: 17 out of 20
        - Insights: 

        Sum of AI Scores: 85/100
        Average AI Score: 17
        Overall Insight:  

    """

    interview_questions_and_answers = """
    1. Can you share a specific example of a challenging situation you encountered at work and how you handled it?
    - Suggested answer: In my previous role, we faced a tight deadline for a crucial project. I organized a team meeting to discuss priorities, delegated tasks based on team members' strengths, and implemented a more efficient workflow, which resulted in meeting the deadline successfully.
    - Applicant Answer: In a previous role, I encountered a situation where our project faced a tight deadline. I called for a team meeting to discuss the challenges, assigned tasks based on team members' strengths, and implemented a more efficient workflow. As a result, we successfully met the deadline and delivered a high-quality project.

    2. How do you prioritize and manage your workload when faced with multiple tasks or projects?
    - Suggested answer: I prioritize tasks by assessing deadlines, urgency, and impact on overall goals. I use project management tools to create a timeline and ensure that each task receives the attention it deserves. Regular check-ins help me adjust priorities if needed.
    - Applicant Answer: When dealing with multiple tasks or projects, I prioritize by assessing deadlines, urgency, and impact on overall goals. I utilize project management tools to create a detailed timeline, ensuring each task is given the necessary attention. Regular check-ins and flexibility in adjusting priorities help me maintain an organized workflow.

    3. Describe a situation where you had to adapt to a significant change in the workplace. How did you handle it?
    - Suggested answer: When our company underwent a restructuring, I proactively sought information, attended relevant workshops, and communicated openly with my team. I embraced the change, fostering a positive environment and helping team members adapt to new roles and responsibilities.
    - Applicant Answer: During a company restructuring, I proactively sought information, attended relevant workshops, and communicated openly with my team. I embraced the change, fostering a positive environment and helping team members adapt to new roles. My proactive approach contributed to a smooth transition, and the team continued to excel in our new structure.

    4. How do you approach collaboration and communication within a team?
    - Suggested answer: I believe in open communication and active collaboration. Regular team meetings, transparent communication channels, and encouraging diverse perspectives contribute to a positive team dynamic. I also make an effort to listen actively and ensure everyone's input is valued.
    - Applicant Answer: I approach collaboration and communication within a team through open dialogue and active engagement. I organize regular team meetings to discuss ongoing projects, ensure everyone is on the same page, and encourage the exchange of ideas. I believe in fostering a positive team dynamic where each member feels valued and heard.

    5. Can you provide an example of a successful project you initiated or contributed to? What was your role, and what was the outcome?
    - Suggested answer: In my previous position, I proposed and led a cross-functional initiative to streamline our customer support processes. By implementing new communication tools and training programs, we improved response times by 30% and received positive feedback from both customers and team members.
    - Applicant Answer: In a previous position, I initiated and led a cross-functional initiative to streamline customer support processes. My role involved proposing new communication tools and implementing training programs. The outcome was a 30% improvement in response times, resulting in positive feedback from both customers and team members.
    """

    gen_screening_text_area = st.empty()
    gen_screening_text_area.text_area("Input question and applicant answer")

    screening_rb = st.radio("Use Screening Sample Info", ['Yes', 'No'])

    gen_ai_screening = st.button('Analyze Applicant')

    if gen_ai_screening:
        prompt = ""
        if screening_rb == 'Yes':
            prompt = gen_screening_text_area.text_area("Input question and applicant answer", value=interview_questions_and_answers)
        else:
            prompt = gen_screening_text_area
        ai_screening_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            ai_screening_placeholder.markdown(full_response + "▌")
        ai_screening_placeholder.markdown(full_response)

with tab5:
    st.title("CV Parsing")
    st.subheader("with PDF and Docx")

    def convert_docx_to_markdown(input_file):
        doc = docx.Document(input_file)
        paragraphs = [p.text for p in doc.paragraphs]
        markdown_text = '\n'.join(paragraphs)
        return markdown_text

    def convert_pdf_to_markdown(input_file):
        with open(input_file, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
                markdown_text = markdown.markdown(text)
        return markdown_text
    
    def display_text(bounds):
        text = []
        for x in bounds:
            t = x[1]
            text.append(t)
        text = ' '.join(text)
        return text 

    input_file = st.file_uploader(label='Resume', type=['pdf', 'docx', 'jpg', 'jpeg', 'png']) 
    if st.button("Parse"):
        
        if input_file:
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, input_file.name)
            with open(path, "wb") as f:
                    f.write(input_file.getvalue())

            st.write(path)
            text = None
            if path.endswith('.docx'):
                text = convert_docx_to_markdown(path)
            elif path.endswith('.pdf'):
                text = convert_pdf_to_markdown(path)
            else:
                img = Image.open(path)
                img = np.array(img)
                
                with st.spinner('Extracting Text from given Image'):
                    eng_reader = easyocr.Reader(['en'])
                    detected_text = eng_reader.readtext(img)
                text = display_text(detected_text)

            resume_prompt = '''
                    The following is a resume format for a candidate named {{name}}, born on {{birthDate}}.

                    content: {
                        name: "{{name}}",
                        birthDate: "{{birthDate}}",
                        description: "null",

                        studentInfo: [
                        {
                            educationalBackground: "{{educationalBackground}}",
                            course: "{{course}}",
                            school: "{{school}}",
                            schoolYear: "{{schoolYear}}",
                        },
                        ],
                        skillSet: [
                        {
                            skill: "{{skill}}",
                            level: "{{level}}",
                            experience: "{{experience}}",
                        },
                        ],
                        workHistory: [
                        {
                            companyName: "{{companyName}}",
                            employmentType: "{{employmentType}}",
                            experienceDate: {
                            startDate: "{{startDate}}",
                            endDate: "{{endDate}}",
                            },
                            industry: "{{industry}}",
                            isPresent: {{isPresent}},
                            jobLevel: "{{jobLevel}}",
                            jobRole:
                            "{{jobRole}}",
                            workSetup: "{{workSetup}}",
                        },
                        ],
                        certificates: [
                        {
                            title: "{{certificateTitle}}",
                            organization: "{{certificateOrganization}}",
                            dateIssued: "{{certificateDateIssued}}",
                        },
                        ],
                    },
                    };
                    '''


            json_format = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": "system", "content": resume_prompt},
                    {"role": "user", "content": text}
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                json_format.markdown(full_response)