# ---------- Shahad's Code ----------
import re, copy, os, fitz, PyPDF2,io
import streamlit as st
import pandas as pd
from datetime import datetime
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas 
from reportlab.pdfbase.ttfonts import TTFont 
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics 
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib import colors 
from graphviz import Digraph
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationChain, ConversationalRetrievalChain
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain import PromptTemplate, LLMChain
from langchain.llms import Ollama
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dataset = [
    {
        'description': 'The analytics embodiment leverages AI, machine learning, predictive analytics, and computerized analytics to generate patent analytics and valuations. It includes smartphone applications for notifications, alerts, reminders, and navigation tools, along with social media features for user connections and crowdsourcing patent validity, relevance, and valuation.',
        'name': 'Crowdsourced and social media ip search and analytics platform',
        'references': 'US20230410233A1'
    },
    {
        'description': 'The invention is a system for easily installing plug-in voice boxes with AI and IoT circuits into intelligent support boxes connected to AC or low voltage DC grids. These support boxes enable bidirectional optical communication via rear and front optoports. Rear optoports connect to a controller using a precisely cut optical cable, while front optoports directly communicate with the plug-in voice boxes.',
        'name': 'Communication infrastructure devices and support tools',
        'references': 'AU2022201978B2'
    },
    {
        'description':'An enterprise system and method for maintaining and transitioning humans to a human-like self-reliant entity is presented. ',
        'name':'Human-like emulation enterprise system and method',
        'references':'US11287847B2',
    },
    {
        'description':'A method for building distributed applications used to integrate different kinds of data in different execution environments includes the steps of: accessing data from one or more data sources (982)',
        'name':'Systems and methods for dynamic lineage tracking, reconfiguration, and life cycle management ',
        'references':'JP2024054219A',
    },
    {
        'description':'Applied Artificial Intelligence Technology for Adaptive Natural Language Understanding Disclosed herein is computer technology that provides adaptive mechanisms for learning concepts that are expressed by natural language sentences, and then applies this learning to appropriately classify new natural language sentences with the relevant concept that they express.',
        'name':'Applied Artificial Intelligence Technology for Adaptive Natural Language Understanding',
        'references':'US10990767B1',
    },
    {
        'description':'Systems and methods for platform-independent whole-body image segmentation',
        'name':'System and method for platform independent whole body image segmentation',
        'references':'JP2024079801A',
    },
    {
        'description':'An ecosystem is configured to facilitate digital exchange of digital assets in a digital asset marketplace.',
        'name':'Methods and systems for management of a blockchain-based computer-enabled networked ecosystem',
        'references':'US20230230079A1',
    },
    {
        'description':'Data protection is a well-known challenge in the storage technology field from the perspective of security and resilience. With regard to conventional solutions, there are well-known solutions such as Erase Code, which is widely used in CDs, DVDs, QR Codes, etc. to improve error correction capabilities, and Shamirs Secret Sharing Scheme (SSSS), which uses polynomial interpolation techniques to protect secrets.',
        'name':'Method and system for distributed data storage with enhanced security, resilience and control - Patents.com',
        'references':'JP7504495B2',
    },
    {
        'description':'Computerized systems and methods are disclosed to generate a document from one or more first and second text prompts, generating one or more context-sensitive text suggestions using a transformer with an encoder on the text prompts and a decoder that produces a text expansion to provide the context-sensitive text suggestions based on the one or more first and second text prompts by applying generative artificial intelligence with token biased weights for zero-shot, one-shot or some-shot generation of the artificial intelligence context-sensitive text suggestions from the one or more first and second text prompts.',
        'name':'Machine content generation',
        'references':'US20230351102A1',
    },
    {
        'description':'The present invention provides a method and system delivering graph-based metric to measure a similarity between weighted sets of classifications codes (presented as nodes) defined on hierarchical taxonomy trees. ',
        'name':'Method and system for peer detection',
        'references':'US10019442B2',
    },
    {
        'description':'Candidate answers are generated by a question-answering system in response to a question from a user. One or more generated candidate answers are compared to previous question-answer sets.',
        'name':'Supplementing candidate answers',
        'references':'US9542447B1',
    },
    {
        'description':'A website building system (WBS) includes a processor implementing a machine learning feedback-based proposal module and a database storing at least the websites of a plurality of users of the WBS, and components of the websites.',
        'name':'System and method for integrating user feedback into website building system services',
        'references':'AU2020284731B2',
    },
    {
        'description':'Higher demands for adaptable, scalable and automated human resources, capabilities and services, such as human well-being, safety and security, universal health care system and educational institutions, are growing in our societies, leading the way to the creation of a new artificially-intelligent support system that can facilitate the lives of the many. ',
        'name':'Human-Robots: The New Specie',
        'references':'US20200167631A1',
    },
    {
        'description':'The present disclosure relates to an automatic syringe device. More specifically, the present disclosure relates to an automatic syringe device having skin pulling and pinching performance, an ergonomic automatic syringe holding device, and other ergonomic improvements for the automatic syringe.',
        'name':'Improved automatic syringe',
        'references':'JP7025362B2',
    },
    {
        'description':'A system for remote servicing of customers includes an interactive display unit at the customer location providing two-way audio/visual communication with a remote service/sales agent, wherein communication inputted by the agent is delivered to customers via a virtual Digital Actor on the display.',
        'name':'Virtual photorealistic digital actor system for remote service of customer',
        'references':'US10152719B2',
    },
    {
        'description':'This invention presents an innovative framework for the application of machine learning (ML) for identification of energetic materials with desired properties of interest. For the output properties of interest, we identify the corresponding driving (input) factors.',
        'name':'Machine Learning to Accelerate Design of Energetic Materials',
        'references':'US20220067249A1',
    },
    {
        'description':'Systems and methods for natural language processing and classification are provided. In some embodiments, the systems and methods include a communication editor dashboard which receives the message, performs natural language processing to divide the message into component parts.',
        'name':'Systems and methods for automated question response',
        'references':'US11010555B2',
    },
    {
        'description':'Embodiments concern methods and compositions for treating or preventing a bacterial infection, particularly infection by a',
        'name':'Compositions and methods related to antibodies to staphylococcal protein a',
        'references':'AU2012296576B2',
    },
    {
        'description':'An electronic furniture assembly of the present invention comprises: (i) a furniture assembly comprising: (A) a base (e.g., a seat portion), (B) at least one transverse member (e.g., a side, armrest or backrest), and (C) a coupler for selectively coupling the base to the transverse member; and (ii) artificial intelligence mounted within one or more portions of the furniture assembly.',
        'name':'Electronic furniture systems with integrated artificial intelligence',
        'references':'US10979241B2',
    },
    {
        'description':'The disclosed visual RRC-humanoid robot is a computer-based system that has been programmed to reach human-like levels of visualization Artificial Intelligence (AI).',
        'name':'Intelligent visual humanoid robot and computer vision system programmed to perform visual artificial intelligence processes',
        'references':'US9573277B2',
    },
    {
        'description':'An Internet appliance, comprising, within a single housing, packet data network interfaces, adapted for communicating with the Internet and a local area network, at least one data interface selected from the group consisting of a universal serial bus, an IEEE-1394 interface, a voice telephony interface, an audio program interface, a video program interface, an audiovisual program interface, a camera interface, a physical security system interface, a wireless networking interface',
        'name':'Internet appliance system and method',
        'references':'US8583263B2',
    },
    {
        'description':'CROSS-REFERENCE TO RELATED APPLICATIONS This application claims the benefit of U.S. Provisional Application No. 62/924,586, filed October 22, 2019, entitled “Personalized Viewing Experience and Contextual Advertising in Streaming Media based on Scene Level Annotation of Videos,” by inventors Suresh Kumar and Raja Bala, Attorney Docket No. PARC-20190487US01, the disclosure of which is incorporated herein by reference.',
        'name':'Systems and methods for generating localized context video annotations ',
        'references':'JP7498640B2',
    },
    {
        'description':'Provided is a process including: receiving a data token to be passed from a first node to a second node; retrieving machine learning model attributes from a collection of one or more of the sub-models of a federated machine-learning model; determining based on the machine learning model attributes, that the data token is learning relevant to members of the collection of one or more of the sub-models and, in response, adding the data toke to a training set to be used by at least some members of the collection of one or more of the sub-models; determining a collection of data tokens to transmit from the second node to a third node of the set of nodes participating in a federated machine-learning model; and transmitting the collection of data tokens.',
        'name':'Federated machine-Learning platform leveraging engineered features based on statistical tests',
        'references':'US20210174257A1',
    },
    {
        'description':'Disclosed herein are example embodiments of an improved narrative generation system where an analysis service that executes data analysis logic that supports story generation is segregated from an authoring service that executes authoring logic for story generation through an interface. Accordingly, when the authoring service needs analysis from the analysis service, it can invoke the analysis service through the interface. By exposing the analysis service to the authoring service through the shared interface, the details of the logic underlying the analysis service are shielded from the authoring service (and vice versa where the details of the authoring service are shielded from the analysis service). Through parameterization of operating variables, the analysis service can thus be designed as a generalized data analysis service that can operate in a number of different content verticals with respect to a variety of different story types.',
        'name':'Applied artificial intelligence technology for narrative generation using an invocable analysis service',
        'references':'US12001807B2',
    },
    {
        'description':'This application relates to a method for amplifying genomic DNA, including: (a) providing a first reaction mixture, including a sample containing genomic DNA, a first primer, a nucleotide monomer mixture, and a nucleic acid polymerase, wherein the first primer From the 5 end to the 3 end, the universal sequence and the first variable sequence including the first random sequence are included; (b) the first reaction mixture is placed in the first temperature loop program to obtain the pre-amplified product; (c) provided The second reaction mixture includes a pre-amplification product, a second primer, a nucleotide monomer mixture, and a nucleic acid polymerase, wherein the second primer contains or consists of a specific sequence and the universal sequence from the 5 end to the 3 end Composition; (d) placing the second reaction mixture in a second temperature loop program for amplification to obtain amplified products. The application also relates to a kit for amplifying genomic DNA.',
        'name':'DNA amplification method',
        'references':'TWI742059B',
    },
    {
        'description':'A fungicidal composition comprising a mixture of an N-phenylamidine defined by formula (I) as defined in claim 1 as component (a) and a further pesticide as component (B), and the use of said composition in agriculture or horticulture for controlling or preventing infestation of plants by phytopathogenic microorganisms, preferably fungi.',
        'name':'Fungicidal compositions',
        'references':'CN110740644B',
    },
    {
        'description':'An automated system/method for identifying and enabling viewer selection/purchase of products or services associated with digital content presented on a display device. Products within the digital content are identified and existing product placement data is ascertained. For products that do not include such data, other methodologies, with the assistance of third-party servers, are employed to assess identity and purchase availability. Viewer input designate products to assess or products can be automatically assessed. Viewers initiate purchase of identified products via the display device or other electronic devices controlled by viewers, such as via viewers'' smart phones. Various processes for identifying products include use of AI processing, access to data on third-party servers, crowd sourcing and other methodologies. Various techniques for selecting products for purchases are employed including employing 3D codes (e.g., QR codes) alongside presented products to enable other portable electronic devices to facilitate purchase. Other features are described.',
        'name':'Systems/methods for identifying products within audio-visual content and enabling seamless purchasing of such identified products by viewers/users of the audio-visual content',
        'references':'US11416918B2',
    },
    {
        'description':'Disclosed herein are example embodiments that describe how a narrative generation techniques can be used in connection with data visualization tools to automatically generate narratives that explain the information conveyed by a visualization of a data set. In example embodiments, new data structures and artificial intelligence (AI) logic can be used by narrative generation software to map different types of visualizations to different types of story configurations that will drive how narrative text is generated by the narrative generation software.',
        'name':'Applied Artificial Intelligence Technology for Automatically Generating Narratives from Visualization Data',
        'references':'US20220114206A1',
    },
    {
        'description':'A LED Light device for house or stores or business application having built-in camera-assembly is powered by AC or-and DC power source for a lamp-holder, LED bulb, security light, flashlight, car torch light, garden, entrance door light or other indoor or outdoor LED light device connected to power source by (1) prongs or (2) male-base has conductive piece can be inserted into a female receiving-piece which connect with power source or (3) wired or AC-plug wires. The device has built-in camera-assembly has plurality functions to make different products and functions. The LED light device has at least one of (a) camera or DV (digital video) to take minimum MP4 or 4K, 8K image or photos, (b) digital data memory kits or cloud storage station, (c) wireless connection kits, Bluetooth or USB set for download function, (d) MCU or CPU or IC with circuit with desired motion sensor /moving detector(s) /other sensor, (e) camera-assembly for connecting Wi-Fi, Wi-Fi extend, or-and 3G/4G/5G internet or network or even settle-lite channel, (f) wireless-system to transmit or-and receiving wireless signal, (g) people had download APP or other platform incorporated with pre-programed or even AI (artificial intelligence) software to operate one or more of area-selections function to make screen or image comparison, detection, identification, recognition, tracing, purchase, or other pre-program following works including but not limited to detect moving object(s), face recognition or personal identification or-and habit or-and crime comparison, purchase, (h) LED light source to offer sufficient brightness under dark environment for camera-assembly take colorful or-and audio data, (i) other electric or mechanical parts & accessories, (j) has moving detector and software built-in to make comparison to judge the moving object(s) from the preferred screen selected-areas; to get desired function(s) for the said LED light device. The said camera-assembly has desired camera, wireless-system, sensor(s), part(s) and related module or circuit(s) or interface or-and backup power, and (k) camera-assembly may in separated housing incorporated with all kind of existing non-built-in camera light device so people can upgrade the non-camera device to has built-in camera and digital device for out-of-date non-camera all kind of light device including security light.',
        'name':'LED Light Has Built-In Camera-Assembly for Colorful Digital-Data Under Dark Environment',
        'references':'US20180332204A1',
    },
    {
        'description':'Narrative generation techniques can be used in connection with data visualization tools to automatically generate narratives that explain the information conveyed by a visualization of a data set. In example embodiments, new data structures and artificial intelligence (AI) logic can be used by narrative generation software to map different types of visualizations to different types of story configurations that will drive how narrative text is generated by the narrative generation software.',
        'name':'Applied artificial intelligence technology for using narrative analytics to interactively generate narratives from visualization data',
        'references':'US11188588B1',
    }]

def load_csv(file_path):
    try:
        loader = CSVLoader(
            file_path=file_path,
            csv_args={"delimiter": ",", "quotechar": '"'},
            encoding='utf-8-sig'
        )
        return loader.load()
    except RuntimeError as e:
        st.error(f"Runtime error loading the file: {e}")
        return []

# Load data and preprocess
data = load_csv(r"C:\Users\fareah0b\Documents\Patent-Project\30Patents.csv")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=40)
text_chunks = text_splitter.split_documents(data)

llm = Ollama(model='llama3:instruct', callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = 'chroma_db'

db = Chroma.from_documents(documents=text_chunks, embedding=embedding_function, persist_directory=persist_directory)
db.persist()

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, verbose=True, memory=memory)
retriever = db.as_retriever()
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
retrieval = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

#-------------------------------------- Generating an answer ----------------------------------------
def generate_answer(idea, first):
    if first:
        memory.clear()
    
    similarity = db.similarity_search(idea, k=1)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([doc['description'] for doc in dataset])
    idea_tfidf = tfidf_vectorizer.transform([idea])
    similarity_scores = cosine_similarity(idea_tfidf, tfidf_matrix)
    most_similar_index = similarity_scores.argmax()
    max_similarity = similarity_scores[0, most_similar_index]
    similarities = [result for result in similarity]
    
    threshold = 0.5
    if max_similarity > threshold:
        st.error("Similarity found, so your idea is not patentable")
        Innovation_Generator(idea)
    else:
        st.success("Found no similar idea, so your idea is patentable")
        
# I commented it since it's not required anymore
        # Generate and display description and claims
        # description_claims = conversation.predict(
        #     input=f"Could you add detailed description and claims to the idea: {idea} (just give a comprehensive answer without branching into additional topics)")
        # st.info(description_claims)
# -------------------------------------- Innovation Generator --------------------------------------------------
def Innovation_Generator(idea):
    st.info(conversation.predict(input="Could you add features to make the user's idea unique and add detailed description to the idea? (just give a comprehensive answer without branching into additional topics)"))
# I did not use this method in this code version
# -------------------------------------- Idea Builder --------------------------------------------------
def Idea_builder(filtered_components, response):
    os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
    dot = Digraph()

    # Add nodes with specific shapes and colors (First and last node always)
    for i, component in enumerate(filtered_components):
        # Determine the shape based on the component text
        if 'If ' in component or '?' in component:
            shape = 'diamond'
        else:
            shape = 'box' if i != 0 and i != len(filtered_components) - 1 else 'ellipse'
        
        color = 'lightblue'

        dot.node(component, shape=shape, color=color, style='filled', fontsize='10')

    # Add edges between nodes
    for i in range(len(filtered_components) - 1):
        dot.edge(filtered_components[i], filtered_components[i + 1], style ='solid', penwidth='2')

    return dot.source
# --------------------------------------- Create & Write --------------------------------------------
def create(output_pdf, header_path, info_path, page_2_data):
    documentTitle = 'Invention Disclosure Form'
    title = 'Invention Disclosure Form'
    textLines = []
    key_lines = []  # To keep track of keys with lines

    # Prepare textLines with formatted keys and values
    for key, value in page_2_data.items():
        if key.endswith(':'):
            formatted_key = f'<b>{key[:-1]}</b>'
            key_lines.append(len(textLines))  # Track index for keys that originally had ':'
        else:
            formatted_key = f'<b>{key}</b>'
        
        formatted_value = f'<br />{value}'
        textLines.append(f'{formatted_key}<br />{formatted_value}')

    pdf = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    pdf.setTitle(documentTitle)

    header_width = 100
    header_height = 50
    header_x = 10
    header_y = height - 50

    def draw_header():
        try:
            pdf.drawInlineImage(header_path, header_x, header_y, width=header_width, height=header_height)
        except Exception as e:
            print(f"Error drawing header image: {e}")
        pdf.setFont('Courier-Bold', 14)
        pdf.drawCentredString(width / 2, height - 80, title)
        line_y_position = height - 130
        pdf.line(30, line_y_position, width - 30, line_y_position)

    # Create ParagraphStyles for key and value
    styles = getSampleStyleSheet()
    text_x = 40
    text_y = height - 100
    max_width = width - 80
    line_spacing = 7  # Space between lines

    # Draw the initial header and content
    draw_header()

    # Draw text on PDF
    for idx, line in enumerate(textLines):
        p = Paragraph(line, style=styles['Normal'])
        p_width, p_height = p.wrap(max_width, 0)

        if text_y - p_height - line_spacing < 0:
            # If space is insufficient, create a new page
            pdf.showPage()
            text_y = height - 100
            draw_header()  # Redraw header on the new page

        if idx in key_lines:
            # Draw a line above the key
            pdf.line(text_x, text_y, text_x + max_width, text_y)
            text_y -= line_spacing  # Space between line and text

        # Draw the paragraph (key and value)
        p.drawOn(pdf, text_x, text_y - p_height)
        text_y -= p_height + line_spacing

        if idx in key_lines:
            # Draw a line below the key and value
            pdf.line(text_x, text_y, text_x + max_width, text_y)
            text_y -= line_spacing  # Space between line and next content

    pdf.save()
# --------------------------------------- post_process_text --------------------------------------------
def post_process_text(text):
    cleaned_text = text.strip() # Clean up the text
    
    # Remove unwanted introductory phrases
    intro_patterns = [
        r"^Here is the friendly conversation between a human and an AI[:\s]*",
        r"^Here are the answers[:\s]*",
        r"^The answer is[:\s]*",
        r"^Answer[:\s]*",
        r"^Note[:\s]*",
        r"^I would say[:\s]*",
        r"^In response[:\s]*",
        r"^(Note: These technical problems were solved during the development of this system that manages digital rights pertaining to the 3D printing of a 3D digital model or object.)"
    ]
    for pattern in intro_patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE).strip()

    # Remove repeated prompt from the response
    prompt_pattern = re.escape(prompt).strip()
    if cleaned_text.startswith(prompt_pattern):
        cleaned_text = cleaned_text[len(prompt_pattern):].strip()

    # Remove unwanted keywords
    unwanted_keywords = ["irrelevant", "error"]
    for keyword in unwanted_keywords:
        cleaned_text = cleaned_text.replace(keyword, "[REMOVED]")
    
    # Ensure the response is concise
    if len(cleaned_text) > 200:
        cleaned_text = "Response is too lengthy and may include unnecessary details."

    return cleaned_text
# -------------------------------------- Streamlit app -----------------------------------------
st.set_page_config(page_title="InventoFill", page_icon="🧠", layout="wide")
# -------------------------------------- main function --------------------------------------------------
def main():
    st.title("InventoFill")
    page_2_data = {}
    output_pdf = r'C:\Users\fareah0b\Downloads\filled_form.pdf'
    header_path = r"C:\Users\fareah0b\Downloads\saudi-aramco-logo.jpeg"
    info_path = r"C:\Users\fareah0b\Downloads\info.jpeg"
        
    page = st.sidebar.radio("Patent Assessor Pages", ["Page 1", "Page 2"])

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "Page 1"
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
    if 'idea' not in st.session_state:
        st.session_state.idea = ""
    if 'page_2_data' not in st.session_state:
        st.session_state.page_2_data = {}

    if page == "Page 1":
        with st.form(key='idea_form'):
            title=st.text_input("**1. Title of Invention:**")
            idea = st.text_area("**2. Brief Summary of the Invention:**", height=100, value=st.session_state.idea)
            first = st.checkbox("**Is this the first time you are submitting the idea?**")

            submit_button1 = st.form_submit_button("Submit Idea")

        if submit_button1:
            if not idea:
                st.error("Please enter a brief summary of the invention.")
            else:
                st.session_state.title = title
                st.session_state.idea = idea
                st.session_state.first = first
                generate_answer(idea, True)

        if 'form_submitted' in st.session_state:
                st.session_state.form_submitted = False
                
    if page == "Page 2":
        with st.form(key='pdf_form'):
            if st.session_state.idea == "":
                st.error("Idea not found. Please submit an idea on Page 1 first.")
                st.stop() # Stops further execution if no idea i

            title = st.session_state.title
            idea = st.session_state.idea  # Retrieve idea from session_state
            st.write("**3. Technology Division / Focus Area:**")
            answer = conversation.predict(input = f'''Answer the following question directly and without additional information: [Provide the technology division/focus area of the idea e.g IoT or Generative AI or Autonomous Systems or Predictive AI or Machine Learning]''')
            st.text_input("", value=answer)
            
            previous_inven = st.selectbox("**4. Is this invention related to a previously disclosed invention?**", ['Yes', 'No'])
            if previous_inven == 'Yes':
                sa_number = st.text_input("**Enter SA Number:**")

            sub1 = st.subheader("Invertors")
            num_inventors = int(st.number_input("**5. Number of Inventors (Maximum 3):**", min_value=1, max_value=3))
            for i in range(num_inventors):
                st.subheader(f"Inventor {i+1}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    full_name = st.text_input(f"**Full Name {i+1}:**", key=f"full_name_{i}")
                    citizenship = st.text_input(f"**Citizenship {i+1}:**", key=f"citizenship_{i}")
                    employer = st.text_input(f"**Employer at the Time of Invention {i+1}:**", key=f"employer_{i}")
                with col2:
                    alias = st.text_input(f"**Alias (outlook) {i+1}:**", key=f"alias_{i}**")
                    badge = st.text_input(f"**Badge Number {i+1}:**", key=f"badge_{i}")
                    job = st.text_input(f"**Job Title {i+1}:**", key=f"job_{i}")
                with col3:
                    organ = st.text_input(f"**Organization {i+1}:**", key=f"organ_{i}")
                    loc = st.text_input(f"**Location {i+1}:**", key=f"loc_{i}")
                    contri = st.text_area(f"**Contribution to Invention {i+1}:**", height=100, key=f"contri_{i}")

                # Office Address
                sub2 = st.subheader(f"Office Address {i+1}")
                col1, col2 = st.columns(2)
                with col1:
                    office_address = st.text_input(f"**Office Address {i+1}:**", placeholder="123 King's Road", key=f"office_address_{i}")
                    phone1 = st.text_input(f"**Phone (Office):**", placeholder="+966 (000) 123-4567", key=f"phone1_{i}")
                    mobile = st.text_input(f"**Mobile (Office):**", placeholder="+966 (000) 123-4567", key=f"mobile_{i}")
                with col2:
                    fax = st.text_input(f"**Fax (Office):**", placeholder="+966 (000) 123-4567", key=f"fax_{i}")
                    email = st.text_input(f"**Email (Office):**", placeholder="example@domain.com", key=f"email_{i}")
                
                # Mailing Address
                sub3 = st.subheader(f"Mailing Address {i+1}")
                mail_address = st.text_input(f"**Mailing Address {i+1}:**", placeholder="123 King's Road", key=f"mail_address_{i}")
                
                # Home Address
                sub4 = st.subheader(f"Home Address {i+1}")
                col1, col2 = st.columns(2)
                with col1:
                    home_address = st.text_input(f"**Home Address {i+1}:**", placeholder="123 King's Road", key=f"home_address_{i}")
                    phone1_home = st.text_input(f"**Phone 1 (Home):**", placeholder="+966 (000) 123-4567", key=f"phone1_home_{i}")
                    mobile_home = st.text_input(f"**Mobile (Home):**", placeholder="+966 (000) 123-4567", key=f"mobile_home_{i}")
                with col2:
                    fax_home = st.text_input(f"**Fax (Home):**", placeholder="+966 (000) 123-4567", key=f"fax_home_{i}")
                    email_home = st.text_input(f"**Email (Home):**", placeholder="home@example.com", key=f"email_home_{i}")

                # Manager's
                sub5 = st.subheader(f"Manager’s name and Address {i+1}:")
                col1, col2 = st.columns(2)
                with col1:
                    manager_name = st.text_input(f"**Manager Name {i+1}:**", placeholder="Khalid Alqhatani", key=f"manager_name{i}")
                    manager_address = st.text_input(f"**Manager Address {i+1}:**", placeholder="123 King's Road", key=f"manager_address_{i}")
                with col2:
                    phone1_manager = st.text_input(f"**Phone:**", placeholder="+966 (000) 123-4567", key=f"phone1_manager_{i}")
                    email_manager = st.text_input(f"**Email:**", placeholder="home@example.com", key=f"email_manager_{i}")
                
                st.write('''I hereby solemnly swear and affirm under oath that I am an inventor of this invention and that I have not knowingly omitted the inclusion
                of any other inventor besides me, and that the information provided in this disclosure is, to the best of my knowledge, true and accurate.''')
                uploaded_file = st.file_uploader("**Choose an image file**", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    st.image(uploaded_file, caption="Signature", use_column_width=True)
                    with open("signature_uploaded.png", "wb") as f:
                        f.write(uploaded_file.read())
                    st.success("Signature saved as 'signature_uploaded.png'")

                now_date = st.date_input("**Select a date**", datetime.now())

                # Invention Details
                sub6= st.subheader(f"Invention Details")
                st.write("**Technology Type:**")
                type =conversation.predict(input = f'''Answer the following question directly and without additional information: [Provide the Technology Type of the {idea} Software or algorithm or Chemical compound or formulation or Apparatus or device or Method or process ]''')
                st.text_input("", value=type)

                inventors = []
                inventors.append({
                "full name": full_name,
                "citizenship": citizenship,
                "employer at time of invention": employer,
                "alias": alias,
                "badge": badge,
                "job": job,
                "organ": organ,
                "loc": loc,
                "contri": contri,
                "office_address": office_address,
                "phone1": phone1,
                "mobile": mobile,
                "fax": fax,
                "email": email,
                "mail address": mail_address,
                "home address": home_address,
                "phone1 home": phone1_home,
                "mobile home": mobile_home,
                "fax home": fax_home,
                "personal email": email_home,
                "manager name": manager_name,
                "manager address": manager_address,
                "phone1 manager": phone1_manager,
                "email manager": email_manager })
                
#------------------ Detailed description of the invention -----------------------
                sub7 = st.subheader(f"Detailed description of the invention:")
                response = conversation.predict(input=f'''briefly list version of key steps, bold out the main steps of developing for the following idea: {idea}, and make the first step always 1.Start and Last step always End
                Always write straight forward answer similar to this format:
                **Start**
                **1. Model Licensing**
                **2. Access Control**
                **3. Usage Tracking**
                **4. Digital Watermarking**
                **5. Royalty Collection**
                **6. Notifications and Alerts**
                **7. Integration**
                **End**''')
            # Split the response into different lines
                components = [comp.strip() for comp in response.split('\n') 
                            if comp.strip() and (comp[0].isdigit() or
                                                comp.strip().startswith("I. ") or 
                                                comp.strip().startswith('•') or 
                                                comp.strip().startswith('-') or
                                                comp.strip().startswith('*'))]

            # Extract only the text up to the first colon ":"
                filtered_components = []
                for comp in components:
            # Take text before the first colon, if present
                    if ':' in comp:
                        filtered_components.append(comp.split(':')[0].strip())
                    else:
                        filtered_components.append(comp.strip())
            
                st.write("**Describe the invention in detail with reference to drawings/ diagrams/ formulas:** ")
                description = conversation.predict(input = f'''Describe the {idea} in detail,  without any additional details neither intro nor outro nor "Here is the response:" or similar respond (just the main answer)''')
                st.text_area("", value=description, height = 500)
                graph_source = Idea_builder(filtered_components, response) 
                graph_dict = st.graphviz_chart(graph_source)

                st.write("**Describe any unexpected results of your invention.**")
                unexpected =conversation.predict(input = f'''Answer the following question directly and without additional information: [
                                            Unexpected results of this {idea} can include:
                                            1. Unintended Side Effects: The invention may cause unforeseen side effects or health issues.
                                            2. Environmental Impact: It could have adverse effects on the environment that were not initially considered.
                                            3. Market Reactions: The invention might not perform as expected in the market, leading to lower sales or negative customer feedback.
                                            4. Technical Issues: The invention might have technical problems that were not identified during development, leading to functionality issues.
                                            5. Legal Challenges: There could be unforeseen legal issues, such as patent disputes or regulatory challenges.
                                            6. Social Impact: The invention might have unexpected social consequences, such as altering social behavior or contributing to inequality. ]''')
                st.text_input("", value = unexpected)

                st.write("**Technical problem(s) solved:**")
                tech_prob = conversation.predict(input = f'''Answer the following question directly and without additional information: [Technical problem(s) solved of developing this {idea}]''')
                st.text_area("", value=tech_prob, height = 500)
                
                aramco_q = st.write("**Is this related to any current Aramco research or operations?**")
                aramco_q2= conversation.predict(input = f'''Answer the following question directly and without additional information: [what technical problem(s) does this invention solve?]''')
                st.text_area("", value=aramco_q2, height = 500)
                
                st.write("**What is your proposed technical solution to the technical problem stated above?**")
                aramco_q3= conversation.predict(input = f'''Answer the following question directly and without additional information: [What is your proposed technical solution to the technical problem stated here{aramco_q2}]''')
                st.text_area("", value=aramco_q3, height = 500, key = 'aramco question 3')
                
                conduct= st.text_input("**Did you conduct any lab or pilot plant experiments to verify your invention? If so, explain.**")
                
                helpful_data = st.file_uploader("**Identify and provide any data, diagrams, tables, pictures, etc. from your experiments that would help explain your invention.**")
                if helpful_data is not None:
                    if helpful_data.type.startswith("image/"):
                        st.image(helpful_data, caption="Uploaded Image", use_column_width=True)
                    with open("uploaded_file", "wb") as f:
                        f.write(helpful_data.read())
                    st.success("Data Saved")
                
                uploaded_file2 = st.file_uploader("**Attach copies of key lab notebook records and test data.**", type=["ipnyb", "py"])
                if uploaded_file2 is not None:
                    st.image(uploaded_file2, caption="test data", use_column_width=True)
                    with open("test_data.ipynb", "wb") as f:
                        f.write(uploaded_file2.read())
                    st.success("data saved as 'test_data.ipynb'")

                # Invention History
                sub8 = st.subheader(f"Invention History")
                estimate_date = st.date_input("**Select estimate of the date when the following occurred or will occur:**", datetime.now())

                date_q = st.radio("**Date:**", ('A. Conception of invention', 'B. First written description (attach a copy, if available', '''C. First public disclosure (external to Aramco) giving an enabling description of your 
                                            invention, for example, a disclosure made in an abstract, proposal, paper submission, 
                                            talk, or meeting with industry. Please attach a copy of the disclosure, if available, 
                                            stating to whom and where the disclosure was made. If the only public disclosure was 
                                            made orally, then provide a description of the disclosure and state to whom the 
                                            disclosure was made.''', 'D. Completion of model or prototype', 'E. First operational test'))
                
                publication = st.radio("**Do any Company publication(s) or industry journal articles submitted by you describe the invention in whole or in part (news, articles or scientific papers)? **", ('Yes', 'No'))
                publication_= st.text_input("**provide date, reference information and copies, if available.**")
                publication2 = st.radio("**Is a publication, paper or other disclosure planned within the next 6 months? **", ('Yes', 'No'))

                # 3. Background Research And Prior Art
                sub9 = st.subheader(f"Background Research And Prior Art")

                st.write("**Conduct a literature and patent search and attach a list of all relevant patents and publications that result**")
                publication3 = conversation.predict(input = f'''Answer the following question directly and without additional information: [Conduct a literature and patent search and attach a list of all relevant patents to {idea} and publications that result.]''')
                st.text_area("", value=publication3, height = 500)
                
                st.write("**List key words used in search (attach search parameters).**")
                publication4 = conversation.predict(input = f"Answer the following question directly and without additional information: [ List key words used in search (attach search parameters) to {idea}]")
                st.text_area("", value=publication4, height = 500)                

                st.write("**List databases used in search.**")
                publication5 = conversation.predict(input = f"Answer the following question directly and without additional information: [ List databases used in search to {idea}]")
                st.text_area("", value=publication5, height = 500)
                attached = st.checkbox("**I have attached the publications and references that I believe to be relevant prior art.**")
                
                st.write("**From the research/ prior art found, please identify the most relevant reference(s) that cover the invention**")
                prior= conversation.predict(input = f"Answer the following question directly and without additional information: [identify the most relevant reference(s) that cover the invention {idea} \n Patent Databases: reference: \n Google Patents reference:]")
                st.text_area("", value=prior, height = 500)
                
                st.write("**Please explain technical differences or advantages of the invention over prior art:**")
                prior2= conversation.predict(input = f'''Answer the following question directly and without additional information: [Please explain technical differences or advantages of the invention over prior art: {idea}
                                Differences or advantages  ]''')
                st.text_area("", value=prior2, height = 500,key=f"prior_{i}")
                
                st.write("** What are the deficiencies of the prior art methods or approach(s) for solving the technical problem? What are the technical improvements and technical advantages provided by this invention? **")
                prior3= conversation.predict(input = f''''Answer the following question directly and without additional information: [What are the deficiencies of the prior art methods or approach(s) for solving the technical problem? What are the 
                                technical improvements and technical advantages provided by this invention {idea}
                                 technical improvements:
                                technical advantages:]''')
                st.text_area("", value=prior3, height = 500)
                
                st.write("**Distinguish why this invention is better and different over combinations of methods cited in the prior art.**")
                prior4= conversation.predict(input = f''''Answer the following question directly and without additional information: [Distinguish why this invention is better and different over combinations of methods cited in the prior art.{idea}]''')
                st.text_area("", value=prior4, height = 500)

            # 4. Commercialization Potential
                sub10 = st.subheader(f"Commercialization Potential")
                st.write("**What are the limitations that must be overcome prior to practical application?**")
                commer = conversation.predict(input = f'''A. What are the limitations that must be overcome prior to practical application{idea}
                                Always write straight forward answer similar to this format: The limitations that must be overcome: " without any additional details neither intro nor outro nor "Here is the response:" or similar respond (just the main answer)''')
                st.text_area("", value=commer, height = 500)
                
                st.write("**What are the advantages of the invention versus commercially available alternatives?**")
                commer2= conversation.predict(input = f'''B. What are the advantages of the invention versus commercially available alternatives?{idea}
                                Always write straight forward answer similar to this format: The advantages of the invention versus commercially available: " without any additional details neither intro nor outro nor "Here is the response:" or similar respond (just the main answer)''')
                st.text_area("", value=commer2, height = 500)

                commer3 = st.radio("**Do you anticipate further development of the invention, e.g., the addition of additional features or improvements, over the next 12 months?**", ('Yes','No'))
                
                commer4 = st.text_area("**Do you know of any industrial organizations that may be interested in licensing this technology? If so, include company name, contact person and contact information.**")

            # 5. Third-Party Obligations Outside of Aramco
                sub11 = st.subheader(f"Third-Party Obligations Outside of Aramco")
                third = st.radio("**Are you aware of any third-party background intellectual property related to this invention?**", ('Yes','No'))
                third2 = st.radio("**Are you now collaborating or have you previously collaborated with any third parties (anyone outside of Aramco) on this invention?**", ('Yes','No'))
                third2_= st.text_input("**Who else collaborated with you and what is his/ her specific contribution to the invention?**")
                third3 = st.write("**Please indicate the type of agreement, if any, which governs the involvement of any third-party collaborator for this invention and provide a copy, if available**")
                
                third4 = st.checkbox("General Contract")
                contract_number = st.text_input("Contract #", placeholder="Enter contract number")
                third5 = st.checkbox("Other (e.g., masters and/or Doctoral thesis) (Explain the circumstances and details of the 3rd party Involvement). ")
                third6 = st.radio("**Was a Non-Disclosure Agreement or Confidential Disclosure Agreement signed with the company before discussing this invention?**", ('Yes','No'))
                third6_= st.text_input("**Provide the Non-Disclosure or Confidential Disclosure Agreement Reference Number?**")
            # 6. To Be Completed By Inventor’s Supervisor
                sub12 = st.subheader(f"6. To Be Completed By Inventor’s Supervisor")
                super = st.radio("**Is the inventor currently working with any of the following companies on developing this technology: (a) Saudi Aramco; or (b) a Saudi Aramco subsidiary?**", ('Yes','No'))
                super_= st.text_input("**Identify the company and the relevant department within the company?**")
                super2 = st.radio("**Is the invention being used by any of the following companies today: (a) Saudi Aramco; or (b) a Saudi Aramco subsidiary?**", ('Yes','No'))
                super2_= st.text_input("**Identify the company?**")
                super3= st.text_input("**What are the expected commercial applications for this invention?**")
                super4= st.text_input("**Is this invention part of an on-going project? If so, please identify the project**")

            # 7. Execution By Witnesses
                sub13 = st.subheader(f"7. Execution By Witnesses")
                wit = st.checkbox("**I have read this invention disclosure (including attached pages, if any) and understand its subject matter.**")
                signature = st.text_input("**Signature**")
                wit2 = st.text_input("**Print Name:**")
                wit3 = st.date_input("**Date**", datetime.now())
            page_2_data = {
            "Title of Invention:":title,
            "Brief Summary of The Invention:" : idea,
            "Technology Division / Focus Area:": answer,
            "Is This Invention Related To A Previously Disclosed Invention?:" : previous_inven,
            "If Yes, SA Number:": sa_number,
            "Signature": uploaded_file,
            "Date":now_date,
            "1.Invention Details:":"",
            "Technology Type": type,
            "Detailed description of the invention: INCLUDE ALL SUPPORT MATERIALS, FIGURES, REPORTS, LAB NOTEBOOKS AND PROPOSED PUBLICATIONS:":"",
            "Describe the invention in detail": description,
            "Flow chart of the software":response,
            "Describe any unexpected results of your invention": unexpected,
            "Technical problem(s) solved":tech_prob,
            "Is this related to any current Aramco research or operations?":aramco_q,
            "If so, what technical problem(s) does this invention solve? ":aramco_q2,
            "What is your proposed technical solution to the technical problem stated above? ":aramco_q3,
            "Did you conduct any lab or pilot plant experiments to verify your invention? If so, explain":conduct,
            "Identify and provide any data, diagrams, tables, pictures, etc. from your experiments that would help explain your invention":helpful_data,
            "Attach copies of key lab notebook records and test data. (These essential records define the priority date and verify the invention.)":uploaded_file2, 
            "2. Invention History: ":"",
            "Select estimate of the date when the following occurred or will occur":estimate_date,
            "Date: ":date_q,
            "Do any Company publication(s) or industry journal articles submitted by you describe the invention in whole or in part (news, articles or scientific papers)?": publication,
            "Provide date, reference information and copies, if available.": publication_,
            "Is a publication, paper or other disclosure planned within the next 6 months?":publication2,
            "3. Background Research And Prior Art: ":"",
            "Conduct a literature and patent search and attach a list of all relevant patents and publications that result. ":publication3,
            "List key words used in search (attach search parameters).":publication4,
            "List databases used in search.":publication5,
            "I have attached the publications and references that I believe to be relevant prior art":attached,
            "From the research/ prior art found, please identify the most relevant reference(s) that cover the invention":prior,
            "Please explain technical differences or advantages of the invention over prior art":prior2,
            "What are the deficiencies of the prior art methods or approach(s) for solving the technical problem? What are the technical improvements and technical advantages provided by this invention": prior3, 
            "Distinguish why this invention is better and different over combinations of methods cited in the prior art":prior4,
            "4. Commercialization Potential: ":"",
            "What are the limitations that must be overcome prior to practical application":commer,
            "What are the advantages of the invention versus commercially available alternatives?":commer2,
            "Do you anticipate further development of the invention, e.g., the addition of additional features or improvements, over the next 12 months?":commer3,
            "Do you know of any industrial organizations that may be interested in licensing this technology? If so, include company name, contact person and contact information":commer4,
            "5. Third-Party Obligations Outside of Aramco: ":"",
            "Are you aware of any third-party background intellectual property related to this invention?":third,
            "Are you now collaborating or have you previously collaborated with any third parties (anyone outside of Aramco) on this invention?":third2,
            "Who else collaborated with you and what is his/ her specific contribution to the invention?": third2_,
            "Please indicate the type of agreement, if any, which governs the involvement of any third-party collaborator for this invention and provide a copy, if available":"",
            "General Contract":third4,
            "Contract #":contract_number,
            "Other (e.g., masters and/or Doctoral thesis) (Explain the circumstances and details of the 3rd party Involvement). ":third5,
            "Was a Non-Disclosure Agreement or Confidential Disclosure Agreement signed with the company before discussing this invention?": third6 ,
            "Provide the Non-Disclosure or Confidential Disclosure Agreement Reference Number?" : third6_,
            "Is the inventor currently working with any of the following companies on developing this technology: (a) Saudi Aramco; or (b) a Saudi Aramco subsidiary?":super,
            "Identify the company and the relevant department within the company?":super_,
            "6. To Be Completed By Inventor’s Supervisor: ":"",
            "Is the invention being used by any of the following companies today: (a) Saudi Aramco; or (b) a Saudi Aramco subsidiary?":super2,
            "Identify the company and the relevant department within the company?":super2_,
            "What are the expected commercial applications for this invention?": super3,
            "Is this invention part of an on-going project? If so, please identify the project ":super4,
            "7. Execution By Witnesses: ":"",
            "I have read this invention disclosure (including attached pages, if any) and understand its subject matter.":wit,
            "Signature":signature,
            "Print Name": wit2,
            "Date":wit3}
            create(output_pdf, header_path, info_path, page_2_data)
    
            submit_button2 = st.form_submit_button(label= "submit")
            if submit_button2:
                with open(output_pdf, "rb") as file:
                    st.session_state.file_content = file.read()
                    st.session_state.form_submitted = True

        if st.session_state.get('form_submitted', False):
            st.download_button(
                label="Download PDF",
                data=st.session_state.file_content,
                file_name="filled_form.pdf",
                mime="application/pdf")
            st.success("PDF generated successfully")

if __name__ == "__main__":
    main()

st.markdown(
    """
    <style>
    .chat-container {
        display: flex;
        color: #000;
        border: 1px solid #ADD8E6;
        flex-direction: column;
        height: 40vh;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        background-color: #f9f9f9;
    }
    .message {
        margin-bottom: 10px;
        border: 1px solid #ADD8E6;
        color: #000;
    }
    .user-message {
        text-align: right;
        border: 1px solid #ADD8E6;
        color: #000;
    }
    .assistant-message {
        text-align: left;
        border: 1px solid #ADD8E6;
        color: #555;
    }
    .input-container {
        display: flex;
        border: 1px solid #ADD8E6;
        align-items: center;
        margin-top: 10px;
        color: #000;
    }
    .input-container textarea {
        flex: 1;
        color: #000;
        border: 1px solid #ADD8E6; /* Changed to light blue border */
        border-radius: 4px;
        padding: 10px;
        font-size: 16px;
        background-color: #f0f8ff; /* Light blue background */
    }
    .stButton > button {
        background-color: #ADD8E6; /* Light blue button */
        border: 1px solid #ADD8E6;
        color: black;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    .input-container button {
        background-color: #ADD8E6; /* Light blue button */
        border: 1px solid #ADD8E6;
        color: #000;
        border: none;
        padding: 10px 20px;
        margin-right: 10px;
        border-radius: 4px;
        cursor: pointer;
    }
    .input-container button:hover {
        background-color: #588BAE; /* Darker blue on hover */
        border: 1px solid #ADD8E6;
        color: black;
    }
    .stButton > button:hover {
        background-color: #588BAE; /* Darker blue on hover */
        border: 1px solid #ADD8E6;
        color: black;
    }
    .stRadio > div[data-baseweb="radio"] {
        border: 1px solid #ADD8E6; /* Light blue border */
        color: #000;
    }
    .stTextArea > div[data-baseweb="input"] {
        border: 1px solid #ADD8E6; /* Light blue border */
        color: #000;
    }
    .stTextInput > div[data-baseweb="input"] {
        border: 1px solid #ADD8E6; /* Light blue border */
        color: #000;
    }
    .stDateInput > div[data-baseweb="input"] {
        border: 1px solid #ADD8E6; /* Light blue border */
        color: #000;
    }
    </style>
    """,
    unsafe_allow_html=True)
