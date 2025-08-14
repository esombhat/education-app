# --- Standard Library Imports ---
import os
import json
import operator
import re
from typing import TypedDict, List, Optional, Dict
from datetime import datetime
import random
from pathlib import Path
from collections import Counter

# --- Third-party Imports ---
import streamlit as st
import pdfplumber
import plotly.graph_objects as go
import pandas as pd
import networkx as nx  # New import for the network graph

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Grammar Tutor",
    page_icon="✍️",
    layout="wide"
)

# --- Environment Setup & API Key ---
# IMPORTANT: For Streamlit deployment, set your API key as a secret.
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Please set it as an environment variable or Streamlit secret.")
    st.stop()

# ==============================================================================
# KNOWLEDGE GRAPH & BACKEND LOGIC
# ==============================================================================

KNOWLEDGE_GRAPH_DATA = {
    # Foundational Reading
    "FFR3a": {"name": "Use Knowledge of Syllabication and Syllable Types to Decode and Encode", "prerequisites": []},
    "FFR3b": {"name": "Use Knowledge of Morphology to Decode Words", "prerequisites": [{"grade": 4, "description": "FFR3a"}]},
    "FFR3c": {"name": "Read Grade-Level High-Frequency Words with Automaticity and Accuracy", "prerequisites": [{"grade": 4, "description": "FFR3b"}]},
    # Reading Comprehension
    "DSRa": {"name": "Read a Variety of Texts with Accuracy, Automaticity, etc.", "prerequisites": [{"grade": 4, "description": "FFR3c"}]},
    "DSRb": {"name": "Read and Comprehend a Variety of Literary and Informational Texts", "prerequisites": [{"grade": 4, "description": "DSRa"}]},
    "DSRc": {"name": "Draw Evidence from Texts to Support Claims, Conclusions, & Inferences", "prerequisites": [{"grade": 4, "description": "DSRb"}]},
    "DSRd": {"name": "Engage in Reading a Series of Texts to Build Knowledge and Vocabulary", "prerequisites": [{"grade": 4, "description": "DSRb"}]},
    "DSRe": {"name": "Use Reading Strategies to Aid and Monitor Comprehension", "prerequisites": [{"grade": 4, "description": "DSRb"}]},
    # Vocabulary
    "RV1a": {"name": "Develop General Academic and Content-Specific Vocabulary", "prerequisites": [{"grade": 4, "description": "DSRd"}]},
    "RV1b": {"name": "Discuss Meanings of Complex Words and Phrases", "prerequisites": [{"grade": 4, "description": "RV1a"}]},
    "RV1c": {"name": "Determine Meaning of Complex Words Using Roots & Inflectional Affixes", "prerequisites": [{"grade": 4, "description": "FFR3b"}]},
    "RV1d": {"name": "Use the Context of a Sentence to Apply Knowledge of Homophones", "prerequisites": []},
    "RV1e": {"name": "Apply Knowledge of Morphology, etc., to Determine Meaning of Words", "prerequisites": [{"grade": 4, "description": "FFR3b"}]},
    "RV1f": {"name": "Develop Vocabulary by Listening to & Reading High Quality Texts", "prerequisites": []},
    "RV1g": {"name": "Distinguish Shades of Meaning Among Verbs and Adjectives", "prerequisites": []},
    "RV1h": {"name": "Use Strategies to Infer Word Meanings", "prerequisites": []},
    "RV1i": {"name": "Use Reference Materials to Clarify Meaning of Words and Phrases", "prerequisites": []},
    "RV1j": {"name": "Use Newly Learned Words & Phrases in Discussions & Speaking Activities", "prerequisites": [{"grade": 4, "description": "RV1a"}]},
    # Writing Foundations
    "FFW1a": {"name": "Maintain Legible Printing", "prerequisites": []},
    "FFW1b": {"name": "Maintain Legible Cursive", "prerequisites": []},
    "FFW1c": {"name": "Sign His/Her First and Last Name", "prerequisites": [{"grade": 4, "description": "FFW1b"}]},
    "FFW2a": {"name": "Use Knowledge of Letter-Sound Combinations, etc. to Spell Accurately", "prerequisites": [{"grade": 4, "description": "FFR3a"}]},
    "FFW2b": {"name": "Use Phoneme/Grapheme Correspondences to Decode and Encode Words", "prerequisites": [{"grade": 4, "description": "FFW2a"}]},
    # Grammar and Mechanics
    "LU1a": {"name": "Produce, Expand, and Rearrange Simple and Compound Sentences", "prerequisites": []},
    "LU1b": {"name": "Use Conjunctions to Join Words and Phrases in a Sentence", "prerequisites": [{"grade": 4, "description": "LU1a"}]},
    "LU1c": {"name": "Use Adjectives to Compare and Describe Noun or Noun Phrases", "prerequisites": []},
    "LU1d": {"name": "Use Modal Words to Convey Various Conditions When Speaking and Writing", "prerequisites": []},
    "LU1e": {"name": "Use Standard Subject-Verb Agreement When Speaking and Writing", "prerequisites": [{"grade": 4, "description": "LU1a"}]},
    "LU1f": {"name": "Use Standard Noun-Pronoun Agreement When Speaking and Writing", "prerequisites": [{"grade": 4, "description": "LU1a"}]},
    "LU2a": {"name": "Use Commas in Series, Dates, Addresses, and Letters in Writing", "prerequisites": []},
    "LU2b": {"name": "Use Commas and Quotation Marks to Indicate Dialogue in Writing", "prerequisites": [{"grade": 4, "description": "LU2a"}]},
    "LU2c": {"name": "Use Apostrophes to Form Contractions and to Show Possession in Writing", "prerequisites": []},
    "LU2d": {"name": "Use Conventional Spelling for High-Frequency and Other Studied Words", "prerequisites": [{"grade": 4, "description": "FFW2a"}]},
    "LU2e": {"name": "Consult Reference Materials to Check and Correct Spelling", "prerequisites": [{"grade": 4, "description": "LU2d"}]},
    # Composition
    "W1a": {"name": "Recognize Different Forms of Writing Have Distinctive Organization", "prerequisites": []},
    "W1b": {"name": "Write Personal or Fictional Narratives That Are Logically Organized", "prerequisites": [{"grade": 4, "description": "W2a"}]},
    "W1c": {"name": "Write Expository Texts to Examine a Topic", "prerequisites": [{"grade": 4, "description": "W2a"}]},
    "W1d": {"name": "Write Persuasive Pieces on Topics or Texts", "prerequisites": [{"grade": 4, "description": "W2a"}]},
    "W1e": {"name": "Write in Response to Texts Read to Demonstrate Thinking", "prerequisites": [{"grade": 4, "description": "DSRc"}]},
    "W2a": {"name": "Engage in Writing as a Process to Compose Well-Developed Paragraphs", "prerequisites": [{"grade": 4, "description": "LU1a"}]},
    "W3a": {"name": "Revise Writing for Ideas, Organization, Sentence Fluency, & Wording", "prerequisites": [{"grade": 4, "description": "W2a"}]},
    "W3b": {"name": "Self- and Peer-Edit Writing for Capitalization, Spelling, etc.", "prerequisites": [{"grade": 4, "description": "LU2d"}, {"grade": 4, "description": "LU2a"}, {"grade": 4, "description": "LU2c"}] },
    # Communication and Research
    "C1a": {"name": "Participate in a Range of Collaborative Discussions", "prerequisites": []},
    "C2a": {"name": "Report Orally on a Topic, Tell a Story, or Recount an Experience", "prerequisites": []},
    "C3a": {"name": "Create Engaging Presentations That Include Multimedia Components", "prerequisites": []},
    "C3b": {"name": "Use Various Modes of Communication to Convey Messages/Develop Themes", "prerequisites": []},
    "C4a": {"name": "Differentiate Among Auditory, Visual, and Written Media Messages", "prerequisites": []},
    "C4b": {"name": "Compare and Contrast How Ideas and Topics are Depicted in Media", "prerequisites": []},
    "R1a": {"name": "Construct and Formulate Questions About a Topic", "prerequisites": []},
    "R1b": {"name": "Identify Search Terms to Locate and Gather Information", "prerequisites": [{"grade": 4, "description": "R1a"}]},
    "R1c": {"name": "Organize and Synthesize Information from Sources; Evaluate Relevance", "prerequisites": [{"grade": 4, "description": "R1b"}]},
    "R1d": {"name": "Develop Notes That Include Important Concepts and Summaries", "prerequisites": [{"grade": 4, "description": "R1c"}]},
    "R1e": {"name": "Organize and Share Information Orally, In Writing, or through Visuals", "prerequisites": [{"grade": 4, "description": "R1d"}]},
    "R1f": {"name": "Avoid Plagiarism and Give Proper Credit by Providing Citations", "prerequisites": [{"grade": 4, "description": "R1c"}]}
}

# --- Backend Functions (cached for performance) ---

@st.cache_data
def load_text_from_upload(uploaded_file):
    # ... (function remains the same)
    if uploaded_file is None: return ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

@st.cache_data
def analyze_writing_for_errors(_student_text: str, knowledge_graph: dict) -> list:
    # ... (function remains the same)
    standards_list_str = "\n".join([f'- {code}: {details["name"]}' for code, details in knowledge_graph.items()])
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a highly accurate grammar expert... Return a JSON array. Available Standards:\n{standards_list_str}"),
        ("human", "{student_text}")
    ])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, response_mime_type="application/json")
    chain = prompt | llm | JsonOutputParser()
    return chain.invoke({"student_text": _student_text})

@st.cache_data
def create_initial_profile(_all_standards: dict, _identified_errors: list, _student_text: str) -> dict:
    profile = {}
    num_sentences = _student_text.count('.') + _student_text.count('?') + _student_text.count('!') + 1
    error_counts = Counter([err.get("standard_code") for err in _identified_errors if err.get("standard_code")])
    
    # EDITED: Default mastery value changed to 0.5
    for std_id in _all_standards.keys():
        profile[std_id] = {"mastery_score": 0.50}
        
    for std_code, count in error_counts.items():
        if std_code in profile:
            error_density = count / num_sentences
            penalty = error_density * 4.0
            profile[std_code]["mastery_score"] = max(0.05, profile[std_code]["mastery_score"] - penalty)
    return profile

# --- Tutoring Session Functions ---
def get_next_question_topic(profile: dict) -> str:
    # ... (function remains the same)
    return min(profile, key=lambda cid: profile[cid]["mastery_score"])

def generate_practice_question(concept_id: str, concept_name: str) -> dict:
    # ... (function remains the same)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.8)
    prompt_text = f"..."
    try:
        response = llm.invoke(prompt_text)
        return json.loads(response.content.strip().replace("```json", "").replace("```", ""))
    except (json.JSONDecodeError, KeyError): return None

def evaluate_answer(problem: dict, user_correction: str) -> bool:
    # ... (function remains the same)
    eval_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    prompt = f"..."
    evaluation = eval_llm.invoke(prompt).content
    return "Correct" in evaluation

def update_mastery(profile: dict, concept_id: str, was_correct: bool) -> dict:
    # ... (function remains the same)
    old_mastery = profile[concept_id]["mastery_score"]
    if was_correct: profile[concept_id]["mastery_score"] += (1.0 - old_mastery) * 0.25
    else: profile[concept_id]["mastery_score"] -= old_mastery * 0.40
    profile[concept_id]["mastery_score"] = max(0.05, min(1.0, profile[concept_id]["mastery_score"]))
    return profile

# ==============================================================================
# UI RENDERING FUNCTIONS
# ==============================================================================

def render_teacher_view():
    st.header("Teacher Dashboard")
    st.markdown("Upload a student's writing sample (PDF) to generate a diagnostic report and create a personalized practice plan.")

    uploaded_file = st.file_uploader("Choose a student's PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Analyze Student Work"):
            with st.spinner("Reading and analyzing the document... This may take a moment."):
                student_text = load_text_from_upload(uploaded_file)
                if student_text:
                    st.session_state.identified_weaknesses = analyze_writing_for_errors(student_text, KNOWLEDGE_GRAPH_DATA)
                    st.session_state.student_profile = create_initial_profile(KNOWLEDGE_GRAPH_DATA, st.session_state.identified_weaknesses, student_text)
                    st.success("Analysis Complete! The student can now log in to their view.")
                else:
                    st.error("Could not extract text from the PDF.")

    if 'identified_weaknesses' in st.session_state:
        st.subheader("Diagnostic Report")
        report = st.session_state.identified_weaknesses
        if not report:
            st.info("No specific grammatical errors were found in the document. Great work!")
        else:
            for i, error in enumerate(report):
                with st.expander(f"Weakness #{i+1}: {error.get('standard_code', 'N/A')}"):
                    st.markdown(f"**Standard:** `{error.get('standard_code', 'N/A')}` - {KNOWLEDGE_GRAPH_DATA.get(error.get('standard_code', ''), {}).get('name', 'Unknown')}")
                    st.markdown(f"**Original Sentence:**")
                    st.warning(error.get('original_sentence', 'N/A'))
                    st.markdown(f"**Suggested Correction:**")
                    st.success(error.get('corrected_sentence', 'N/A'))

# NEW: Function to create an interconnected network graph
def create_mastery_network_graph(profile: dict, knowledge_graph: dict):
    G = nx.DiGraph()
    node_colors = []
    node_sizes = []
    
    # Add nodes with attributes
    for code, details in profile.items():
        G.add_node(code, name=knowledge_graph[code]['name'])
        score = details['mastery_score']
        # Color based on mastery
        if score < 0.5: color = '#d9534f' # Red
        elif score < 0.8: color = '#f0ad4e' # Yellow
        else: color = '#5cb85c' # Green
        node_colors.append(color)
        # Size based on mastery
        node_sizes.append(15 + score * 20)

    # Add edges from prerequisites
    for code, details in knowledge_graph.items():
        for prereq in details.get('prerequisites', []):
            prereq_code = re.match(r'([A-Z0-9a-z]+)', prereq['description']).group(1)
            if G.has_node(prereq_code) and G.has_node(code):
                G.add_edge(prereq_code, code)

    # Get positions for the nodes
    pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)

    # Create Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create Nodes
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        hover_text = f"<b>{node}</b>: {G.nodes[node]['name']}<br>Mastery: {profile[node]['mastery_score']:.2f}"
        node_text.append(hover_text)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line_width=2))
    
    node_trace.text = node_text

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Student Mastery Network',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Each circle is a skill. Lines show prerequisite relationships.",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def render_student_view():
    st.header("Student Practice Center")

    if 'student_profile' not in st.session_state:
        st.warning("Your profile is not yet available. Please ask your teacher to analyze your writing sample first.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Past Work Feedback", "My Mastery Graph", "Practice Session"])

    with tab1:
        # ... (tab remains the same)
        st.subheader("Feedback on Your Recent Writing")
        report = st.session_state.identified_weaknesses
        if not report:
            st.balloons()
            st.success("Great news! No specific errors were found in your last writing sample.")
        else:
            for i, error in enumerate(report):
                with st.container(border=True):
                    st.markdown(f"**Area to Improve #{i+1}:** `{error.get('standard_code', 'N/A')}`")
                    st.markdown(f"**Original:** *{error.get('original_sentence', 'N/A')}*")
                    st.markdown(f"**Suggestion:** **{error.get('corrected_sentence', 'N/A')}**")

    with tab2:
        st.subheader("Your Progress")
        # EDITED: Call the new network graph function
        fig = create_mastery_network_graph(st.session_state.student_profile, KNOWLEDGE_GRAPH_DATA)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Legend:**
        - <span style="color: #d9534f;">Red</span>: Needs Practice
        - <span style="color: #f0ad4e;">Yellow</span>: Getting There
        - <span style="color: #5cb85c;">Green</span>: Proficient
        """, unsafe_allow_html=True)


    with tab3:
        st.subheader("Let's Practice!")
        
        if 'session_question_count' not in st.session_state:
            st.session_state.session_question_count = 0

        if st.session_state.session_question_count >= 5:
            st.success("You've completed your 5 practice questions for this session! Great work.")
            st.balloons()
            if st.button("Start a New Session"):
                st.session_state.session_question_count = 0
                st.rerun()
        else:
            st.progress((st.session_state.session_question_count) / 5, text=f"Question {st.session_state.session_question_count + 1} of 5")
            
            if 'current_question' not in st.session_state or st.session_state.current_question is None:
                with st.spinner("Generating your next question..."):
                    topic_id = get_next_question_topic(st.session_state.student_profile)
                    topic_name = KNOWLEDGE_GRAPH_DATA[topic_id]['name']
                    st.session_state.current_question = generate_practice_question(topic_id, topic_name)
            
            problem = st.session_state.current_question

            if problem:
                with st.form(key='practice_form'):
                    st.markdown(f"**{problem['instruction_text']}**")
                    st.info(f"'{problem['problem_sentence']}'")
                    user_answer = st.text_input("Type your corrected sentence here:")
                    submitted = st.form_submit_button("Check My Answer")

                    if submitted:
                        was_correct = evaluate_answer(problem, user_answer)
                        # EDITED: Always show feedback, just with a different tone.
                        if was_correct:
                            st.success("That's correct! Nicely done.")
                        else:
                            st.error(f"Not quite. Let's take a look.")
                        
                        st.info(f"The corrected sentence is: **'{problem['correct_sentence']}'**")
                        
                        # Use a temporary state to hold the feedback before rerunning
                        st.session_state.last_result_processed = True
                        
                        # Update profile and move to the next question
                        current_topic = get_next_question_topic(st.session_state.student_profile)
                        st.session_state.student_profile = update_mastery(st.session_state.student_profile, current_topic, was_correct)
                        st.session_state.session_question_count += 1
                        st.session_state.current_question = None
                        
                        # Add a small delay and a button to continue
                        st.button("Next Question")
                        st.stop() # Stop the script here until the user clicks the button
            else:
                st.error("Could not generate a practice question. Please try again.")

# ==============================================================================
# MAIN APP ROUTER
# ==============================================================================

st.title("✍️ AI Grammar Analysis & Tutoring")

with st.sidebar:
    st.header("Navigation")
    view_choice = st.radio(
        "Choose Your View",
        ["Teacher View", "Student View"],
        key="view_selection"
    )

if view_choice == "Teacher View":
    render_teacher_view()
else:
    render_student_view()