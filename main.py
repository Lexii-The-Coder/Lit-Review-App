import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import re
from openai import OpenAI
from difflib import SequenceMatcher
from pyvis.network import Network
import networkx as nx
import streamlit.components.v1 as components

"""
Literature Review Dashboard - Python/Streamlit Version
Customizable app for quick and easy literature reviews across research domains
"""


# Page configuration
st.set_page_config(
    page_title="Literature Review Dashboard",
    page_icon="📚",
    layout="wide"
)

# Initialize session state for storing papers
if 'papers' not in st.session_state:
    st.session_state.papers = []

if 'custom_prompt' not in st.session_state:
    # Default customizable prompt template
    st.session_state.custom_prompt = """Analyze this research paper and provide:
1. Research Type (choose one from: 'Dissertation', 'Study', 'Policy Paper', 'Review', 'Technical Report', 'Conference Paper','Proposal' , 'Other')
2. Author(s)
3. Title
4. Overview/Top Level Summary (one paragraph, about five sentences)
5. Primary research area (choose one of: 'Biology', 'Environmental Science', 'Hydrology','Atmospheric Science','Computer Networks')
6. Main research question(s)
7. Key findings/contributions
8. Methodology Used
9. Relevance score (0-100) based on novelty and impact
10. 3-5 keywords
11. Publication Date
12. Publcation Venue (ie Journal or Conference Name)
13. DOI (if applicable)
14. Publication Year

Format as JSON with keys: research_type, authors, title, summary, research_area, questions, findings, methodology, relevance_score, keywords, publication_venue, doi, year"""


def analyze_text_stats(text):
    """Extract basic text statistics from paper"""
    word_count = len(text.split())
    
    patterns = {
        'year_parens': r'\(\d{4}\)',
        'et_al': r'et al\.',
        'brackets': r'\[\d+\]',
        'author_year': r'[A-Z][a-z]+ et al\.'
    }
    references = sum(len(re.findall(pattern, text)) for pattern in patterns.values())
    
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    reading_time = round(word_count / 225, 1)
    
    return {
        'word_count': word_count,
        'references': references,
        'paragraphs': paragraphs,
        'reading_time': reading_time
    }


def process_with_openai(text, api_key):
    """Process paper using OpenAI API with customizable prompt"""
    
    # Get text stats first (these always work)
    stats = analyze_text_stats(text)
    text_sample = text[:4000]
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Make API call
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an advanced (~PhD level) academic research assistant. Respond in valid JSON format only."
                },
                {
                    "role": "user", 
                    "content": f"{st.session_state.custom_prompt}\n\nPaper text:\n{text_sample}"
                }
            ]
        )
        
        # Extract response
        ai_response = response.choices[0].message.content
        
        # Clean up markdown formatting if present
        if '```json' in ai_response:
            ai_response = ai_response.split('```json')[1].split('```')[0].strip()
        elif '```' in ai_response:
            ai_response = ai_response.split('```')[1].split('```')[0].strip()
        
        # Parse JSON
        parsed = json.loads(ai_response)
        
        # Ensure all required fields exist
        result = {
            'research_area': parsed.get('research_area', 'Unknown'),
            'research_type': parsed.get('research_type','Unknown'),
            'authors': parsed.get('authors','N/A'),
            'title': parsed.get('title','N/A'),
            'summary': parsed.get('summary','N/A'),
            'questions': parsed.get('questions', 'N/A'),
            'findings': parsed.get('findings', 'N/A'),
            'methodology': parsed.get('methodology', 'N/A'),
            'relevance_score': int(parsed.get('relevance_score', 50)),
            'keywords': parsed.get('keywords', []),
            'publication_venue': parsed.get('publication_venue', []),
            'doi':parsed.get('doi','N/A'),
            **stats,
            'timestamp': datetime.now().isoformat(),
            'preview': text[:200] + "...",
            'year': parsed.get('year',0)
        }
        return result
       # Format as JSON with keys:   publication_venue, doi"""

    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {str(e)}")
        st.error(f"AI Response was: {ai_response[:500]}")
        raise
        
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        raise


# SIDEBAR: Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    
    st.divider()
    
    # Import JSON
    st.subheader("📤 Import Previous Review")
    uploaded_file = st.file_uploader("Upload JSON file", type=['json'])
    
    if uploaded_file is not None:
        try:
            json_data = json.load(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Merge"):
                    st.session_state.papers.extend(json_data)
                    st.success(f"Added {len(json_data)} papers!")
                    st.rerun()
            with col2:
                if st.button("Replace"):
                    st.session_state.papers = json_data
                    st.success(f"Loaded {len(json_data)} papers!")
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error loading JSON: {str(e)}")
    
    st.divider()
    
    # Custom prompt editor
    with st.expander("🔧 Customize Analysis Prompt"):
        st.write("Edit the prompt to tailor analysis to your research domain:")
        custom_prompt = st.text_area(
            "Analysis Prompt",
            value=st.session_state.custom_prompt,
            height=200,
            help="The AI will use this prompt to analyze papers. Use JSON format for structured output."
        )
        if st.button("Save Prompt"):
            st.session_state.custom_prompt = custom_prompt
            st.success("Prompt saved!")
    
    st.divider()
    
    # Export options
    st.subheader("📥 Export Data")
    if st.session_state.papers:
        df = pd.DataFrame(st.session_state.papers)
        
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"lit_review_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
        
        json_str = json.dumps(st.session_state.papers, indent=2)
        st.download_button(
            "Download JSON",
            json_str,
            f"lit_review_{datetime.now().strftime('%Y%m%d')}.json",
            "application/json"
        )
    else:
        st.info("No papers to export yet")
    
    st.divider()
    
    # Clear data
    if st.button("🗑️ Clear All Papers"):
        st.session_state.papers = []
        st.rerun()


def get_author_key(name):
    """Pull out (first_initial, last_name) as the matching fingerprint"""
    parts = name.strip().split()
    if not parts:
        return ('', '')
    last = parts[-1].lower().rstrip('.,')
    first_initial = parts[0][0].lower()
    return (first_initial, last)

def cluster_authors(authors):
    """Returns a dict of {variant_name: canonical_name}
    Canonical = the longest/most complete version found"""
    clusters = []  # each entry: [key, canonical_name, [all variants]]

    for author in authors:
        key = get_author_key(author)
        matched = False
        for cluster in clusters:
            same_initial = key[0] == cluster[0][0]
            last_similar = SequenceMatcher(None, key[1], cluster[0][1]).ratio() > 0.8
            if same_initial and last_similar:
                cluster[2].append(author)
                if len(author) > len(cluster[1]):  # keep the most complete name
                    cluster[1] = author
                matched = True
                break
        if not matched:
            clusters.append([key, author, [author]])

    return {variant: canonical for _, canonical, variants in clusters for variant in variants}


# MAIN CONTENT
st.title("📚 Literature Review Dashboard")
st.markdown("Customizable tool for quick and easy literature reviews")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Add Paper", "📊 Overview", "📖 Paper Library", "Author Network"])

# TAB 1: Add Paper
with tab1:
    st.header("Add New Paper")
    
    paper_text = st.text_area(
        "Paste paper text here",
        height=300,
        placeholder="Paste the full text of your research paper here..."
    )
    
    process_button = st.button("🔍 Process Paper", type="primary")
    
    if process_button:
        if not paper_text:
            st.error("Please paste paper text first")
        elif not api_key:
            st.error("Please enter OpenAI API key in sidebar")
        else:
            try:
                with st.spinner("Processing paper with AI..."):
                    result = process_with_openai(paper_text, api_key)
                    st.session_state.papers.append(result)
                    st.success("✅ Paper processed successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

# TAB 2: Overview
with tab2:
    

    if not st.session_state.papers:
        st.info("No papers processed yet. Add a paper in the 'Add Paper' tab.")
    else:
        df = pd.DataFrame(st.session_state.papers)
        
        all_authors = sorted(set(author.strip()
        for entry in df['authors']
        for author in (entry if isinstance(entry, list) else [entry])))

        all_authors_flat = sorted(set(author.strip()
        for entry in df['authors']
        for author in (entry if isinstance(entry, list) else [entry])))
        author_map = cluster_authors(all_authors_flat) 
        canonical_authors = sorted(set(author_map.values()))
        
        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Papers", len(df))
        with col2:
            st.metric("Research Areas", df['research_area'].nunique())
        with col3:
            st.metric("Avg Relevance", f"{df['relevance_score'].mean():.1f}")
        with col4:
            st.metric("Total Words", f"{df['word_count'].sum():,}")
        with col5:
            st.metric("Total Unique Authors", f"{len(canonical_authors):,}")
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            area_counts = df['research_area'].value_counts()
            fig1 = px.bar(
                x=area_counts.index,
                y=area_counts.values,
                labels={'x': 'Research Area', 'y': 'Number of Papers'},
                title='Papers by Research Area'
            )
            st.plotly_chart(fig1)
        
        with col2:
            fig2 = px.histogram(
                df,
                x='relevance_score',
                nbins=10,
                labels={'relevance_score': 'Relevance Score'},
                title='Relevance Score Distribution'
            )
            st.plotly_chart(fig2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = px.box(
                df,
                y='word_count',
                title='Word Count Distribution'
            )
            st.plotly_chart(fig3)
        
        with col2:
            fig4 = px.scatter(
                df,
                x='references',
                y='relevance_score',
                size='word_count',
                hover_data=['research_area','authors'],
                labels={'references': 'Number of References', 'relevance_score': 'Relevance Score'},
                title='References vs Relevance'
            )
            st.plotly_chart(fig4)

# TAB 3: Paper Library
with tab3:

    if not st.session_state.papers:
        st.info("No papers in library yet. Add papers in the 'Add Paper' tab.")
    else:
        st.header("Paper Library")
        
        df = pd.DataFrame(st.session_state.papers)
        
        all_authors_flat = sorted(set(author.strip()
        for entry in df['authors']
        for author in (entry if isinstance(entry, list) else [entry])))
        author_map = cluster_authors(all_authors_flat)
        canonical_authors = sorted(set(author_map.values()))

        if 'year' not in df.columns:
            df['year'] = 0
            
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            areas = ['All'] + list(df['research_area'].unique())
            selected_area = st.selectbox("Filter by Research Area", areas)
        
        with col2:
            authors = ['All'] + canonical_authors
            selected_author = st.selectbox("Filter by Author", authors)

        with col3:
            type = ['All'] + list(df['research_type'].unique())
            selected_type = st.selectbox("Filter by Publication Type",type)

        with col4:
            sort_by = st.selectbox(
                "Sort by",
                ['timestamp', 'relevance_score', 'word_count', 'references','year']
            )

        filtered_df = df if selected_area == 'All' else df[df['research_area'] == selected_area]
        filtered_df = filtered_df.sort_values(sort_by, ascending=False, na_position='last')
        if selected_author != 'All':
            filtered_df = filtered_df[filtered_df['authors'].apply(
                lambda a: selected_author in [
                    author_map.get(n.strip(), n.strip())
                    for n in (a if isinstance(a, list) else [a])])]
        if selected_type != 'All':
            filtered_df = filtered_df[filtered_df['research_type'] == selected_type]
            

        for idx, paper in filtered_df.iterrows():
            with st.expander(f"📄 {list(paper.get('authors', 'Unknown'))[0]} - {paper.get('research_type', 0)}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Title:** {paper.get('title','N/A')}")
                    st.markdown(f"**Author(s):** {paper.get('authors','N/A')}")
                    st.markdown(f"**Paper Summary:** {paper.get('summary','N/A')}")
                    st.markdown(f"**Research Area:** {paper.get('research_area', 'N/A')}")
                    st.markdown(f"**Research Questions:** {paper.get('questions', 'N/A')}")
                    st.markdown(f"**Key Findings:** {paper.get('findings', 'N/A')}")
                    st.markdown(f"**Methodology:** {paper.get('methodology', 'N/A')}")
                    st.markdown(f"**Publication Venue** {paper.get('publication_venue','N/A')}")
                    st.markdown(f"**DOI**:{paper.get('doi','N/A')}")
                    keywords = paper.get('keywords', [])
                    if isinstance(keywords, list):
                        keywords_str = ', '.join(keywords)
                    else:
                        keywords_str = str(keywords)
                    st.markdown(f"**Keywords:** {keywords_str}")
                
                with col2:
                    st.metric("Year",f"{paper.get('year',0)}")
                    st.metric("Relevance Score", f"{paper.get('relevance_score', 0):.0f}/100")
                    st.metric("Word Count", f"{paper.get('word_count', 0):,}")
                    st.metric("References", paper.get('references', 0))
                    st.metric("Reading Time", f"{paper.get('reading_time', 0)} min")

                st.markdown("**Preview:**")
                st.text(paper.get('preview', 'No preview available'))

with tab4:
    if not st.session_state.papers:
        st.info("No papers yet.")
    else:
        st.header("Author Collaboration Network")

        col1, col2 = st.columns(2)
        with col1:
            min_papers = st.slider("Min papers per author", 1, 10, 1)
        with col2:
            physics_on = st.toggle("Physics simulation", value=True)

        net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white")
        net.toggle_physics(physics_on)

        # Count papers per author first (needed for filtering + node sizing)
        author_paper_counts = {}
        for paper in st.session_state.papers:
            authors = paper.get('authors', [])
            if isinstance(authors, str):
                authors = [authors]
            for a in [author_map.get(a.strip(), a.strip()) for a in authors]:
                author_paper_counts[a] = author_paper_counts.get(a, 0) + 1

        
                # Collect lead authors across all papers
        lead_authors = set()
        for paper in st.session_state.papers:
            authors = paper.get('authors', [])
            if isinstance(authors, str):
                authors = [authors]
            authors = [author_map.get(a.strip(), a.strip()) for a in authors]
            if authors:
                lead_authors.add(authors[0])

        # Author nodes — gold for leads, blue for co-authors
        for author, count in author_paper_counts.items():
            if count >= min_papers:
                is_lead = author in lead_authors
                color = "#de0ff1bb" if is_lead else "#022751"
                label = f"{author}" if is_lead else author
                net.add_node(author, label=label, size=10 + count * 5,
                            color=color, title=f"{author} — {count} paper(s)")
        # # Author nodes
        # for author, count in author_paper_counts.items():
        #     if count >= min_papers:
        #         net.add_node(author, label=author, size=10 + count * 5,
        #                      color="#4e9af1", title=f"{author} — {count} paper(s)")

        # Paper hub nodes + edges to authors
        for paper in st.session_state.papers:
            authors = paper.get('authors', [])
            if isinstance(authors, str):
                authors = [authors]
            authors = [author_map.get(a.strip(), a.strip()) for a in authors]
            visible_authors = [a for a in authors if author_paper_counts.get(a, 0) >= min_papers]
            if len(visible_authors) < 2:  # solo papers don't generate edges
                continue

            lead = visible_authors[0]
            for co_author in visible_authors[1:]:
                net.add_edge(lead, co_author, color="#4e9af1", width=1.5,
                            title=paper.get('title', ''))

        components.html(net.generate_html(), height=620, scrolling=False)

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Authors", len(net.nodes))
        col2.metric("Papers",  len(st.session_state.papers))
        col3.metric("Connections", len(net.edges))

st.divider()
st.markdown("*(c) Alex Weirauch*")