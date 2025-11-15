import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import unicodedata
import joblib
from pathlib import Path

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from textstat import flesch_kincaid_grade, syllable_count
import textstat
import inflect

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
words_to_keep = {"how", "what", "where", "why", "when"}
stop_words = stop_words - words_to_keep
lemmatizer = WordNetLemmatizer()

# Page configuration
st.set_page_config(
    page_title="Bloom's Taxonomy Cognitive Classifier",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .dual-label {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.5rem;
        display: inline-block;
    }
    .info-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .taxonomy-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    </style>
""", unsafe_allow_html=True)

# Unified difficulty-to-Bloom's mapping
COGNITIVE_LEVELS = {
    'Very Easy': {
        'bloom_level': 'Remember',
        'color': '#28a745',
        'description': 'Recall facts and basic concepts',
        'verbs': 'Define, List, Name, Recall, Identify',
        'difficulty_range': '(-5.0, -3.0)'
    },
    'Easy': {
        'bloom_level': 'Understand',
        'color': '#17a2b8',
        'description': 'Explain ideas or concepts',
        'verbs': 'Describe, Explain, Summarize, Interpret',
        'difficulty_range': '(-2.9, -1.0)'
    },
    'Average': {
        'bloom_level': 'Apply',
        'color': '#ffc107',
        'description': 'Use information in new situations',
        'verbs': 'Apply, Calculate, Solve, Demonstrate',
        'difficulty_range': '(-0.9, 1.0)'
    },
    'Hard': {
        'bloom_level': 'Analyze',
        'color': '#fd7e14',
        'description': 'Draw connections among ideas',
        'verbs': 'Analyze, Compare, Contrast, Examine',
        'difficulty_range': '(1.1, 3.0)'
    },
    'Very Hard': {
        'bloom_level': 'Evaluate',
        'color': '#dc3545',
        'description': 'Justify a stand or decision',
        'verbs': 'Evaluate, Assess, Critique, Judge',
        'difficulty_range': '(3.1, 5.0)'
    }
}

EXAMPLE_QUESTIONS = {
    'Very Easy': "What is the capital of France?",
    'Easy': "Explain the process of photosynthesis in your own words.",
    'Average': "Calculate the compound interest on $1000 at 5% for 3 years.",
    'Hard': "Analyze the key differences between democracy and authoritarianism.",
    'Very Hard': "Evaluate the ethical implications of AI in healthcare decision-making."
}

@st.cache_resource
def load_models():
    """Load the trained models and preprocessors"""
    try:
        model = joblib.load('trained_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, vectorizer, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("Please ensure these files are in the same directory: trained_model.pkl, tfidf_vectorizer.pkl, scaler.pkl")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

def preprocess_text(text):
    """Preprocess text with code snippet handling"""
    code_snippets = re.findall(r"`.*?`", text)
    text_without_code = re.sub(r"`.*?`", "", text)
    text_without_code = re.sub(r"[^a-zA-Z0-9\s]", "", text_without_code.lower())
    tokens = word_tokenize(text_without_code)
    
    filtered_words = [word for word in tokens if word not in stop_words]
    preprocessed_tokens = [lemmatizer.lemmatize(word) for word in filtered_words if word.isalpha()]
    
    preprocessed_text = " ".join(preprocessed_tokens)
    code_text = " ".join(code_snippets)
    combined_text = f"{preprocessed_text} {code_text}".strip()
    
    return preprocessed_tokens, combined_text

def extract_features(question, tfidf_vectorizer):
    """Extract comprehensive features from question text"""
    tokenized_question, preprocessed_question = preprocess_text(question)
    
    features = {}
    
    # TF-IDF features
    try:
        tfidf_vector = tfidf_vectorizer.transform([preprocessed_question]).toarray()[0]
        features.update({f"tfidf_{i}": val for i, val in enumerate(tfidf_vector)})
    except Exception as e:
        st.error(f"TF-IDF transformation error: {e}")
        return None
    
    # Text length features
    features["word_count"] = len(tokenized_question)
    features["sentence_count"] = len(sent_tokenize(question))
    features["avg_word_length"] = np.mean([len(word) for word in tokenized_question]) if tokenized_question else 0
    features["readability"] = flesch_kincaid_grade(question) if question.strip() else 0
    
    # Lexical complexity
    unique_words = len(set(tokenized_question))
    features["unique_word_count"] = unique_words
    features["vocabulary_diversity"] = unique_words / len(tokenized_question) if len(tokenized_question) > 0 else 0
    features["complex_word_count"] = sum(syllable_count(word) > 3 for word in tokenized_question)
    
    return features

def match_difficulty(prediction_index):
    """Map prediction index to difficulty level"""
    difficulty_mapping = {
        0: "Very Easy",
        1: "Easy",
        2: "Average",
        3: "Hard",
        4: "Very Hard"
    }
    return difficulty_mapping.get(prediction_index, "Unknown")

def predict_cognitive_level(question, model, vectorizer, scaler):
    """Complete prediction pipeline"""
    features = extract_features(question, vectorizer)
    if features is None:
        return None, None, None
    
    feature_df = pd.DataFrame([features])
    
    try:
        normalized_features = scaler.transform(feature_df)
        probabilities = model.predict_proba(normalized_features)[0]
        prediction_index = np.argmax(probabilities)
        difficulty = match_difficulty(prediction_index)
        return difficulty, probabilities, features
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

def create_dual_chart(probabilities):
    """Create chart showing both difficulty and Bloom's levels"""
    difficulties = list(COGNITIVE_LEVELS.keys())
    blooms = [COGNITIVE_LEVELS[d]['bloom_level'] for d in difficulties]
    colors = [COGNITIVE_LEVELS[d]['color'] for d in difficulties]
    
    labels = [f"{b}<br>({d})" for b, d in zip(blooms, difficulties)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probabilities,
            marker=dict(color=colors),
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Cognitive Level Classification",
        xaxis_title="Bloom's Taxonomy Level (Difficulty)",
        yaxis_title="Probability",
        yaxis=dict(tickformat='.0%'),
        height=450,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.markdown('<p class="main-header">🎓 Bloom\'s Taxonomy Cognitive Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML-powered cognitive level assessment using difficulty-based classification</p>', unsafe_allow_html=True)
    
    # Load models
    model, vectorizer, scaler = load_models()
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🔬 Model Architecture")
        
        st.markdown("""
        ### Classification Approach
        This model predicts cognitive complexity through difficulty assessment, mapping difficulty levels to Bloom's Taxonomy:
        
        **Difficulty → Bloom's Level**
        - Very Easy → Remember
        - Easy → Understand  
        - Average → Apply
        - Hard → Analyze
        - Very Hard → Evaluate
        
        ### Feature Engineering
        - **TF-IDF Vectorization**: Term frequency analysis
        - **Readability Metrics**: Flesch-Kincaid grade level
        - **Lexical Complexity**: Syllable count, vocabulary diversity
        - **Structural Analysis**: Sentence/word patterns
        
        ### Algorithm
        - Random Forest Classifier
        - Standard feature scaling
        - Multi-class probability output
        """)
        
        st.markdown("---")
        st.markdown("### 📚 Bloom's Taxonomy")
        st.caption("Note: 'Create' level not included in this classification model")
        
        for diff, info in COGNITIVE_LEVELS.items():
            st.markdown(f"**{info['bloom_level']}** ({diff})")
            st.caption(f"{info['description']}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Classify Question Cognitive Level")
        
        question = st.text_area(
            "Enter an educational question:",
            height=150,
            placeholder="Type your question here...",
            help="The model analyzes linguistic complexity to determine both difficulty and Bloom's taxonomy level"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        if classify_btn and question.strip():
            with st.spinner("Analyzing cognitive complexity..."):
                difficulty, probabilities, features = predict_cognitive_level(question, model, vectorizer, scaler)
                
                if difficulty:
                    bloom_level = COGNITIVE_LEVELS[difficulty]['bloom_level']
                    confidence = max(probabilities)
                    color = COGNITIVE_LEVELS[difficulty]['color']
                    
                    # Main prediction display
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="margin:0;">🎓 {bloom_level}</h2>
                        <div class="dual-label">
                            <strong>Difficulty Level:</strong> {difficulty}
                        </div>
                        <div class="dual-label">
                            <strong>Confidence:</strong> {confidence:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Description card
                    st.markdown(f"""
                    <div class="taxonomy-card" style="border-left-color: {color};">
                        <strong>{bloom_level}</strong>: {COGNITIVE_LEVELS[difficulty]['description']}<br>
                        <em>Key verbs: {COGNITIVE_LEVELS[difficulty]['verbs']}</em>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Dual visualization
                    st.plotly_chart(create_dual_chart(probabilities), use_container_width=True)
                    
                    # Metrics row
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("Word Count", int(features.get('word_count', 0)))
                    with col_m2:
                        st.metric("Reading Grade", f"{features.get('readability', 0):.1f}")
                    with col_m3:
                        st.metric("Vocab Diversity", f"{features.get('vocabulary_diversity', 0):.1%}")
                    with col_m4:
                        st.metric("Complex Words", int(features.get('complex_word_count', 0)))
                    
                    # Detailed analysis
                    with st.expander("📊 View Detailed Classification Breakdown"):
                        # Probability table with both labels
                        prob_data = []
                        for i, (diff, prob) in enumerate(zip(COGNITIVE_LEVELS.keys(), probabilities)):
                            bloom = COGNITIVE_LEVELS[diff]['bloom_level']
                            prob_data.append({
                                "Bloom's Level": bloom,
                                "Difficulty": diff,
                                "Probability": prob,
                                "Percentage": f"{prob:.2%}"
                            })
                        
                        prob_df = pd.DataFrame(prob_data).sort_values('Probability', ascending=False)
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                        
                        st.markdown("### 🔍 Feature Analysis")
                        feat_col1, feat_col2 = st.columns(2)
                        with feat_col1:
                            st.markdown(f"""
                            **Structural Features:**
                            - Sentences: {features.get('sentence_count', 0)}
                            - Average Word Length: {features.get('avg_word_length', 0):.2f}
                            - Unique Words: {features.get('unique_word_count', 0)}
                            """)
                        with feat_col2:
                            st.markdown(f"""
                            **Complexity Indicators:**
                            - Complex Words (>3 syllables): {features.get('complex_word_count', 0)}
                            - Flesch-Kincaid Grade: {features.get('readability', 0):.1f}
                            - TF-IDF Features: {sum(1 for k in features if k.startswith('tfidf_'))}
                            """)
        
        elif classify_btn:
            st.warning("⚠️ Please enter a question to classify.")
    
    with col2:
        st.subheader("💡 Example Questions")
        st.caption("Click to try different cognitive levels")
        
        for diff, example in EXAMPLE_QUESTIONS.items():
            bloom = COGNITIVE_LEVELS[diff]['bloom_level']
            color = COGNITIVE_LEVELS[diff]['color']
            
            if st.button(
                f"**{bloom}**\n*({diff})*\n\n{example[:60]}...",
                key=f"ex_{diff}",
                use_container_width=True,
                help=example
            ):
                st.session_state['selected_example'] = example
                st.rerun()
    
    if 'selected_example' in st.session_state:
        st.session_state.pop('selected_example')
    
    # Educational context
    st.markdown("---")
    st.markdown("### 📖 Understanding the Difficulty-Bloom's Mapping")
    
    st.info("""
    This classifier uses **question difficulty as a proxy for cognitive complexity**, based on the principle that questions requiring 
    higher-order thinking skills naturally present greater linguistic and conceptual difficulty.
    
    The model analyzes readability, vocabulary sophistication, and structural complexity to infer the cognitive level required to answer the question.
    """)
    
    info_cols = st.columns(3)
    with info_cols[0]:
        st.markdown("""
        <div class="info-section">
        <h4>Lower-Order Thinking</h4>
        <strong>Remember & Understand</strong><br>
        Basic recall and comprehension. Questions use simple vocabulary and direct phrasing.
        </div>
        """, unsafe_allow_html=True)
    
    with info_cols[1]:
        st.markdown("""
        <div class="info-section">
        <h4>Middle-Order Thinking</h4>
        <strong>Apply</strong><br>
        Practical application requiring moderate complexity and problem-solving context.
        </div>
        """, unsafe_allow_html=True)
    
    with info_cols[2]:
        st.markdown("""
        <div class="info-section">
        <h4>Higher-Order Thinking</h4>
        <strong>Analyze & Evaluate</strong><br>
        Critical thinking with advanced vocabulary, complex syntax, and multi-layered reasoning.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()