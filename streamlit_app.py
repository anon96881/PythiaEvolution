import streamlit as st
import json
import os
import re
from typing import List, Dict, Any

# Set page config
st.set_page_config(
    page_title="Neuron Evolution - Pythia Models",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light mode and hide GitHub repo link
st.markdown("""
<style>
    .stApp {
        background-color: white;
    }
    .stApp > header {
        background-color: white;
    }
    .stApp > main {
        background-color: white;
    }
    .stSidebar {
        background-color: #f8f9fa;
    }
    .stSidebar > div {
        background-color: #f8f9fa;
    }
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
<style>
.token {
    display: inline-block;
    padding: 2px 4px;
    margin: 1px;
    border-radius: 3px;
    transition: all 0.2s ease;
}

.activation-low {
    background-color: rgba(255, 107, 107, 0.1);
    border: 1px solid rgba(255, 107, 107, 0.3);
}

.activation-high {
    background-color: rgba(255, 107, 107, 0.8);
    border: 1px solid rgba(255, 107, 107, 1.0);
    font-weight: bold;
}

.cluster {
    margin-bottom: 15px;
    border: 2px solid;
    border-radius: 8px;
    overflow: hidden;
}

.cluster-header {
    padding: 8px 12px;
    font-weight: bold;
    color: white;
    text-align: center;
}

.example {
    padding: 8px;
    background: white;
    border-bottom: 1px solid #e9ecef;
    line-height: 1.4;
}

.stSelectbox > div > div {
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_neuron_data(model_name):
    """Load all neuron data from the results directories for a specific model"""
    all_neuron_data = {}
    available_neurons = []
    
    # Define paths to search based on model
    if model_name == "Pythia-70M":
        search_paths = [
            "results",
            "results/pythia70m"
        ]
    elif model_name == "Pythia-160M":
        search_paths = [
            "results/pythia160m"
        ]
    else:
        return {}, []
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
        
        files_found = 0
            
        for filename in os.listdir(search_path):
            # Check for the appropriate file pattern based on model
            if model_name == "Pythia-70M" and filename.endswith("_pythia70m_ckpt_series.jsonl"):
                neuron_id = filename.replace("_pythia70m_ckpt_series.jsonl", "")
            elif model_name == "Pythia-160M" and filename.endswith("_pythia160m_ckpt_series.jsonl"):
                neuron_id = filename.replace("_pythia160m_ckpt_series.jsonl", "")
            else:
                continue
                
            file_path = os.path.join(search_path, filename)
            
            try:
                with open(file_path, 'r') as f:
                    data = []
                    for line in f:
                        data.append(json.loads(line.strip()))
                    all_neuron_data[neuron_id] = data
                    available_neurons.append(neuron_id)
                    files_found += 1
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
                continue
        
        # This line seems to be necessary for functionality - keeping it but hiding output
        pass
    
    return all_neuron_data, sorted(available_neurons)

def analyze_common_words(texts: List[str], min_length: int = 3, top_n: int = 2) -> List[str]:
    """Analyze texts to find the most commonly appearing words and word-fragments"""
    if not texts:
        return []
    
    # Combine all texts
    combined_text = ' '.join(texts).lower()
    
    # Define stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'among', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'not', 'no', 'yes',
        'as', 'so', 'if', 'when', 'where', 'why', 'how', 'what', 'who', 'which', 'whose', 'whom',
        'all', 'any', 'some', 'many', 'few', 'most', 'more', 'less', 'much', 'little', 'very', 'too',
        'also', 'only', 'just', 'even', 'still', 'again', 'here', 'there', 'now', 'then', 'today',
        'yesterday', 'tomorrow', 'one', 'two', 'three', 'first', 'second', 'third', 'last', 'next'
    }
    
    # Count whole words
    word_counts = {}
    words = re.findall(r'\b[a-z]+\b', combined_text)
    for word in words:
        if len(word) >= min_length and word not in stop_words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Count fragments
    fragment_counts = {}
    for word in words:
        if len(word) >= min_length and word not in stop_words:
            for frag_len in range(min_length, min(len(word) + 1, 8)):
                for start in range(len(word) - frag_len + 1):
                    fragment = word[start:start + frag_len]
                    if fragment not in stop_words:
                        fragment_counts[fragment] = fragment_counts.get(fragment, 0) + 1
    
    # Combine counts
    all_counts = {}
    
    # Add whole words with higher weight
    for word, count in word_counts.items():
        all_counts[word] = count * 2
    
    # Add fragments but only if they appear significantly
    for fragment, count in fragment_counts.items():
        if count >= 5:  # Only include fragments that appear at least 5 times
            if fragment not in all_counts:
                all_counts[fragment] = count
            else:
                all_counts[fragment] += count
    
    # Sort by frequency and return top N
    sorted_words = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:top_n]]

def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words and punctuation"""
    tokens = re.findall(r'\S+|\s+', text)
    return [token for token in tokens if token.strip()]

def highlight_tokens(text: str, cluster_texts: List[str], is_right_panel: bool = False) -> str:
    """Apply highlighting to tokens based on common words analysis"""
    if not is_right_panel:
        return text
    
    if not cluster_texts:
        return text
    
    common_words = analyze_common_words(cluster_texts)
    tokens = tokenize_text(text)
    
    highlighted_tokens = []
    for token in tokens:
        token_lower = token.lower()
        clean_token = re.sub(r'[^\w]', '', token_lower)
        
        is_highlighted = False
        if clean_token:
            for common_word in common_words:
                if common_word in clean_token or clean_token in common_word:
                    is_highlighted = True
                    break
        
        if is_highlighted:
            highlighted_tokens.append(f'<span class="token activation-high">{token}</span>')
        else:
            highlighted_tokens.append(f'<span class="token activation-low">{token}</span>')
    
    return ''.join(highlighted_tokens)

def display_cluster_data(data: Dict[str, Any], is_right_panel: bool = False):
    """Display cluster data with highlighting"""
    cluster_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#F39C12", "#E74C3C", "#9B59B6", "#3498DB"
    ]
    
    if 'cluster_labels' not in data or 'text_examples' not in data:
        st.error("Invalid data format")
        return
    
    cluster_labels = data['cluster_labels']
    text_examples = data['text_examples']
    
    # Group examples by cluster
    clusters = {}
    for i, (label, example) in enumerate(zip(cluster_labels, text_examples)):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(example)
    
    # Display clusters
    for cluster_id in sorted(clusters.keys()):
        examples = clusters[cluster_id]
        color = cluster_colors[int(cluster_id) % len(cluster_colors)]
        
        st.markdown(f"""
        <div class="cluster" style="border-color: {color};">
            <div class="cluster-header" style="background-color: {color};">
                Cluster {cluster_id} ({len(examples)} examples)
            </div>
        """, unsafe_allow_html=True)
        
        for example in examples[:5]:  # Show first 5 examples
            highlighted_text = highlight_tokens(example, examples, is_right_panel)
            st.markdown(f'<div class="example">{highlighted_text}</div>', unsafe_allow_html=True)
        
        if len(examples) > 5:
            st.markdown(f'<div class="example"><em>... and {len(examples) - 5} more examples</em></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Remove title and subtitle
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Model selector
    st.sidebar.markdown("**Select Model:**")
    model_name = st.sidebar.selectbox(
        "Model:",
        ["Pythia-70M", "Pythia-160M"],
        index=0,
        label_visibility="collapsed"
    )
    
    # Show current model
    st.sidebar.markdown(f"**Current Model:** {model_name}")
    st.sidebar.markdown("---")
    
    # Load data for selected model
    with st.spinner(f"Loading {model_name} neuron data..."):
        all_neuron_data, available_neurons = load_neuron_data(model_name)
    
    if not available_neurons:
        st.error(f"No neuron data found for {model_name}. Please make sure the results directory exists and contains the required files.")
        return
    
    # Layer and neuron inputs - dynamic based on model
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if model_name == "Pythia-70M":
            layer = st.number_input("Layer (0-5):", min_value=0, max_value=5, value=0, step=1)
        else:  # Pythia-160M
            layer = st.number_input("Layer (0-11):", min_value=0, max_value=11, value=0, step=1)
    with col2:
        if model_name == "Pythia-70M":
            neuron = st.number_input("Neuron (0-2000, steps of 20):", min_value=0, max_value=2000, value=0, step=20)
        else:  # Pythia-160M
            neuron = st.number_input("Neuron (0-3060, steps of 60):", min_value=0, max_value=3060, value=0, step=60)
    
    # Construct neuron ID
    current_neuron = f"L{layer}N{neuron}"
    
    # Checkpoint slider
    if current_neuron in all_neuron_data:
        data = all_neuron_data[current_neuron]
        checkpoint_steps = sorted([d['checkpoint_step'] for d in data])
        
        checkpoint = st.sidebar.select_slider(
            "Checkpoint Step:",
            options=checkpoint_steps,
            value=checkpoint_steps[0] if checkpoint_steps else 3000
        )
    else:
        st.sidebar.error(f"No data available for {current_neuron}")
        return
    
    # Legend
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Token Activation:**")
    st.sidebar.markdown(
        '<span class="token activation-high">High</span>: Common words/fragments<br>'
        '<span class="token activation-low">Low</span>: Other tokens',
        unsafe_allow_html=True
    )
    
    # Main content - make left panel wider
    col1, col2 = st.columns([3, 2])  # 3:2 ratio instead of 1:1
    
    with col1:
        st.subheader(f"Checkpoint {checkpoint}")
        current_data = next((d for d in data if d['checkpoint_step'] == checkpoint), None)
        if current_data:
            display_cluster_data(current_data, is_right_panel=False)
        else:
            st.error("No data available for this checkpoint")
    
    with col2:
        st.subheader("Checkpoint 143000 (Final)")
        final_data = next((d for d in data if d['checkpoint_step'] == 143000), data[-1] if data else None)
        if final_data:
            display_cluster_data(final_data, is_right_panel=True)
        else:
            st.error("No final data available")
    
    # Display current neuron info
    st.markdown(f"**Current Neuron:** {current_neuron} | **Available Checkpoints:** {len(checkpoint_steps)}")

if __name__ == "__main__":
    main()
