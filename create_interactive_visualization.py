#!/usr/bin/env python
"""
Create an interactive HTML/CSS visualization showing the evolution of neurons across checkpoints
Allows users to select different neurons and navigate through checkpoints with a slider
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Any

def load_checkpoint_data(neuron_id: str, model: str = "pythia70m") -> List[Dict[str, Any]]:
    """Load checkpoint data for a specific neuron"""
    file_path = f"results/{model}/{neuron_id}_{model}_ckpt_series.jsonl"
    if not os.path.exists(file_path):
        # Try alternative path
        file_path = f"results/{neuron_id}_{model}_ckpt_series.jsonl"
    
    if not os.path.exists(file_path):
        print(f"Could not find data file for {neuron_id}")
        return []
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    return data

def get_available_neurons(model: str = "pythia70m") -> List[str]:
    """Get list of available neurons"""
    neurons = []
    
    # Check both possible directories
    for base_dir in ["results", f"results/{model}"]:
        if os.path.exists(base_dir):
            for file in os.listdir(base_dir):
                if file.endswith(f"_{model}_ckpt_series.jsonl"):
                    neuron_id = file.replace(f"_{model}_ckpt_series.jsonl", "")
                    neurons.append(neuron_id)
    
    return sorted(neurons, key=lambda x: (int(x.split('N')[0][1:]), int(x.split('N')[1])))

def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words and punctuation"""
    tokens = re.findall(r'\S+|\s+', text)
    return [token for token in tokens if token.strip()]

def analyze_common_words(texts: List[str], min_length: int = 3, top_n: int = 2) -> List[str]:
    """Analyze texts to find the most commonly appearing words and word-fragments"""
    if not texts:
        return []
    
    # Combine all texts
    combined_text = ' '.join(texts).lower()
    
    # Common stop words to ignore
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
    
    # Extract words and count frequencies
    word_counts = {}
    
    # Split by whitespace and punctuation, keep alphanumeric sequences
    words = re.findall(r'\b[a-z]+\b', combined_text)
    
    for word in words:
        if len(word) >= min_length and word not in stop_words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Also look for common word fragments (substrings that appear frequently)
    fragment_counts = {}
    for word in words:
        if len(word) >= min_length and word not in stop_words:
            # Generate fragments of different lengths
            for frag_len in range(min_length, min(len(word) + 1, 8)):  # max fragment length of 7
                for start in range(len(word) - frag_len + 1):
                    fragment = word[start:start + frag_len]
                    if fragment not in stop_words:
                        fragment_counts[fragment] = fragment_counts.get(fragment, 0) + 1
    
    # Combine and sort by frequency
    all_counts = {}
    
    # Add whole words with higher weight
    for word, count in word_counts.items():
        all_counts[word] = count * 2  # Give whole words more weight
    
    # Add fragments but only if they appear significantly
    for fragment, count in fragment_counts.items():
        if count >= 5:  # Only include fragments that appear at least 5 times (more selective)
            if fragment not in all_counts:
                all_counts[fragment] = count
            else:
                all_counts[fragment] += count
    
    # Sort by frequency and return top N
    sorted_words = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:top_n]]

def highlight_pattern(tokens: List[str], pattern: str = "what is known", cluster_id: str = None, is_right_panel: bool = False, cluster_texts: List[str] = None) -> List[float]:
    """Generate activation values based on commonly appearing words in the cluster"""
    if not cluster_texts:
        return [0.1] * len(tokens)
    
    # Don't highlight anything on the left side
    if not is_right_panel:
        return [0.1] * len(tokens)
    
    # Analyze common words in this cluster
    common_words = analyze_common_words(cluster_texts)
    
    activations = []
    for token in tokens:
        token_lower = token.lower()
        # Remove punctuation from token for matching
        clean_token = re.sub(r'[^\w]', '', token_lower)
        
        # Check if token or any fragment matches common words
        is_highlighted = False
        if clean_token:
            for common_word in common_words:
                if common_word in clean_token or clean_token in common_word:
                    is_highlighted = True
                    break
        
        if is_highlighted:
            activations.append(0.9)  # High activation for commonly appearing words/fragments
        else:
            activations.append(0.1)  # Low activation for other tokens
    
    return activations

def create_interactive_visualization():
    """Create interactive HTML/CSS visualization"""
    
    # Get available neurons
    available_neurons = get_available_neurons()
    
    if not available_neurons:
        print("No neuron data found!")
        return
    
    # Load sample data to get checkpoint steps
    sample_data = load_checkpoint_data(available_neurons[0])
    if not sample_data:
        print("Could not load sample data!")
        return
    
    checkpoint_steps = [checkpoint['checkpoint_step'] for checkpoint in sample_data]
    
    # Load all neuron data
    all_neuron_data = {}
    print("Loading neuron data...")
    for i, neuron in enumerate(available_neurons):
        print(f"Loading {neuron} ({i+1}/{len(available_neurons)})")
        data = load_checkpoint_data(neuron)
        if data:
            all_neuron_data[neuron] = data
    
    # Colors for clusters
    cluster_colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57", "#ff9ff3", "#54a0ff", "#5f27cd"]
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Neuron Evolution Visualization</title>
    <style>
        body {{
            font-family: 'DejaVu Sans Mono', 'Courier New', monospace;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        .controls {{
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .control-group {{
            margin-bottom: 15px;
        }}
        
        .control-group label {{
            display: inline-block;
            font-weight: bold;
            margin-right: 10px;
            margin-bottom: 5px;
            color: #495057;
        }}
        
        .control-group input[type="number"] {{
            width: 80px;
            padding: 8px;
            margin-right: 15px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .slider-container input[type="range"] {{
            flex: 1;
        }}
        
        .slider-value {{
            min-width: 80px;
            text-align: center;
            font-weight: bold;
            color: #495057;
        }}
        
        .content {{
            display: flex;
            min-height: 600px;
        }}
        
        .panel {{
            flex: 1;
            padding: 20px;
            border-right: 2px solid #e9ecef;
            overflow-y: auto;
        }}
        
        .panel:last-child {{
            border-right: none;
        }}
        
        .panel-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            text-align: center;
        }}
        
        .cluster {{
            margin-bottom: 15px;
            border: 2px solid;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .cluster-header {{
            padding: 8px 12px;
            font-weight: bold;
            color: white;
            text-align: center;
        }}
        
        .example {{
            padding: 8px;
            background: white;
            border-bottom: 1px solid #e9ecef;
            line-height: 1.4;
        }}
        
        .example:last-child {{
            border-bottom: none;
        }}
        
        .token {{
            display: inline-block;
            padding: 2px 4px;
            margin: 1px;
            border-radius: 3px;
            transition: all 0.2s ease;
        }}
        
        .token:hover {{
            transform: scale(1.05);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        
        .activation-low {{
            background-color: rgba(255, 107, 107, 0.1);
            border: 1px solid rgba(255, 107, 107, 0.3);
        }}
        
        .activation-medium {{
            background-color: rgba(255, 107, 107, 0.4);
            border: 1px solid rgba(255, 107, 107, 0.6);
        }}
        
        .activation-high {{
            background-color: rgba(255, 107, 107, 0.8);
            border: 1px solid rgba(255, 107, 107, 1.0);
            font-weight: bold;
        }}
        
        .legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(255, 107, 107, 0.9);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
            max-width: 250px;
        }}
        
        .loading {{
            text-align: center;
            padding: 40px;
            color: #6c757d;
            font-style: italic;
        }}
        
        .error {{
            text-align: center;
            padding: 40px;
            color: #dc3545;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="controls">
            <div class="control-group">
                <label for="layer-input">Layer (0-5):</label>
                <input type="number" id="layer-input" min="0" max="5" value="0" placeholder="0-5">
                <label for="neuron-input">Neuron (0-2000, steps of 20):</label>
                <input type="number" id="neuron-input" min="0" max="2000" step="20" value="0" placeholder="0, 20, 40...">
            </div>
            
            <div class="control-group">
                <label for="checkpoint-slider">Checkpoint Step:</label>
                <div class="slider-container">
                    <input type="range" id="checkpoint-slider" 
                           min="{min(checkpoint_steps)}" 
                           max="{max(checkpoint_steps)}" 
                           step="{checkpoint_steps[1] - checkpoint_steps[0] if len(checkpoint_steps) > 1 else 1000}"
                           value="{checkpoint_steps[0]}">
                    <span class="slider-value" id="slider-value">{checkpoint_steps[0]}</span>
                </div>
            </div>
        </div>
        
        <div class="content">
            <div class="panel">
                <div class="panel-title">Checkpoint <span id="current-checkpoint">-</span></div>
                <div id="left-content" class="loading">Loading...</div>
            </div>
            
            <div class="panel">
                <div class="panel-title">Checkpoint 143000 (Final)</div>
                <div id="right-content" class="loading">Loading...</div>
            </div>
        </div>
        
        <div class="legend">
            <strong>Token Activation:</strong><br>
            • <span class="token activation-high">High</span>: Common words/fragments<br>
            • <span class="token activation-low">Low</span>: Other tokens
        </div>
    </div>

    <script>
        // Embedded neuron data
        const neuronData = {json.dumps(all_neuron_data)};
        const availableNeurons = {json.dumps(available_neurons)};
        const checkpointSteps = {json.dumps(checkpoint_steps)};
        const clusterColors = {json.dumps(cluster_colors)};
        
        // DOM elements
        const layerInput = document.getElementById('layer-input');
        const neuronInput = document.getElementById('neuron-input');
        const checkpointSlider = document.getElementById('checkpoint-slider');
        const sliderValue = document.getElementById('slider-value');
        const currentCheckpoint = document.getElementById('current-checkpoint');
        const leftContent = document.getElementById('left-content');
        const rightContent = document.getElementById('right-content');
        
        // Current state
        let currentNeuron = 'L0N0';
        let currentCheckpointStep = parseInt(checkpointSlider.value);
        
        // Event listeners
        layerInput.addEventListener('input', function() {{
            updateNeuronFromInputs();
        }});
        
        neuronInput.addEventListener('input', function() {{
            updateNeuronFromInputs();
        }});
        
        function updateNeuronFromInputs() {{
            const layer = parseInt(layerInput.value) || 0;
            const neuron = parseInt(neuronInput.value) || 0;
            currentNeuron = `L${{layer}}N${{neuron}}`;
            updateVisualization();
        }}
        
        checkpointSlider.addEventListener('input', function() {{
            currentCheckpointStep = parseInt(this.value);
            sliderValue.textContent = this.value;
            currentCheckpoint.textContent = this.value;
            updateVisualization();
        }});
        
        // Update visualization
        function updateVisualization() {{
            if (!neuronData[currentNeuron]) {{
                leftContent.innerHTML = '<div class="error">No data available for this neuron</div>';
                rightContent.innerHTML = '<div class="error">No data available for this neuron</div>';
                return;
            }}
            
            const data = neuronData[currentNeuron];
            const currentData = data.find(d => d.checkpoint_step === currentCheckpointStep);
            const finalData = data.find(d => d.checkpoint_step === 143000) || data[data.length - 1];
            
            if (currentData) {{
                leftContent.innerHTML = generateClusterHTML(currentData, false);
            }} else {{
                leftContent.innerHTML = '<div class="error">No data available for this checkpoint</div>';
            }}
            
            if (finalData) {{
                rightContent.innerHTML = generateClusterHTML(finalData, true);
            }} else {{
                rightContent.innerHTML = '<div class="error">No final data available</div>';
            }}
        }}
        
        // Generate cluster HTML
        function generateClusterHTML(checkpointData, isRightPanel) {{
            if (!checkpointData || !checkpointData.text_examples) {{
                return '<div class="error">No data available for this checkpoint</div>';
            }}
            
            // Group text examples by cluster
            const clusters = {{}};
            checkpointData.text_examples.forEach((text, index) => {{
                const clusterId = checkpointData.cluster_labels[index];
                if (!clusters[clusterId]) {{
                    clusters[clusterId] = [];
                }}
                clusters[clusterId].push(text);
            }});
            
            let html = '';
            const sortedClusters = Object.entries(clusters).sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
            
            sortedClusters.forEach(([clusterId, texts], index) => {{
                const color = clusterColors[index % clusterColors.length];
                html += `
                    <div class="cluster" style="border-color: ${{color}};">
                        <div class="cluster-header" style="background-color: ${{color}};">
                            Cluster ${{clusterId}}
                        </div>
                `;
                
                texts.forEach(text => {{
                    const tokens = tokenizeText(text);
                    const activations = highlightPattern(tokens, "what is known", clusterId, isRightPanel, texts);
                    
                    html += '<div class="example">';
                    tokens.forEach((token, i) => {{
                        const activation = activations[i] || 0.1;
                        let cssClass = "activation-low";
                        if (activation > 0.7) {{
                            cssClass = "activation-high";
                        }} else if (activation > 0.3) {{
                            cssClass = "activation-medium";
                        }}
                        html += `${{'<span class="token ' + cssClass + '">' + token + '</span>'}}`;
                    }});
                    html += '</div>';
                }});
                
                html += '</div>';
            }});
            
            return html;
        }}
        
        // Tokenize text (JavaScript version)
        function tokenizeText(text) {{
            const tokens = text.match(/\\S+|\\s+/g) || [];
            return tokens.filter(token => token.trim());
        }}
        
        // Analyze common words (JavaScript version)
        function analyzeCommonWords(texts, minLength = 3, topN = 2) {{
            if (!texts || texts.length === 0) {{
                return [];
            }}
            
            const combinedText = texts.join(' ').toLowerCase();
            
            // Common stop words to ignore
            const stopWords = new Set([
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
            ]);
            
            const wordCounts = {{}};
            const fragmentCounts = {{}};
            
            // Extract words
            const words = combinedText.match(/\\b[a-z]+\\b/g) || [];
            
            // Count whole words
            for (const word of words) {{
                if (word.length >= minLength && !stopWords.has(word)) {{
                    wordCounts[word] = (wordCounts[word] || 0) + 1;
                }}
            }}
            
            // Count fragments
            for (const word of words) {{
                if (word.length >= minLength && !stopWords.has(word)) {{
                    for (let fragLen = minLength; fragLen < Math.min(word.length + 1, 8); fragLen++) {{
                        for (let start = 0; start <= word.length - fragLen; start++) {{
                            const fragment = word.substring(start, start + fragLen);
                            if (!stopWords.has(fragment)) {{
                                fragmentCounts[fragment] = (fragmentCounts[fragment] || 0) + 1;
                            }}
                        }}
                    }}
                }}
            }}
            
            // Combine counts
            const allCounts = {{}};
            
            // Add whole words with higher weight
            for (const [word, count] of Object.entries(wordCounts)) {{
                allCounts[word] = count * 2;
            }}
            
            // Add significant fragments
            for (const [fragment, count] of Object.entries(fragmentCounts)) {{
                if (count >= 5) {{  // Only include fragments that appear at least 5 times (more selective)
                    allCounts[fragment] = (allCounts[fragment] || 0) + count;
                }}
            }}
            
            // Sort by frequency and return top N
            const sortedWords = Object.entries(allCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, topN)
                .map(([word, count]) => word);
            
            return sortedWords;
        }}
        
        // Highlight pattern (JavaScript version)
        function highlightPattern(tokens, pattern, clusterId, isRightPanel, cluster_texts) {{
            if (!cluster_texts || cluster_texts.length === 0) {{
                return tokens.map(() => 0.1);
            }}
            
            // Don't highlight anything on the left side
            if (!isRightPanel) {{
                return tokens.map(() => 0.1);
            }}
            
            // Analyze common words in this cluster
            const commonWords = analyzeCommonWords(cluster_texts);
            
            return tokens.map((token) => {{
                const tokenLower = token.toLowerCase();
                // Remove punctuation from token for matching
                const cleanToken = tokenLower.replace(/[^\\w]/g, '');
                
                // Check if token or any fragment matches common words
                let isHighlighted = false;
                if (cleanToken) {{
                    for (const commonWord of commonWords) {{
                        if (commonWord.includes(cleanToken) || cleanToken.includes(commonWord)) {{
                            isHighlighted = true;
                            break;
                        }}
                    }}
                }}
                
                return isHighlighted ? 0.9 : 0.1;  // High activation for common words/fragments, low for others
            }});
        }}
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            updateVisualization();
        }});
    </script>
</body>
</html>
"""
    
    # Save the HTML file
    output_file = Path("interactive_visualization.html")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive visualization saved to {output_file}")
    print("Open this file in a web browser to view the visualization")
    print(f"Available neurons: {len(available_neurons)}")
    print(f"Checkpoint range: {min(checkpoint_steps)} to {max(checkpoint_steps)}")

if __name__ == "__main__":
    create_interactive_visualization()
