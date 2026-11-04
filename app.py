"""
Shared Computation AI Lab - Streamlit Application
Interactive Generative AI Dashboard demonstrating shared computation for next-word prediction.
"""

import warnings
warnings.filterwarnings("ignore")

import io
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

import model as lab_model

# ---------------------------------------------------------
# Page Configuration & Modern Dark Navy Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Shared Computation AI Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Main app dark navy styling */
.stApp {
    background-color: #0B0F19;
    color: #E2E8F0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0E1626;
    border-right: 1px solid #1E293B;
}

/* Glassmorphism card */
.glass-card {
    background: linear-gradient(135deg, rgba(17, 27, 46, 0.85) 0%, rgba(15, 23, 42, 0.7) 100%);
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(8px);
    transition: border-color 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(139, 92, 246, 0.45);
}

/* Card titles */
.card-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 0.5rem;
}

/* Metric badges */
.metric-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}

.badge-blue {
    background: rgba(56, 189, 248, 0.15);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}

.badge-purple {
    background: rgba(168, 85, 247, 0.15);
    color: #C084FC;
    border: 1px solid rgba(168, 85, 247, 0.3);
}

.badge-green {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* Token chips */
.token-chip {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    margin: 0.2rem;
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid #334155;
    border-radius: 8px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.85rem;
    color: #F1F5F9;
}

.token-chip-highlight {
    background: rgba(139, 92, 246, 0.25);
    border-color: #8B5CF6;
    color: #DDD6FE;
    font-weight: 600;
}

/* Pair display flex row */
.pair-box {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #111A2E;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    margin-bottom: 0.5rem;
}

.pair-input {
    color: #38BDF8;
    font-weight: 700;
    font-size: 1.05rem;
}

.pair-arrow {
    color: #94A3B8;
    font-size: 1.1rem;
    padding: 0 0.5rem;
}

.pair-target {
    color: #A855F7;
    font-weight: 700;
    font-size: 1.05rem;
}

/* Generation trace card */
.step-card {
    background: #11192C;
    border: 1px solid #1F2E4A;
    border-left: 4px solid #8B5CF6;
    border-radius: 8px;
    padding: 0.8rem 1.1rem;
    margin-bottom: 0.6rem;
}

/* Button enhancements */
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}

/* Plotly chart container styling */
.plotly-container {
    border-radius: 12px;
    overflow: hidden;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
def init_session_state():
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = lab_model.SharedTokenizer()
    
    if "embedding_dim" not in st.session_state:
        st.session_state.embedding_dim = 16
        
    if "random_seed" not in st.session_state:
        st.session_state.random_seed = 42
        
    if "untrained_model" not in st.session_state:
        st.session_state.untrained_model = lab_model.SharedNextWordModel(
            vocab_size=st.session_state.tokenizer.vocab_size,
            embedding_dim=st.session_state.embedding_dim,
            seed=st.session_state.random_seed
        )
        
    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None
        
    if "is_trained" not in st.session_state:
        st.session_state.is_trained = False
        
    if "training_history" not in st.session_state:
        st.session_state.training_history = []
        
    if "final_accuracy" not in st.session_state:
        st.session_state.final_accuracy = 0.0
        
    if "training_epochs" not in st.session_state:
        st.session_state.training_epochs = 250
        
    if "training_lr" not in st.session_state:
        st.session_state.training_lr = 0.05

init_session_state()

tokenizer: lab_model.SharedTokenizer = st.session_state.tokenizer
X_tensor, Y_tensor = tokenizer.get_tensors()


# ---------------------------------------------------------
# Plotly Dark Theme Preset & Helpers
# ---------------------------------------------------------
def apply_dark_theme(fig: go.Figure, title: str = "", y_range: list = None) -> go.Figure:
    """Applies consistent dark navy theme and styling to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#0E1626",
        plot_bgcolor="#111B2E",
        font=dict(color="#E2E8F0", family="sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15, color="#F8FAFC")))
    fig.update_xaxes(
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        tickfont=dict(color="#94A3B8")
    )
    if y_range is not None:
        fig.update_yaxes(
            gridcolor="#1E293B",
            zerolinecolor="#334155",
            tickfont=dict(color="#94A3B8"),
            range=y_range
        )
    else:
        fig.update_yaxes(
            gridcolor="#1E293B",
            zerolinecolor="#334155",
            tickfont=dict(color="#94A3B8")
        )
    return fig


def render_probability_bars(probs_matrix: pd.DataFrame, title: str, highlight_targets: bool = False):
    """Generates an interactive Plotly grouped bar chart showing next-word distribution for each input."""
    fig = go.Figure()
    colors = ["#38BDF8", "#818CF8", "#A78BFA", "#C084FC", "#F472B6", "#FB7185", "#34D399"]
    
    for i, vocab_col in enumerate(probs_matrix.columns):
        fig.add_trace(go.Bar(
            x=probs_matrix.index,
            y=probs_matrix[vocab_col],
            name=vocab_col,
            marker_color=colors[i % len(colors)],
            hovertemplate="<b>Input:</b> %{x}<br><b>Candidate:</b> " + vocab_col + "<br><b>Probability:</b> %{y:.2f}%<extra></extra>"
        ))
        
    fig.update_layout(
        xaxis_title="Input Token Context (Pair #)",
        yaxis_title="Next-Word Probability (%)",
        barmode="group",
        legend=dict(
            title="Candidate Token",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        )
    )
    return apply_dark_theme(fig, title=title, y_range=[0, 105])


def render_probability_heatmap(probs_matrix: pd.DataFrame, title: str):
    """Generates an interactive heatmap of the 6x7 transition probability matrix."""
    fig = px.imshow(
        probs_matrix,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale=[[0, "#0E1626"], [0.25, "#1E3A8A"], [0.6, "#7C3AED"], [1.0, "#38BDF8"]],
        labels=dict(x="Predicted Next-Word Token", y="Input Token", color="Probability (%)")
    )
    fig.update_traces(
        textfont=dict(color="#FFFFFF", size=12)
    )
    return apply_dark_theme(fig, title=title)


# ---------------------------------------------------------
# Sidebar Navigation & Hyperparameter Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔬 Shared Computation AI Lab")
    st.caption("One Sentence • Seven Tokens • Single Neural Network")

    # Status Indicator Card
    status_color = "#34D399" if st.session_state.is_trained else "#FBBF24"
    status_text = "TRAINED (100% Convergence)" if st.session_state.is_trained else "UNTRAINED (Baseline)"
    
    st.markdown(
        f"""
        <div style="background: #111B2E; border: 1px solid #1E293B; border-radius: 10px; padding: 0.8rem; margin-bottom: 1rem;">
            <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Model Engine State</div>
            <div style="color: {status_color}; font-weight: 800; font-size: 0.95rem; margin-top: 0.2rem;">
                ● {status_text}
            </div>
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.2rem;">
                Vocab: {tokenizer.vocab_size} tokens | Pairs: {len(tokenizer.pairs)} | Batch Size: 6
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### 🧭 Module Navigation")
    modules = [
        "1. Executive Overview",
        "2. Tokenization & Bigrams",
        "3. Shared Computation Architecture",
        "4. Before Training (Untrained Model)",
        "5. Training Engine (Batch Gradient Descent)",
        "6. After Training (Model Evaluation)",
        "7. Live Autoregressive Generation",
        "8. Comparative Visualizations",
        "9. Downloadable Results & Data"
    ]
    
    selected_module = st.radio(
        "Select Lab Module",
        modules,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Hyperparameter Controls")
    
    epochs_input = st.slider(
        "Training Epochs",
        min_value=50,
        max_value=600,
        value=st.session_state.training_epochs,
        step=25,
        help="Number of batch gradient descent iterations."
    )
    st.session_state.training_epochs = epochs_input

    lr_input = st.slider(
        "Learning Rate (Adam)",
        min_value=0.005,
        max_value=0.15,
        value=st.session_state.training_lr,
        step=0.005,
        format="%.3f",
        help="Step size for shared weight parameter optimization."
    )
    st.session_state.training_lr = lr_input

    emb_dim_input = st.selectbox(
        "Embedding Dimension ($d$)",
        [8, 16, 32],
        index=1,
        help="Vector dimensionality for the shared token embedding space."
    )
    
    # If embedding dimension changes, reinitialize untrained model
    if emb_dim_input != st.session_state.embedding_dim:
        st.session_state.embedding_dim = emb_dim_input
        st.session_state.untrained_model = lab_model.SharedNextWordModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=emb_dim_input,
            seed=st.session_state.random_seed
        )
        st.session_state.trained_model = None
        st.session_state.is_trained = False
        st.rerun()

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Reset Lab", use_container_width=True):
            st.session_state.trained_model = None
            st.session_state.is_trained = False
            st.session_state.training_history = []
            st.session_state.untrained_model = lab_model.SharedNextWordModel(
                vocab_size=tokenizer.vocab_size,
                embedding_dim=st.session_state.embedding_dim,
                seed=st.session_state.random_seed
            )
            st.success("Reset to baseline!")
            st.rerun()

    with col_btn2:
        st.markdown(
            f"""
            <div style="font-size: 0.72rem; color: #94A3B8; text-align: center; padding-top: 0.5rem;">
                Seed: {st.session_state.random_seed} (Locked)
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# MODULE 1: EXECUTIVE OVERVIEW
# ---------------------------------------------------------
if selected_module == "1. Executive Overview":
    st.markdown("## 📊 1. Executive Overview")
    st.markdown(
        """
        Welcome to the **Shared Computation AI Lab**. This laboratory demonstrates how modern neural language 
        models achieve parallel efficiency and parameter sharing using **one unified neural network** 
        to process all sequence transitions simultaneously.
        """
    )

    # Top Metric Banner Cards
    param_info = st.session_state.untrained_model.count_parameters()
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 1rem; text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Sequence Tokens</div>
                <div style="font-size: 2rem; font-weight: 800; color: #38BDF8;">{tokenizer.vocab_size}</div>
                <div class="metric-badge badge-blue">1 Sentence</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 1rem; text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Training Pairs</div>
                <div style="font-size: 2rem; font-weight: 800; color: #A855F7;">{len(tokenizer.pairs)}</div>
                <div class="metric-badge badge-purple">Bigram Transitions</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 1rem; text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Vocabulary Size</div>
                <div style="font-size: 2rem; font-weight: 800; color: #34D399;">{tokenizer.vocab_size}</div>
                <div class="metric-badge badge-green">Discrete Classes</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 1rem; text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Shared Parameters</div>
                <div style="font-size: 2rem; font-weight: 800; color: #FBBF24;">{param_info['total_params']}</div>
                <div class="metric-badge badge-amber">Single Network</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col5:
        curr_acc = st.session_state.final_accuracy * 100.0 if st.session_state.is_trained else 0.0
        acc_color = "#34D399" if st.session_state.is_trained else "#94A3B8"
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 1rem; text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Model Accuracy</div>
                <div style="font-size: 2rem; font-weight: 800; color: {acc_color};">{curr_acc:.1f}%</div>
                <div class="metric-badge {'badge-green' if st.session_state.is_trained else 'badge-blue'}">{'Trained' if st.session_state.is_trained else 'Untrained'}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Core Sentence Card
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-header">
                <span>📖 Training Corpus & Sequence</span>
                <span class="metric-badge badge-blue">Exact Ground Truth</span>
            </div>
            <div style="font-size: 1.25rem; font-weight: 600; color: #F1F5F9; padding: 0.4rem 0;">
                "{lab_model.TRAINING_SENTENCE}"
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.4rem;">
                Each transition token $x_t \\to y_t$ forms an autoregressive learning target. 
                Instead of six independent networks, <b>one shared neural network</b> solves all six transitions in parallel.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Conceptual Explanation Columns
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-header">
                    <span>⚡ What is Shared Computation?</span>
                </div>
                <p style="font-size: 0.9rem; line-height: 1.5; color: #CBD5E1;">
                    In classical computing, computing 6 predictions might require 6 serial executions or separate models.
                    In deep learning:
                </p>
                <ul style="font-size: 0.88rem; color: #94A3B8; line-height: 1.6;">
                    <li><b>Parameter Sharing:</b> The exact same weights matrix $W$ and embedding matrix $E$ are reused across sequence positions.</li>
                    <li><b>Batch Parallelism:</b> All 6 input tokens are packaged into a batch tensor $\\mathbf{X} \\in \\mathbb{R}^6$ and passed simultaneously through the GPU/CPU matrix multiplication kernels.</li>
                    <li><b>Joint Optimization:</b> The Cross-Entropy loss aggregates errors across all 6 targets into a single scalar loss: $\\mathcal{L} = \\frac{1}{6} \\sum_{i=1}^6 \\ell(x_i, y_i)$.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_right:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-header">
                    <span>📐 Model Architecture Specifications</span>
                </div>
                <table style="width: 100%; font-size: 0.88rem; color: #CBD5E1; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #1E293B;">
                        <td style="padding: 0.4rem 0; color: #94A3B8;">Vocabulary Size ($V$)</td>
                        <td style="text-align: right; font-weight: 600; color: #38BDF8;">{tokenizer.vocab_size}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1E293B;">
                        <td style="padding: 0.4rem 0; color: #94A3B8;">Embedding Layer</td>
                        <td style="text-align: right; font-weight: 600; color: #A855F7;">nn.Embedding(7, {st.session_state.embedding_dim}) [{param_info['embedding_params']} params]</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1E293B;">
                        <td style="padding: 0.4rem 0; color: #94A3B8;">Linear Output Layer</td>
                        <td style="text-align: right; font-weight: 600; color: #34D399;">nn.Linear({st.session_state.embedding_dim}, 7) [{param_info['linear_w_params'] + param_info['linear_b_params']} params]</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1E293B;">
                        <td style="padding: 0.4rem 0; color: #94A3B8;">Total Trainable Weights</td>
                        <td style="text-align: right; font-weight: 700; color: #FBBF24;">{param_info['total_params']} parameters</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.4rem 0; color: #94A3B8;">Loss & Optimizer</td>
                        <td style="text-align: right; font-weight: 600; color: #CBD5E1;">CrossEntropyLoss + Adam</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# MODULE 2: TOKENIZATION & BIGRAMS
# ---------------------------------------------------------
elif selected_module == "2. Tokenization & Bigrams":
    st.markdown("## 🔤 2. Tokenization & Bigrams")
    st.markdown(
        """
        Tokenization maps natural language words into discrete integer token IDs.
        From the seven extracted tokens, six consecutive bigram $(x_t, y_t)$ training pairs are generated.
        """
    )

    col_t1, col_t2 = st.columns([1, 1.4])

    with col_t1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-header">
                    <span>🗂️ Vocabulary Dictionary (V = {tokenizer.vocab_size})</span>
                </div>
                <div style="margin-bottom: 0.8rem;">
            """,
            unsafe_allow_html=True
        )
        for token_id, token_text in tokenizer.id_to_token.items():
            st.markdown(
                f"""
                <span class="token-chip">
                    <span style="color: #A855F7; font-weight: bold;">#{token_id}</span> : 
                    <span style="color: #38BDF8; font-weight: bold;">"{token_text}"</span>
                </span>
                """,
                unsafe_allow_html=True
            )
        st.markdown("</div></div>", unsafe_allow_html=True)

        vocab_df = tokenizer.get_vocabulary_dataframe()
        st.dataframe(vocab_df, use_container_width=True, hide_index=True)

    with col_t2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-header">
                    <span>🔗 All Six Input-Target Bigram Pairs</span>
                    <span class="metric-badge badge-purple">Sequence Bigrams</span>
                </div>
            """,
            unsafe_allow_html=True
        )

        for i, (inp, tgt) in enumerate(tokenizer.pairs):
            inp_id = tokenizer.encode(inp)
            tgt_id = tokenizer.encode(tgt)
            st.markdown(
                f"""
                <div class="pair-box">
                    <div style="display: flex; align-items: center; gap: 0.6rem;">
                        <span class="metric-badge badge-blue">Pair #{i+1}</span>
                        <span class="pair-input">"{inp}"</span>
                        <span style="font-size: 0.75rem; color: #64748B;">[ID: {inp_id}]</span>
                    </div>
                    <span class="pair-arrow">➔</span>
                    <div style="display: flex; align-items: center; gap: 0.6rem;">
                        <span class="pair-target">"{tgt}"</span>
                        <span style="font-size: 0.75rem; color: #64748B;">[ID: {tgt_id}]</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Tensor Representation Table
    st.markdown("### 🔢 Batch Tensor Representation")
    st.caption("These two 1D PyTorch tensors are passed to the neural network for joint shared computation:")
    
    t_df = pd.DataFrame({
        "Batch Index [b]": list(range(len(tokenizer.pairs))),
        "Input Token": [p[0] for p in tokenizer.pairs],
        "Input ID Tensor X[b]": X_tensor.tolist(),
        "Target Token": [p[1] for p in tokenizer.pairs],
        "Target ID Tensor Y[b]": Y_tensor.tolist(),
    })
    st.dataframe(t_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# MODULE 3: SHARED COMPUTATION ARCHITECTURE
# ---------------------------------------------------------
elif selected_module == "3. Shared Computation Architecture":
    st.markdown("## ⚡ 3. Shared Computation Architecture")
    st.markdown(
        """
        Visualizing **how all six input tokens enter the exact same neural network simultaneously**.
        Notice that every input token accesses the same shared embedding table and passes through
        the same linear transformation weights.
        """
    )

    # Interactive Batch Flow Diagram
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-header">
                <span>🔄 Matrix Parallel Pipeline: 6 Inputs ➔ 1 Shared Network ➔ 6 Probability Distributions</span>
            </div>
            <div style="display: flex; flex-wrap: wrap; justify-content: space-around; align-items: center; gap: 0.5rem; padding: 1rem 0; font-family: monospace;">
                
                <!-- Stage 1 -->
                <div style="background: #111B2E; border: 1px solid #1E293B; border-radius: 8px; padding: 0.8rem; text-align: center; min-width: 130px;">
                    <div style="color: #38BDF8; font-weight: bold; font-size: 0.8rem;">Batch Input</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #F1F5F9; margin: 0.3rem 0;">X ∈ ℕ⁶</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">Tokens: [0..5]</div>
                </div>

                <div style="color: #8B5CF6; font-size: 1.5rem;">➔</div>

                <!-- Stage 2 -->
                <div style="background: #161F36; border: 2px solid #8B5CF6; border-radius: 8px; padding: 0.8rem; text-align: center; min-width: 170px;">
                    <div style="color: #C084FC; font-weight: bold; font-size: 0.8rem;">Shared Embedding</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #F1F5F9; margin: 0.3rem 0;">E ∈ ℝ^{{7×{st.session_state.embedding_dim}}}</div>
                    <div style="font-size: 0.72rem; color: #A78BFA;">Shared Table Lookup</div>
                </div>

                <div style="color: #8B5CF6; font-size: 1.5rem;">➔</div>

                <!-- Stage 3 -->
                <div style="background: #111B2E; border: 1px solid #1E293B; border-radius: 8px; padding: 0.8rem; text-align: center; min-width: 160px;">
                    <div style="color: #38BDF8; font-weight: bold; font-size: 0.8rem;">Hidden States</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #F1F5F9; margin: 0.3rem 0;">H ∈ ℝ^{{6×{st.session_state.embedding_dim}}}</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">6 Embeddings Parallel</div>
                </div>

                <div style="color: #8B5CF6; font-size: 1.5rem;">➔</div>

                <!-- Stage 4 -->
                <div style="background: #161F36; border: 2px solid #8B5CF6; border-radius: 8px; padding: 0.8rem; text-align: center; min-width: 170px;">
                    <div style="color: #C084FC; font-weight: bold; font-size: 0.8rem;">Shared Linear Layer</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #F1F5F9; margin: 0.3rem 0;">W ∈ ℝ^{{{st.session_state.embedding_dim}×7}} + b</div>
                    <div style="font-size: 0.72rem; color: #A78BFA;">Same Weights for all 6</div>
                </div>

                <div style="color: #8B5CF6; font-size: 1.5rem;">➔</div>

                <!-- Stage 5 -->
                <div style="background: #111B2E; border: 1px solid #1E293B; border-radius: 8px; padding: 0.8rem; text-align: center; min-width: 140px;">
                    <div style="color: #34D399; font-weight: bold; font-size: 0.8rem;">Softmax Logits</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #F1F5F9; margin: 0.3rem 0;">P ∈ ℝ⁶ˣ⁷</div>
                    <div style="font-size: 0.72rem; color: #34D399;">Row-wise Softmax</div>
                </div>

            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Mathematical Deep Dive Tabs
    tab_math, tab_weights, tab_flow = st.tabs(["📐 Mathematical Formulations", "🔍 Inspect Shared Weight Tensors", "📊 Batch Tensor Dimensions"])

    with tab_math:
        st.markdown(
            r"""
            ### Unified Batch Equations
            1. **Input Batch Vector:**
               $$X = \begin{bmatrix} x_1 & x_2 & x_3 & x_4 & x_5 & x_6 \end{bmatrix}^T = \begin{bmatrix} 0 & 1 & 2 & 3 & 4 & 5 \end{bmatrix}^T$$
            2. **Shared Embedding Matrix:**
               $$E \in \mathbb{R}^{7 \times d}, \quad H = \text{Embedding}(X, E) \in \mathbb{R}^{6 \times d}$$
               *(Every input token row $i$ extracts row $x_i$ from the same matrix $E$.)*
            3. **Shared Linear Classifier Projection:**
               $$Z = H W^T + \mathbf{b} \in \mathbb{R}^{6 \times 7}$$
               *(The exact same weight matrix $W \in \mathbb{R}^{7 \times d}$ and bias vector $\mathbf{b} \in \mathbb{R}^7$ multiply each of the 6 hidden vectors.)*
            4. **Batch Softmax Probabilities:**
               $$P_{i, j} = \frac{\exp(Z_{i, j})}{\sum_{k=0}^{6} \exp(Z_{i, k})} \quad \forall i \in \{1 \dots 6\}, j \in \{0 \dots 6\}$$
            5. **Joint Cross-Entropy Loss:**
               $$\mathcal{L} = -\frac{1}{6} \sum_{i=1}^{6} \log P_{i, y_i}$$
               A single backward pass $\nabla_\Theta \mathcal{L}$ computes gradients simultaneously for all 6 transitions.
            """
        )

    with tab_weights:
        active_model = st.session_state.trained_model if st.session_state.is_trained else st.session_state.untrained_model
        st.caption(f"Currently viewing parameters of: **{'TRAINED MODEL' if st.session_state.is_trained else 'UNTRAINED MODEL'}**")

        st.markdown("#### Shared Embedding Table (`embedding.weight`)")
        st.caption(f"Shape: {list(active_model.embedding.weight.shape)} (7 vocabulary words × {st.session_state.embedding_dim} hidden dimensions)")
        emb_df = pd.DataFrame(
            active_model.embedding.weight.detach().cpu().numpy(),
            index=[f"{tok} (#{idx})" for idx, tok in tokenizer.id_to_token.items()],
            columns=[f"dim_{d}" for d in range(st.session_state.embedding_dim)]
        )
        st.dataframe(emb_df.round(4), use_container_width=True)

        st.markdown("#### Shared Output Projection Weights (`linear.weight`)")
        st.caption(f"Shape: {list(active_model.linear.weight.shape)} (7 output logits × {st.session_state.embedding_dim} hidden dimensions)")
        lin_df = pd.DataFrame(
            active_model.linear.weight.detach().cpu().numpy(),
            index=[f"Logit '{tok}'" for tok in tokenizer.tokens],
            columns=[f"dim_{d}" for d in range(st.session_state.embedding_dim)]
        )
        st.dataframe(lin_df.round(4), use_container_width=True)

    with tab_flow:
        st.markdown(
            f"""
            | Layer Component | Tensor Variable | Shape | Shared Parameter Count |
            | :--- | :--- | :--- | :--- |
            | Input Batch IDs | `X` | `[6]` | 0 (data) |
            | Target Batch IDs | `Y` | `[6]` | 0 (data) |
            | **Shared Embedding Matrix** | `model.embedding.weight` | `[7, {st.session_state.embedding_dim}]` | **{7 * st.session_state.embedding_dim}** |
            | Batch Embeddings | `H` | `[6, {st.session_state.embedding_dim}]` | 0 (activations) |
            | **Shared Linear Weights** | `model.linear.weight` | `[7, {st.session_state.embedding_dim}]` | **{7 * st.session_state.embedding_dim}** |
            | **Shared Linear Bias** | `model.linear.bias` | `[7]` | **7** |
            | Logits Tensor | `Z` | `[6, 7]` | 0 (activations) |
            | Softmax Probabilities | `P` | `[6, 7]` | 0 (probabilities) |
            | **Total Shared Trainable Parameters** | | | **{active_model.count_parameters()['total_params']}** |
            """
        )


# ---------------------------------------------------------
# MODULE 4: BEFORE TRAINING (UNTRAINED MODEL)
# ---------------------------------------------------------
elif selected_module == "4. Before Training (Untrained Model)":
    st.markdown("## 🎲 4. Before Training: Initial Model Baseline")
    st.markdown(
        """
        Below are the **real probability distributions** produced by the untrained neural network 
        (initialized deterministically with random seed 42).
        Notice that prior to training, probabilities are spread across candidate tokens, 
        yielding high cross-entropy loss.
        """
    )

    untrained_model = st.session_state.untrained_model
    init_loss = lab_model.compute_loss(untrained_model, X_tensor, Y_tensor)
    expected_random_loss = np.log(tokenizer.vocab_size)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center; padding: 0.9rem;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Initial Loss (Measured)</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #F87171;">{init_loss:.4f}</div>
                <div class="metric-badge badge-amber">Untrained State</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_m2:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center; padding: 0.9rem;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Theoretical Expected Random Loss (ln 7)</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #94A3B8;">{expected_random_loss:.4f}</div>
                <div class="metric-badge badge-blue">Statistical Reference</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_m3:
        # Initial accuracy
        eval_untrained = lab_model.get_prediction_analysis(untrained_model, tokenizer, X_tensor, Y_tensor)
        untrained_correct = (eval_untrained["Status"] == "Correct").sum()
        untrained_acc = (untrained_correct / len(tokenizer.pairs)) * 100.0
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center; padding: 0.9rem;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Initial Accuracy</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #FBBF24;">{untrained_acc:.1f}% ({untrained_correct}/6)</div>
                <div class="metric-badge badge-amber">Random Guessing</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Initial Probability Matrix
    st.markdown("### 📊 Initial Next-Word Softmax Probability Matrix (%)")
    st.caption("Each row represents one input token; columns represent the probability distribution over all 7 vocabulary candidates.")
    
    untrained_prob_df = lab_model.get_full_probability_matrix(untrained_model, tokenizer, X_tensor)
    st.dataframe(untrained_prob_df, use_container_width=True)

    # Visualizations
    tab_bar, tab_heat = st.tabs(["📊 Interactive Grouped Bar Chart", "🔥 Probability Heatmap"])
    with tab_bar:
        fig_untrained = render_probability_bars(
            untrained_prob_df,
            "Untrained Model: Probability of Each Vocabulary Token per Input Bigram"
        )
        st.plotly_chart(fig_untrained, use_container_width=True)

    with tab_heat:
        fig_heat_untrained = render_probability_heatmap(
            untrained_prob_df,
            "Untrained Initial Transition Probability Heatmap (6 Inputs × 7 Vocab Words)"
        )
        st.plotly_chart(fig_heat_untrained, use_container_width=True)

    # Initial Prediction Breakdown Table
    st.markdown("### 📋 Pre-Training Prediction Table")
    st.dataframe(eval_untrained, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# MODULE 5: TRAINING ENGINE (BATCH GRADIENT DESCENT)
# ---------------------------------------------------------
elif selected_module == "5. Training Engine (Batch Gradient Descent)":
    st.markdown("## 🚀 5. Training Engine: Shared Batch Optimization")
    st.markdown(
        """
        Train the shared neural network on **all six pairs simultaneously** in a single batch.
        Adjust the training hyperparameters in the sidebar or below, then click **Train Model**.
        """
    )

    col_ctrl1, col_ctrl2 = st.columns([1.2, 1])

    with col_ctrl1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-header">
                    <span>⚙️ Current Training Configuration</span>
                </div>
                <div style="font-size: 0.9rem; line-height: 1.6; color: #CBD5E1;">
                    • <b>Sentence:</b> "{lab_model.TRAINING_SENTENCE}"<br>
                    • <b>Batch Size:</b> 6 (all pairs processed simultaneously)<br>
                    • <b>Epochs:</b> {st.session_state.training_epochs}<br>
                    • <b>Learning Rate:</b> {st.session_state.training_lr}<br>
                    • <b>Optimizer:</b> Adam<br>
                    • <b>Loss Function:</b> Cross-Entropy Loss
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_ctrl2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-header">
                    <span>⚡ Run Real Training</span>
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.8rem;">
                    Clicking will execute a real PyTorch training loop across all 6 bigrams simultaneously.
                </div>
            """,
            unsafe_allow_html=True
        )
        train_btn = st.button("🔥 Train Model Now", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if train_btn:
        progress_bar = st.progress(0, text="Initializing shared batch tensors...")
        status_placeholder = st.empty()

        # Create fresh clone of model with deterministic seed
        active_model = lab_model.clone_model(st.session_state.untrained_model, seed=st.session_state.random_seed)

        def progress_cb(epoch, total_epochs, current_loss, current_acc):
            frac = epoch / total_epochs
            progress_bar.progress(frac, text=f"Epoch {epoch}/{total_epochs} — Loss: {current_loss:.4f} — Accuracy: {current_acc*100:.1f}%")

        history, final_acc = lab_model.train_shared_model(
            active_model,
            X_tensor,
            Y_tensor,
            epochs=st.session_state.training_epochs,
            lr=st.session_state.training_lr,
            progress_callback=progress_cb
        )

        st.session_state.trained_model = active_model
        st.session_state.training_history = history
        st.session_state.final_accuracy = final_acc
        st.session_state.is_trained = True
        progress_bar.progress(1.0, text="Training complete! 100% convergence achieved.")
        st.success(f"Successfully trained! Final Cross-Entropy Loss: {history[-1]['loss']} | Accuracy: {final_acc * 100:.1f}%")

    # If already trained, display the live loss and accuracy curves
    if st.session_state.is_trained and st.session_state.training_history:
        st.markdown("### 📈 Training Convergence Metrics")
        
        hist_df = pd.DataFrame(st.session_state.training_history)
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=hist_df["epoch"],
                y=hist_df["loss"],
                mode="lines",
                name="Batch Cross-Entropy Loss",
                line=dict(color="#38BDF8", width=2.5)
            ))
            # Theoretical baseline line
            fig_loss.add_hline(
                y=np.log(tokenizer.vocab_size),
                line_dash="dash",
                line_color="#F87171",
                annotation_text="Expected Random Guess Loss (ln 7 ≈ 1.946)",
                annotation_position="top right"
            )
            fig_loss.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Cross-Entropy Loss"
            )
            apply_dark_theme(fig_loss, title="Training Loss vs. Epochs")
            st.plotly_chart(fig_loss, use_container_width=True)

        with col_g2:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=hist_df["epoch"],
                y=hist_df["accuracy"],
                mode="lines",
                name="Prediction Accuracy (%)",
                line=dict(color="#34D399", width=2.5)
            ))
            fig_acc.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Accuracy (%)"
            )
            apply_dark_theme(fig_acc, title="Training Accuracy vs. Epochs", y_range=[0, 105])
            st.plotly_chart(fig_acc, use_container_width=True)
            
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 0.9rem;">
                <div style="font-size: 0.9rem; color: #CBD5E1;">
                    🎯 <b>Convergence Summary:</b> 
                    Starting loss decreased from <b>{hist_df['loss'].iloc[0]:.4f}</b> to <b>{hist_df['loss'].iloc[-1]:.5f}</b> 
                    in {len(hist_df)} epochs. All 6 target tokens are now classified with absolute certainty.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif not st.session_state.is_trained:
        st.info("💡 Model is currently in its untrained baseline state. Click **'Train Model Now'** above to run batch optimization.")


# ---------------------------------------------------------
# MODULE 6: AFTER TRAINING (MODEL EVALUATION)
# ---------------------------------------------------------
elif selected_module == "6. After Training (Model Evaluation)":
    st.markdown("## 🎯 6. After Training: Model Evaluation")
    st.markdown(
        """
        Evaluate the trained neural network on all six bigram pairs. 
        Observe how the shared parameters have specialized to assign near 100% confidence to the true ground-truth targets.
        """
    )

    if not st.session_state.is_trained:
        st.warning("⚠️ The model has not been trained yet! Please visit **Module 5: Training Engine** or click below.")
        if st.button("🚀 Quick Train with Defaults (250 Epochs)", type="primary"):
            active_model = lab_model.clone_model(st.session_state.untrained_model, seed=st.session_state.random_seed)
            history, final_acc = lab_model.train_shared_model(
                active_model, X_tensor, Y_tensor, epochs=250, lr=0.05
            )
            st.session_state.trained_model = active_model
            st.session_state.training_history = history
            st.session_state.final_accuracy = final_acc
            st.session_state.is_trained = True
            st.rerun()
    else:
        trained_model = st.session_state.trained_model
        eval_df = lab_model.get_prediction_analysis(trained_model, tokenizer, X_tensor, Y_tensor)
        trained_prob_df = lab_model.get_full_probability_matrix(trained_model, tokenizer, X_tensor)

        # Accuracy & Loss metric banners
        final_loss = lab_model.compute_loss(trained_model, X_tensor, Y_tensor)
        correct_count = (eval_df["Status"] == "Correct").sum()
        total_count = len(eval_df)
        acc_pct = (correct_count / total_count) * 100.0

        c_eval1, c_eval2, c_eval3 = st.columns(3)
        with c_eval1:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center; padding: 0.9rem;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Final Training Loss</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #34D399;">{final_loss:.5f}</div>
                    <div class="metric-badge badge-green">Converged</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_eval2:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center; padding: 0.9rem;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Training Prediction Accuracy</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #38BDF8;">{acc_pct:.1f}%</div>
                    <div class="metric-badge badge-blue">{correct_count}/{total_count} Pairs Correct</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_eval3:
            avg_conf = eval_df["Target Prob (%)"].mean()
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center; padding: 0.9rem;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Average Target Probability</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #A855F7;">{avg_conf:.1f}%</div>
                    <div class="metric-badge badge-purple">High Target Confidence</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Tabular evaluation
        st.markdown("### 📋 Prediction vs Target Comparison")
        st.dataframe(eval_df, use_container_width=True, hide_index=True)

        # Updated Softmax Probabilities
        st.markdown("### 📊 Updated Post-Training Softmax Probability Matrix (%)")
        st.dataframe(trained_prob_df, use_container_width=True)

        tab_post_bar, tab_post_heat = st.tabs(["📊 Post-Training Bar Chart", "🔥 Post-Training Heatmap"])
        with tab_post_bar:
            fig_trained = render_probability_bars(
                trained_prob_df,
                "Trained Model: Probability of Each Vocabulary Token per Input Bigram"
            )
            st.plotly_chart(fig_trained, use_container_width=True)

        with tab_post_heat:
            fig_heat_trained = render_probability_heatmap(
                trained_prob_df,
                "Trained Transition Probability Heatmap (Sharp Diagonal Alignment)"
            )
            st.plotly_chart(fig_heat_trained, use_container_width=True)


# ---------------------------------------------------------
# MODULE 7: LIVE AUTOREGRESSIVE GENERATION
# ---------------------------------------------------------
elif selected_module == "7. Live Autoregressive Generation":
    st.markdown("## 🔮 7. Live Autoregressive Generation")
    st.markdown(
        """
        Test the trained neural language model in a live generative setting!
        Enter or choose a prompt token (e.g., **"The"**). The model will iteratively predict the next token,
        append it to the sequence, and feed it back into the shared network autoregressively.
        """
    )

    active_model = st.session_state.trained_model if st.session_state.is_trained else st.session_state.untrained_model
    model_type_label = "TRAINED MODEL" if st.session_state.is_trained else "UNTRAINED MODEL (Random Output)"

    st.markdown(
        f"""
        <div class="glass-card" style="padding: 0.8rem 1.2rem; margin-bottom: 1rem;">
            <span style="font-size: 0.85rem; color: #94A3B8;">Generator Engine Source: </span>
            <span class="metric-badge {'badge-green' if st.session_state.is_trained else 'badge-amber'}">
                ● {model_type_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_inp1, col_inp2, col_inp3 = st.columns([1.2, 1, 1])

    with col_inp1:
        start_word = st.selectbox(
            "Select Prompt Word:",
            tokenizer.tokens,
            index=0,
            help="Choose the initial word to start autoregressive generation."
        )

    with col_inp2:
        decoding_strategy = st.radio(
            "Decoding Strategy:",
            ["Greedy (Argmax)", "Temperature Sampling"],
            horizontal=True
        )

    with col_inp3:
        if decoding_strategy == "Temperature Sampling":
            temp_val = st.slider("Sampling Temperature:", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
        else:
            temp_val = 1.0
            st.caption("Greedy mode deterministically chooses the token with highest probability.")

    max_steps_val = st.slider("Maximum Generation Steps:", min_value=2, max_value=12, value=7)

    if st.button("✨ Generate Next Words", type="primary", use_container_width=True):
        greedy_flag = (decoding_strategy == "Greedy (Argmax)")
        gen_result = lab_model.generate_autoregressive(
            active_model,
            tokenizer,
            start_token=start_word,
            max_tokens=max_steps_val,
            temperature=temp_val,
            greedy=greedy_flag
        )

        st.markdown("### 📝 Assembled Output Sentence")
        st.markdown(
            f"""
            <div class="glass-card" style="border: 2px solid #8B5CF6; background: #131A2E;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #38BDF8;">
                    "{gen_result['generated_sentence']}"
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.5rem;">
                    <b>Stop Reason:</b> {gen_result['stopped_reason']} | <b>Generated Tokens Count:</b> {len(gen_result['generated_tokens'])}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 🔍 Step-by-Step Autoregressive Trace")
        if not gen_result["steps"]:
            st.info(f"Generation stopped immediately: {gen_result['stopped_reason']}")
        else:
            for step in gen_result["steps"]:
                st.markdown(
                    f"""
                    <div class="step-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                            <span style="font-weight: 700; color: #C084FC;">Step {step['step']}</span>
                            <span class="metric-badge badge-blue">Confidence: {step['confidence']}%</span>
                        </div>
                        <div style="font-size: 0.95rem; color: #F1F5F9;">
                            Current Input: <b style="color: #38BDF8;">"{step['input_token']}"</b> (ID: {step['input_id']}) 
                            ➔ Predicted Next: <b style="color: #34D399;">"{step['predicted_token']}"</b> (ID: {step['predicted_id']})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Show miniature bar chart of distribution for this step
                step_dist_df = pd.DataFrame([step["distribution"]]).T.reset_index()
                step_dist_df.columns = ["Token", "Probability (%)"]
                
                fig_step = px.bar(
                    step_dist_df,
                    x="Token",
                    y="Probability (%)",
                    color="Token",
                    title=f"Step {step['step']} Softmax Distribution for Input '{step['input_token']}'",
                    color_discrete_sequence=["#38BDF8", "#818CF8", "#A78BFA", "#C084FC", "#F472B6", "#FB7185", "#34D399"]
                )
                fig_step.update_layout(
                    height=260,
                    showlegend=False
                )
                apply_dark_theme(fig_step, y_range=[0, 105])
                st.plotly_chart(fig_step, use_container_width=True)


# ---------------------------------------------------------
# MODULE 8: COMPARATIVE VISUALIZATIONS
# ---------------------------------------------------------
elif selected_module == "8. Comparative Visualizations":
    st.markdown("## 📈 8. Comparative Visualizations")
    st.markdown(
        """
        Directly compare the **Pre-Training Baseline** versus the **Post-Training Optimization**.
        Inspect how the probability of each correct target token climbs toward 100% through shared gradient updates.
        """
    )

    untrained_model = st.session_state.untrained_model
    trained_model = st.session_state.trained_model if st.session_state.is_trained else untrained_model

    pre_eval = lab_model.get_prediction_analysis(untrained_model, tokenizer, X_tensor, Y_tensor)
    post_eval = lab_model.get_prediction_analysis(trained_model, tokenizer, X_tensor, Y_tensor)

    # Comparison DataFrame
    comp_df = pd.DataFrame({
        "Pair #": pre_eval["Pair #"],
        "Input": pre_eval["Input Token"],
        "Target": pre_eval["Target Token"],
        "Before Prob (%)": pre_eval["Target Prob (%)"],
        "After Prob (%)": post_eval["Target Prob (%)"],
        "Gain (% pts)": post_eval["Target Prob (%)"] - pre_eval["Target Prob (%)"],
        "Initial Status": pre_eval["Status"],
        "Final Status": post_eval["Status"] if st.session_state.is_trained else "Untrained"
    })

    st.markdown("### 🎯 Target Probability Evolution (Before vs After)")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Plotly Grouped Bar Chart of Target Probabilities
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=[f"Pair {p}: {inp}➔{tgt}" for p, inp, tgt in zip(comp_df["Pair #"], comp_df["Input"], comp_df["Target"])],
        y=comp_df["Before Prob (%)"],
        name="Before Training (Baseline)",
        marker_color="#F87171"
    ))
    fig_comp.add_trace(go.Bar(
        x=[f"Pair {p}: {inp}➔{tgt}" for p, inp, tgt in zip(comp_df["Pair #"], comp_df["Input"], comp_df["Target"])],
        y=comp_df["After Prob (%)"],
        name="After Training (Shared Optimization)",
        marker_color="#34D399"
    ))
    fig_comp.update_layout(
        xaxis_title="Training Bigram Transition",
        yaxis_title="Target Probability (%)",
        barmode="group"
    )
    apply_dark_theme(fig_comp, title="Target Token Probability Comparison per Bigram Pair", y_range=[0, 105])
    st.plotly_chart(fig_comp, use_container_width=True)

    # Dual Heatmaps Side by Side
    st.markdown("### 🔥 Dual Heatmap Comparison")
    untrained_prob_df = lab_model.get_full_probability_matrix(untrained_model, tokenizer, X_tensor)
    trained_prob_df = lab_model.get_full_probability_matrix(trained_model, tokenizer, X_tensor)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.plotly_chart(
            render_probability_heatmap(untrained_prob_df, "Untrained Probability Matrix (%)"),
            use_container_width=True
        )
    with col_h2:
        st.plotly_chart(
            render_probability_heatmap(trained_prob_df, "Trained Probability Matrix (%)"),
            use_container_width=True
        )


# ---------------------------------------------------------
# MODULE 9: DOWNLOADABLE RESULTS & DATA
# ---------------------------------------------------------
elif selected_module == "9. Downloadable Results & Data":
    st.markdown("## 💾 9. Downloadable Results & Data")
    st.markdown(
        """
        Export all datasets, transition matrices, training histories, and performance metrics
        as standard CSV files for offline analysis or laboratory reporting.
        """
    )

    untrained_model = st.session_state.untrained_model
    trained_model = st.session_state.trained_model if st.session_state.is_trained else untrained_model

    # 1. Training pairs dataset
    pairs_df = tokenizer.get_pairs_dataframe()
    
    # 2. Probability matrices
    pre_probs_df = lab_model.get_full_probability_matrix(untrained_model, tokenizer, X_tensor)
    post_probs_df = lab_model.get_full_probability_matrix(trained_model, tokenizer, X_tensor)
    
    # 3. Prediction Analysis
    analysis_df = lab_model.get_prediction_analysis(trained_model, tokenizer, X_tensor, Y_tensor)
    
    # 4. Training History
    hist_df = pd.DataFrame(st.session_state.training_history) if st.session_state.training_history else pd.DataFrame([{"epoch": 0, "loss": np.nan, "accuracy": np.nan}])

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-header">
                    <span>📄 Bigram Training Pairs CSV</span>
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.8rem;">
                    Contains all 6 token transitions, discrete IDs, and sequential positions.
                </div>
            """,
            unsafe_allow_html=True
        )
        csv_pairs = pairs_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download training_pairs.csv",
            data=csv_pairs,
            file_name="training_pairs.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_d2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-header">
                    <span>📄 Model Predictions & Evaluation CSV</span>
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.8rem;">
                    Detailed accuracy, top predicted tokens, and target probability comparisons.
                </div>
            """,
            unsafe_allow_html=True
        )
        csv_analysis = analysis_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download prediction_evaluation.csv",
            data=csv_analysis,
            file_name="prediction_evaluation.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    col_d3, col_d4 = st.columns(2)

    with col_d3:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-header">
                    <span>📄 Probability Matrices CSV</span>
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.8rem;">
                    Full 6×7 softmax probability distributions for both untrained and trained states.
                </div>
            """,
            unsafe_allow_html=True
        )
        # Combined export
        combined_prob = pd.concat({"Untrained": pre_probs_df, "Trained": post_probs_df}, axis=1)
        csv_probs = combined_prob.to_csv().encode('utf-8')
        st.download_button(
            label="⬇️ Download full_probabilities.csv",
            data=csv_probs,
            file_name="full_probabilities.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_d4:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-header">
                    <span>📄 Training Loss & Accuracy History CSV</span>
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.8rem;">
                    Epoch-by-epoch loss reduction trajectory and convergence progression.
                </div>
            """,
            unsafe_allow_html=True
        )
        csv_hist = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download training_history.csv",
            data=csv_hist,
            file_name="training_history.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Preview Tables
    st.markdown("### 🔍 Live Dataset Previews")
    preview_tab1, preview_tab2, preview_tab3 = st.tabs(["Bigram Pairs", "Prediction Report", "Loss History"])
    with preview_tab1:
        st.dataframe(pairs_df, use_container_width=True)
    with preview_tab2:
        st.dataframe(analysis_df, use_container_width=True)
    with preview_tab3:
        st.dataframe(hist_df.head(20), use_container_width=True)
