"""
Shared Computation AI Lab - Neural Network & Tokenization Engine
Demonstrates shared computation for next-word prediction using one training sentence:
"The student studies AI and builds projects."
"""

import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAINING_SENTENCE = "The student studies AI and builds projects."
DEFAULT_TOKENS = ["The", "student", "studies", "AI", "and", "builds", "projects"]
DEFAULT_RANDOM_SEED = 42


def set_seed(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """Set deterministic random seeds across PyTorch and NumPy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SharedTokenizer:
    """
    Tokenizer and Dataset Builder for the 7-token sentence.
    Maps words to discrete IDs and builds sequential input-target training pairs.
    """

    def __init__(self, sentence: str = TRAINING_SENTENCE, tokens: Optional[List[str]] = None):
        self.sentence = sentence.strip()
        if tokens is not None:
            self.tokens = tokens
        else:
            # Clean punctuation and tokenize preserving exact casing
            cleaned = self.sentence.replace(".", "").replace(",", "").replace("!", "").replace("?", "")
            self.tokens = cleaned.split()

        self.vocab_size = len(self.tokens)
        self.token_to_id: Dict[str, int] = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token: Dict[int, str] = {idx: token for idx, token in enumerate(self.tokens)}

        # Build 6 sequential bigram pairs
        self.pairs: List[Tuple[str, str]] = [
            (self.tokens[i], self.tokens[i + 1]) for i in range(len(self.tokens) - 1)
        ]

    def encode(self, token: str) -> int:
        if token not in self.token_to_id:
            raise KeyError(f"Token '{token}' not found in vocabulary {list(self.token_to_id.keys())}")
        return self.token_to_id[token]

    def decode(self, token_id: int) -> str:
        if token_id not in self.id_to_token:
            raise KeyError(f"Token ID {token_id} not found in vocabulary")
        return self.id_to_token[token_id]

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns input tensor X (shape: [6]) and target tensor Y (shape: [6]).
        Processed as a single batch for shared computation.
        """
        x_ids = [self.encode(inp) for inp, _ in self.pairs]
        y_ids = [self.encode(tgt) for _, tgt in self.pairs]
        return torch.tensor(x_ids, dtype=torch.long), torch.tensor(y_ids, dtype=torch.long)

    def get_pairs_dataframe(self) -> pd.DataFrame:
        """Tabular view of all input-target training pairs."""
        rows = []
        for i, (inp, tgt) in enumerate(self.pairs):
            rows.append({
                "Pair #": i + 1,
                "Input Token": inp,
                "Input ID": self.encode(inp),
                "Arrow": "➔",
                "Target Token": tgt,
                "Target ID": self.encode(tgt)
            })
        return pd.DataFrame(rows)

    def get_vocabulary_dataframe(self) -> pd.DataFrame:
        """Tabular view of the vocabulary."""
        return pd.DataFrame([
            {"Token ID": idx, "Token": token, "Character Length": len(token)}
            for idx, token in self.id_to_token.items()
        ])


class SharedNextWordModel(nn.Module):
    """
    PyTorch Next-Word Prediction Model with Shared Computation.
    
    Architecture:
    - Shared Embedding Layer: maps token ID -> continuous embedding vector of dimension d
    - Shared Linear Output Layer: maps embedding vector -> vocab_size logits
    
    Both layers share weights across all sequence positions and training pairs.
    All 6 pairs are evaluated simultaneously in a single forward pass.
    """

    def __init__(self, vocab_size: int = 7, embedding_dim: int = 16, seed: int = DEFAULT_RANDOM_SEED):
        super().__init__()
        set_seed(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Shared embedding layer: vocab_size x embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Shared linear classification layer: embedding_dim x vocab_size
        self.linear = nn.Linear(embedding_dim, vocab_size)

        # Initialize weights with a small normal distribution for deterministic, clean starting state
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.2)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.2)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of token IDs.
        Input x: shape [batch_size]
        Embedding output: shape [batch_size, embedding_dim]
        Logits output: shape [batch_size, vocab_size]
        """
        emb = self.embedding(x)          # [B, embedding_dim]
        logits = self.linear(emb)        # [B, vocab_size]
        return logits

    def get_probabilities(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns raw logits and softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return logits, probs

    def count_parameters(self) -> Dict[str, Any]:
        """Computes parameter shapes and total trainable parameter count."""
        emb_params = self.embedding.weight.numel()
        linear_w_params = self.linear.weight.numel()
        linear_b_params = self.linear.bias.numel()
        total_params = emb_params + linear_w_params + linear_b_params

        return {
            "embedding_shape": list(self.embedding.weight.shape),
            "embedding_params": emb_params,
            "linear_w_shape": list(self.linear.weight.shape),
            "linear_w_params": linear_w_params,
            "linear_b_shape": list(self.linear.bias.shape),
            "linear_b_params": linear_b_params,
            "total_params": total_params
        }


def clone_model(source_model: SharedNextWordModel, seed: int = DEFAULT_RANDOM_SEED) -> SharedNextWordModel:
    """Creates a fresh untrained model with identical architecture and seed."""
    new_model = SharedNextWordModel(
        vocab_size=source_model.vocab_size,
        embedding_dim=source_model.embedding_dim,
        seed=seed
    )
    return new_model


def compute_loss(model: SharedNextWordModel, x: torch.Tensor, y: torch.Tensor) -> float:
    """Computes cross-entropy loss without gradient computation."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y)
    return float(loss.item())


def train_shared_model(
    model: SharedNextWordModel,
    X: torch.Tensor,
    Y: torch.Tensor,
    epochs: int = 250,
    lr: float = 0.05,
    progress_callback: Optional[Any] = None
) -> Tuple[List[Dict[str, float]], float]:
    """
    Trains the shared model over the entire batch of 6 pairs simultaneously.
    
    Args:
        model: SharedNextWordModel instance
        X: Tensor of input token IDs [6]
        Y: Tensor of target token IDs [6]
        epochs: Number of training iterations
        lr: Learning rate for Adam optimizer
        progress_callback: Optional callable(epoch, loss, accuracy)
        
    Returns:
        history: List of dicts [{'epoch': e, 'loss': float, 'accuracy': float}]
        final_accuracy: Accuracy on the 6 training pairs (0.0 to 1.0)
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(X)                # Single shared forward pass for all 6 inputs
        loss = criterion(logits, Y)      # Batch loss across all 6 pairs
        loss.backward()                  # Shared gradient computation
        optimizer.step()                 # Update shared parameters

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == Y).sum().item()
            acc = correct / len(Y)

        history.append({
            "epoch": epoch,
            "loss": round(float(loss.item()), 5),
            "accuracy": round(acc * 100.0, 2)
        })

        if progress_callback is not None and (epoch % max(1, epochs // 50) == 0 or epoch == epochs):
            progress_callback(epoch, epochs, float(loss.item()), acc)

    model.eval()
    final_acc = history[-1]["accuracy"] / 100.0
    return history, final_acc


def get_prediction_analysis(
    model: SharedNextWordModel,
    tokenizer: SharedTokenizer,
    X: torch.Tensor,
    Y: torch.Tensor
) -> pd.DataFrame:
    """
    Generates a comprehensive analysis DataFrame comparing targets and model predictions.
    Computes real softmax probabilities from the model.
    """
    _, probs = model.get_probabilities(X)
    probs_np = probs.cpu().numpy()
    
    rows = []
    for i, (inp_token, target_token) in enumerate(tokenizer.pairs):
        inp_id = X[i].item()
        tgt_id = Y[i].item()
        
        prob_dist = probs_np[i]
        pred_id = int(np.argmax(prob_dist))
        pred_token = tokenizer.decode(pred_id)
        
        target_prob = float(prob_dist[tgt_id])
        pred_prob = float(prob_dist[pred_id])
        is_correct = (pred_id == tgt_id)

        rows.append({
            "Pair #": i + 1,
            "Input Token": inp_token,
            "Input ID": inp_id,
            "Target Token": target_token,
            "Target ID": tgt_id,
            "Predicted Token": pred_token,
            "Predicted ID": pred_id,
            "Target Prob (%)": round(target_prob * 100.0, 2),
            "Top Prob (%)": round(pred_prob * 100.0, 2),
            "Status": "Correct" if is_correct else "Incorrect"
        })

    return pd.DataFrame(rows)


def get_full_probability_matrix(
    model: SharedNextWordModel,
    tokenizer: SharedTokenizer,
    X: torch.Tensor
) -> pd.DataFrame:
    """
    Returns a DataFrame of shape [num_pairs, vocab_size] containing softmax probabilities.
    Index: Input Tokens
    Columns: Vocabulary Tokens
    """
    _, probs = model.get_probabilities(X)
    probs_np = probs.cpu().numpy() * 100.0  # as percentages

    col_names = [f"{token} (#{idx})" for idx, token in enumerate(tokenizer.tokens)]
    row_names = [f"{pair[0]} (Pair #{i+1})" for i, pair in enumerate(tokenizer.pairs)]

    df = pd.DataFrame(probs_np, index=row_names, columns=col_names)
    return df.round(2)


def generate_autoregressive(
    model: SharedNextWordModel,
    tokenizer: SharedTokenizer,
    start_token: str,
    max_tokens: int = 7,
    temperature: float = 1.0,
    greedy: bool = True
) -> Dict[str, Any]:
    """
    Autoregressively generates text token-by-token from a starting prompt word.
    
    Returns:
        Dict with:
        - generated_tokens: List[str]
        - generated_sentence: str
        - steps: List[Dict] with step-by-step distributions and selections
        - stopped_reason: reason why generation completed
    """
    model.eval()
    
    # Handle casing gracefully: match exact token or case-insensitive fallback
    matched_token = None
    for token in tokenizer.tokens:
        if token.lower() == start_token.strip().lower():
            matched_token = token
            break
            
    if matched_token is None:
        return {
            "error": f"Token '{start_token}' is not in the vocabulary: {tokenizer.tokens}",
            "generated_tokens": [],
            "generated_sentence": "",
            "steps": [],
            "stopped_reason": "Out of Vocabulary"
        }

    current_token = matched_token
    generated_tokens = [current_token]
    steps: List[Dict[str, Any]] = []
    stopped_reason = f"Reached max length ({max_tokens})"

    for step_idx in range(1, max_tokens):
        # Stop condition: terminal token of training sentence
        if current_token == tokenizer.tokens[-1]:
            stopped_reason = f"Reached terminal sequence token ('{current_token}')"
            break

        curr_id = tokenizer.encode(current_token)
        curr_tensor = torch.tensor([curr_id], dtype=torch.long)

        with torch.no_grad():
            logits = model(curr_tensor)  # [1, vocab_size]
            
            # Apply temperature
            temp = max(1e-4, temperature)
            scaled_logits = logits / temp
            probs = F.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()

        if greedy:
            next_id = int(np.argmax(probs))
        else:
            # Sample from categorical distribution
            next_id = int(np.random.choice(len(probs), p=probs))

        next_token = tokenizer.decode(next_id)
        confidence = float(probs[next_id]) * 100.0

        # Save step breakdown
        distribution = {
            tokenizer.decode(v_id): round(float(probs[v_id]) * 100.0, 2)
            for v_id in range(tokenizer.vocab_size)
        }

        steps.append({
            "step": step_idx,
            "input_token": current_token,
            "input_id": curr_id,
            "predicted_token": next_token,
            "predicted_id": next_id,
            "confidence": round(confidence, 2),
            "distribution": distribution
        })

        generated_tokens.append(next_token)
        current_token = next_token

        # Loop detector (if the same 2-word cycle repeats 3 times)
        if len(generated_tokens) >= 4 and generated_tokens[-1] == generated_tokens[-3] and generated_tokens[-2] == generated_tokens[-4]:
            stopped_reason = "Repetition loop detected"
            break

    # Build final sentence with clean punctuation if ending with projects
    final_sentence = " ".join(generated_tokens)
    if generated_tokens[-1] == "projects":
        final_sentence += "."

    return {
        "generated_tokens": generated_tokens,
        "generated_sentence": final_sentence,
        "steps": steps,
        "stopped_reason": stopped_reason
    }
