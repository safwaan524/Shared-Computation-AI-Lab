"""
Comprehensive Automated Test Suite for Shared Computation AI Lab
Tests tokenization, model architecture, training convergence, and Streamlit app flow.
"""

import unittest
import torch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import model as lab_model


class TestSharedComputationEngine(unittest.TestCase):
    def setUp(self):
        self.tokenizer = lab_model.SharedTokenizer()
        self.X, self.Y = self.tokenizer.get_tensors()
        self.model = lab_model.SharedNextWordModel(vocab_size=7, embedding_dim=16, seed=42)

    def test_tokenization_exact_match(self):
        expected_tokens = ["The", "student", "studies", "AI", "and", "builds", "projects"]
        self.assertEqual(self.tokenizer.tokens, expected_tokens)
        self.assertEqual(len(self.tokenizer.tokens), 7)
        self.assertEqual(self.tokenizer.vocab_size, 7)

    def test_bigram_pairs_count_and_sequence(self):
        expected_pairs = [
            ("The", "student"),
            ("student", "studies"),
            ("studies", "AI"),
            ("AI", "and"),
            ("and", "builds"),
            ("builds", "projects")
        ]
        self.assertEqual(self.tokenizer.pairs, expected_pairs)
        self.assertEqual(len(self.tokenizer.pairs), 6)

    def test_batch_tensors_shape_and_values(self):
        self.assertEqual(self.X.shape, torch.Size([6]))
        self.assertEqual(self.Y.shape, torch.Size([6]))
        self.assertEqual(self.X.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(self.Y.tolist(), [1, 2, 3, 4, 5, 6])

    def test_model_architecture_and_parameters(self):
        params = self.model.count_parameters()
        self.assertEqual(params["embedding_shape"], [7, 16])
        self.assertEqual(params["linear_w_shape"], [7, 16])
        self.assertEqual(params["linear_b_shape"], [7])
        self.assertEqual(params["total_params"], 112 + 112 + 7)  # 231

    def test_untrained_model_probabilities(self):
        logits, probs = self.model.get_probabilities(self.X)
        self.assertEqual(logits.shape, torch.Size([6, 7]))
        self.assertEqual(probs.shape, torch.Size([6, 7]))
        
        # Softmax rows should sum to 1.0
        prob_sums = probs.sum(dim=-1).numpy()
        np.testing.assert_allclose(prob_sums, np.ones(6), rtol=1e-5)

        # Baseline loss should be around -ln(1/7) ≈ 1.9459
        loss = lab_model.compute_loss(self.model, self.X, self.Y)
        self.assertAlmostEqual(loss, 1.9459, delta=0.5)

    def test_training_convergence_and_100pct_accuracy(self):
        history, final_acc = lab_model.train_shared_model(
            self.model, self.X, self.Y, epochs=250, lr=0.05
        )
        self.assertEqual(final_acc, 1.0)
        self.assertLess(history[-1]["loss"], 0.05)
        
        analysis = lab_model.get_prediction_analysis(self.model, self.tokenizer, self.X, self.Y)
        self.assertEqual(list(analysis["Status"]), ["Correct"] * 6)

    def test_autoregressive_generation_from_start_token(self):
        # Train model first
        lab_model.train_shared_model(self.model, self.X, self.Y, epochs=250, lr=0.05)
        
        gen = lab_model.generate_autoregressive(self.model, self.tokenizer, start_token="The", max_tokens=7)
        self.assertEqual(gen["generated_sentence"], "The student studies AI and builds projects.")
        self.assertEqual(gen["generated_tokens"], ["The", "student", "studies", "AI", "and", "builds", "projects"])
        self.assertEqual(len(gen["steps"]), 6)


class TestStreamlitApp(unittest.TestCase):
    def test_all_sidebar_navigation_modules(self):
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
        
        for mod in modules:
            at = AppTest.from_file("app.py", default_timeout=30)
            at.run()
            at.sidebar.radio[0].set_value(mod).run()
            self.assertFalse(at.exception, f"Module '{mod}' threw exception: {at.exception}")

    def test_streamlit_train_button_and_evaluation(self):
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        # Navigate to training module
        at.sidebar.radio[0].set_value("5. Training Engine (Batch Gradient Descent)").run()
        self.assertFalse(at.exception)
        
        # Click Train button
        train_btn = [btn for btn in at.button if "Train Model Now" in btn.label][0]
        train_btn.click().run()
        self.assertFalse(at.exception)
        self.assertTrue(at.session_state.is_trained)
        self.assertEqual(at.session_state.final_accuracy, 1.0)
        
        # Navigate to After Training
        at.sidebar.radio[0].set_value("6. After Training (Model Evaluation)").run()
        self.assertFalse(at.exception)

        # Navigate to Live Generation and click generate
        at.sidebar.radio[0].set_value("7. Live Autoregressive Generation").run()
        gen_btn = [btn for btn in at.button if "Generate Next Words" in btn.label]
        if gen_btn:
            gen_btn[0].click().run()
            self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main()
