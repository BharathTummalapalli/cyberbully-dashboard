import re
import random
import os

# Try importing ML dependencies. Fallback to mock mode if not present or on low-resource machines.
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from lime.lime_text import LimeTextExplainer
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# List of toxic keywords for the fallback mock engine
TOXIC_KEYWORDS = {
    "trash": 0.72,
    "loser": 0.85,
    "ugly": 0.81,
    "stupid": 0.68,
    "idiot": 0.79,
    "fat": 0.74,
    "gross": 0.65,
    "worthless": 0.88,
    "brainless": 0.76,
    "troll": 0.55,
    "annoying": 0.58,
    "hateful": 0.70,
    "jerk": 0.62,
    "pathetic": 0.82,
    "hate": 0.67,
    "banned": 0.52,
    "toxic": 0.69,
    "parasite": 0.80,
    "worst": 0.64,
    "delete": 0.48,
    "kill": 0.92,
    "disappointment": 0.71,
    "garbage": 0.59,
    "shame": 0.50
}

# Global references for ML models
_nlp_pipeline = None
_explainer = None

def init_ml_model():
    """Initializes the HuggingFace model and LIME explainer if available."""
    global _nlp_pipeline, _explainer
    if not ML_AVAILABLE:
        print("[-] ML dependencies missing. Backend will run in Fast Demo Mode.")
        return False
    
    try:
        # Using a specialized model for toxicity/toxic comments classification
        # 'unitary/toxic-bert' is the industry standard for this task
        model_name = "unitary/toxic-bert"
        print(f"[+] Loading HuggingFace model: {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # CPU/GPU mapping
        device = 0 if torch.cuda.is_available() else -1
        _nlp_pipeline = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer, 
            device=device,
            top_k=None # returns scores for all labels
        )
        
        _explainer = LimeTextExplainer(class_names=["Safe", "Cyberbullying"])
        print("[+] HuggingFace Model and LIME Explainer loaded successfully!")
        return True
    except Exception as e:
        print(f"[-] Failed to load HuggingFace model: {e}. Falling back to Fast Demo Mode.")
        return False

def get_mock_lime_explanation(text):
    """
    Simulates LIME text perturbations instantly.
    Analyzes keywords, calculates confidence, highlights toxic tokens, and maps weights.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    highlighted = []
    lime_weights = {}
    
    # Identify matching toxic words and pull their predefined weights
    matched_toxic = []
    for word in words:
        if word in TOXIC_KEYWORDS:
            matched_toxic.append((word, TOXIC_KEYWORDS[word]))
            
    # Remove duplicate matching words
    matched_toxic = list(set(matched_toxic))
    
    # Calculate confidence score
    if matched_toxic:
        prediction = "Cyberbullying"
        # Base confidence starts at 70%, increases with toxic words, capped at 99.0%
        raw_conf = 70.0 + sum([w[1] * 10 for w in matched_toxic])
        confidence = min(raw_conf, 99.0)
        
        # Sort matched toxic words by their weight
        matched_toxic.sort(key=lambda x: x[1], reverse=True)
        for word, weight in matched_toxic[:5]:
            lime_weights[word] = round(weight, 3)
            highlighted.append(word)
            
        # Add a couple of neutral words to resemble true perturbation lists
        neutral_words = [w for w in words if w not in TOXIC_KEYWORDS]
        random.shuffle(neutral_words)
        for w in neutral_words[:min(2, len(neutral_words))]:
            lime_weights[w] = round(random.uniform(0.01, 0.08), 3)
    else:
        prediction = "Safe"
        confidence = round(random.uniform(85.0, 96.5), 1)
        
        # Safe posts: select words that contribute to 'Safe' classification (negative toxicity weight)
        neutral_words = list(set(words))
        random.shuffle(neutral_words)
        for w in neutral_words[:min(3, len(neutral_words))]:
            lime_weights[w] = round(random.uniform(-0.15, -0.01), 3)
            
    # Generate user-friendly explanation strings
    explanations = []
    for word, weight in lime_weights.items():
        if weight > 0:
            explanations.append(f"Word '{word}' increased cyberbullying score by {int(weight * 100)}%.")
        else:
            explanations.append(f"Word '{word}' supported Safe classification by {int(abs(weight) * 100)}%.")
            
    return prediction, confidence, explanations, highlighted, lime_weights

def get_lime_explanation(text, use_real_model=False):
    """
    Main entry point for explainability.
    If use_real_model is True and dependencies are loaded, performs real LIME perturbation.
    Otherwise, defaults to fast mock explanation.
    """
    global _nlp_pipeline, _explainer
    
    # Use real model if requested, available, and initialized
    if use_real_model and ML_AVAILABLE:
        if _nlp_pipeline is None:
            initialized = init_ml_model()
            if not initialized:
                return get_mock_lime_explanation(text)
        
        try:
            # Define predictor function for LIME.
            # LIME requires a function that accepts a list of texts and returns a 2D numpy array of probabilities [P(Safe), P(Bullying)]
            def predictor_fn(texts):
                pipeline_outputs = _nlp_pipeline(texts)
                probs = []
                for output in pipeline_outputs:
                    # 'toxic-bert' outputs labels: 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
                    # We map this to overall toxicity vs safety.
                    # 'toxic' is the primary classifier label.
                    toxic_score = 0.0
                    for label_dict in output:
                        if label_dict['label'] == 'toxic':
                            toxic_score = label_dict['score']
                            break
                    # If toxic_score is high, it is cyberbullying.
                    # Probability array: [Safe, Cyberbullying]
                    probs.append([1.0 - toxic_score, toxic_score])
                return np.array(probs)

            # Generate LIME explanation (using 100 perturbations for speed on CPU)
            exp = _explainer.explain_instance(
                text, 
                predictor_fn, 
                num_features=5, 
                num_samples=100
            )
            
            # Extract results
            # exp.as_list() returns list of tuples: (word, weight)
            explanation_list = exp.as_list()
            
            # Predict overall label
            # Get prediction for single input text
            probs = predictor_fn([text])[0]
            confidence = float(probs[1] * 100) # Bullying probability
            
            if probs[1] >= 0.5:
                prediction = "Cyberbullying"
                confidence = round(confidence, 1)
            else:
                prediction = "Safe"
                confidence = round((1.0 - probs[1]) * 100, 1)

            highlighted = []
            lime_weights = {}
            explanations = []
            
            for word, weight in explanation_list:
                lime_weights[word] = round(float(weight), 3)
                if prediction == "Cyberbullying" and weight > 0.05:
                    highlighted.append(word)
                    explanations.append(f"Word '{word}' increased cyberbullying score by {int(weight * 100)}%.")
                elif prediction == "Safe" and weight < -0.05:
                    explanations.append(f"Word '{word}' supported Safe classification by {int(abs(weight) * 100)}%.")
                else:
                    impact = "increased" if weight > 0 else "decreased"
                    explanations.append(f"Word '{word}' {impact} risk profile by {int(abs(weight) * 100)}%.")

            return prediction, confidence, explanations, highlighted, lime_weights
            
        except Exception as e:
            print(f"[-] Error during real LIME execution: {e}. Falling back to Mock.")
            return get_mock_lime_explanation(text)
            
    # Default fallback
    return get_mock_lime_explanation(text)
