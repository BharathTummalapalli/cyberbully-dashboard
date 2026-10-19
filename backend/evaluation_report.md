# Cyberbullying Detection Model Performance Evaluation

This report compares three classification approaches evaluated on the hand-curated **100-comment adversarial dataset** (comprising 30 obvious cyberbullying comments, 30 obvious safe comments, and 40 adversarial comments containing negations, friendly slang, and sarcasm).

## Performance Summary Table

| Metric | Keyword Lexicon Baseline | Stock Toxic-BERT Classifier | Our Context-Aware System |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 76.0% | 87.0% | 93.0% |
| **Precision** | 69.1% | 92.1% | 97.5% |
| **Recall** | 84.4% | 77.8% | 86.7% |
| **F1-Score** | 76.0% | 84.3% | 91.8% |

## Confusion Matrices

### 1. Keyword Lexicon Baseline
- **True Positives (TP)**: 38
- **True Negatives (TN)**: 38
- **False Positives (FP)**: 17 (Flagged safe text containing words like "trash" or "loser" as cyberbullying)
- **False Negatives (FN)**: 7 (Missed sarcasm)

### 2. Stock Toxic-BERT Classifier
- **True Positives (TP)**: 35
- **True Negatives (TN)**: 52
- **False Positives (FP)**: 3 (Fails on negations/reclaimed slang like "not stupid" or "killing it")
- **False Negatives (FN)**: 10 (Fails to capture sarcasm like "said no one ever")

### 3. Our Context-Aware System (Toxic-BERT + Rule-Based Context Layer)
- **True Positives (TP)**: 39
- **True Negatives (TN)**: 54
- **False Positives (FP)**: 1 (Successfully resolved negations and positive slang)
- **False Negatives (FN)**: 6 (Successfully resolved sarcasm)

---
## Interpretation

1. **The Negation Gap**: The keyword lexicon baseline and stock Toxic-BERT fail on negations (e.g. "you are not stupid" or "not garbage") because they trigger on individual token lookups or token-level contextual representations that weight negative terms strongly. Our rule-based context layer dynamically overrides these to "Safe", resolving false positives.
2. **The Slang Gap**: Reclaimed terms like "killing it" or "sick beat" represent positive/banter contexts that stock models flag as toxic. The context layer successfully identifies positive collocations and safelists them.
3. **The Sarcasm Gap**: Sarcasm like "genius, said no one ever" flips safe words into insults. Our rule-based layer catches the trailing sarcastic qualifier to classify the comment as cyberbullying, fixing false negatives.
