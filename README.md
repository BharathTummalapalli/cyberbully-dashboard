# Context-Aware Cyberbullying Detection & Explainability System

A complete Full-Stack Minimum Viable Product (MVP) for B.Tech final-year project presenting a **Context-Aware Cyberbullying Detection System**. 

The system leverages Natural Language Processing (NLP) to classify social media posts, Local Interpretable Model-agnostic Explanations (LIME) for token explainability, and NetworkX directed graph analysis to trace abuser relationships and group clusters.

---

## 🚀 System Architecture

```
   ┌────────────────────────────────────────────────────────┐
   │                    React.js Frontend                   │
   │   (Vite + Tailwind CSS v4 + Recharts UI components)    │
   └───────────────────────────┬────────────────────────────┘
                               │ HTTP REST Requests
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │                     FastAPI Server                     │
   │       (CORS enabled, routing endpoints & state)        │
   └─────┬─────────────────────┬──────────────────────┬─────┘
         │                     │                      │
         ▼                     ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│    SQLite DB     │  │  LIME Explainer  │  │ NetworkX Graph   │
│ (moderation_logs)│  │ (explain.py path)│  │  (graph.py path) │
└──────────────────┘  └────────┬─────────┘  └──────────────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │    HuggingFace   │
                      │   (toxic-bert)   │
                      └──────────────────┘
```

- **Frontend**: React.js (built on Vite) styled with Tailwind CSS v4, visualizing trends through Recharts and network edges via interactive SVGs.
- **Backend**: FastAPI (Python 3) implementing CORS, model pipelines, and simulator logic.
- **AI/Explainability**: HuggingFace Transformers (`unitary/toxic-bert`) and PyTorch for sentiment categorization. Explainability perturbing uses `LIME`.
- **Graph Engine**: NetworkX to calculate degree centrality indices, clustering coefficients, and weakly connected community components.
- **Database**: SQLite database logging inputs, confidence levels, timestamps, and moderator actions.

---

## 🛠️ Installation & Setup

### Prerequisites
Make sure you have [Python (3.8+)](https://www.python.org/downloads/) and [Node.js (18+)](https://nodejs.org/) installed.

### Step 1: Run the Backend Server
1. Open a terminal and navigate to the project backend folder:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install backend packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   python main.py
   ```
   *The backend will boot on `http://127.0.0.1:8000`. Database tables `moderation.db` will initialize automatically.*

### Step 2: Run the Frontend Server
1. Open a **new** terminal and navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install frontend dependencies:
   ```bash
   npm install
   ```
3. Run the Vite development server:
   ```bash
   npm run dev
   ```
   *The frontend will boot on `http://localhost:5173`. Open this URL in your web browser.*

---

## 🌟 Demo Features Guide (For Project Examiners)

To showcase a stellar presentation to B.Tech project checkers, demonstrate the following flows:

### 1. The Dual-Mode AI Toggle (Fast Demo vs. Deep BERT)
- **Fast Demo Mode (Default)**: Utilizes a lookup lexicon that classifies toxic posts instantly (under 1 millisecond) and mocks LIME attributions. This guarantees a **perfect, zero-lag presentation** even on CPU machines with poor internet connections.
- **Real BERT AI Mode**: In the frontend sidebar, click **"Activate BERT"**. The backend will download and initialize `unitary/toxic-bert` (approx. 400MB weight files) on your machine. Future scans will pass through actual neural network layers and execute real LIME perturbations.

### 2. Live Feed Simulator & Human-in-the-Loop Moderation
- Click **Resume Feed** on the dashboard. The app will pull and scan posts from `tweets.csv` every 3 seconds.
- Review flagged words highlighted in **Glowing Red** (or positive safety words in **Green**).
- Expand **"Why was this flagged? View AI LIME Explainability"** on any card to see horizontal bar charts displaying exactly how much percentage weight each keyword contributed to the toxic decision.
- Click **Approve** or **Hide**. This logs your decision directly into SQLite.

### 3. Dynamic Sandbox Test
- Go to the **Sandbox Text Toxicity Analyzer** at the top of the dashboard.
- Type in any phrase (e.g. *"Stop posting your trash garbage here, you absolute idiot"*).
- Click **Run AI** and watch the system instantly append the custom tweet to the moderation queue, highlight the toxic words, and generate LIME attributions.

### 4. Interactive Threat Graph (NetworkX)
- Go to the **Bullying Graph** tab.
- Hover on any bubble/user node: it dims unrelated edges and highlights attacker-victim lines. Repeated incident counts will render directly on the links.
- Click a node: the **Threat Inspector** displays the NetworkX calculated metrics: role (Abuser vs. Victim), Out-degree centrality (bullying frequency ratio), In-degree centrality, and Clustering coefficient.
- The **Cliques** panel displays modularity sub-groups of users engaged in mutual conflict.

### 5. Live Recharts Dashboard
- Navigate to the **Analytics Panel**.
- Review the **Daily Classification trends** and **AI/Human Alignment Pie chart** (which measures how often the human moderator actions agreed with the AI predictions). These charts fetch live data from the SQLite logs and update immediately as you moderate posts.
