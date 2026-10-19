# CyberShield: Context-Aware Cyberbullying Detection & Explainability System (v2.0)

A production-grade, full-stack cyberbullying detection and AI explainability platform developed as a B.Tech final-year capstone project. 

CyberShield advances beyond conventional keyword lexicons and isolated text classifiers by integrating a **multi-source retrospective context collector**, a **fine-tuned multi-task contextual transformer**, a **learned neural fusion network (FusionMLP)**, an **interactive 3-band decision engine with human-in-the-loop escalation**, **algorithmic fairness audits**, and **social network threat graph analysis**.

---

## 🚀 System Architecture

```
                                    ┌────────────────────────────────────────────────────────┐
                                    │               React 19 + Tailwind v4 UI                │
                                    │  (Vite + Recharts + NetworkX Graph SVG + WebSockets)   │
                                    └───────────────────────────┬────────────────────────────┘
                                                                │ HTTP REST / WebSocket (/ws/feed)
                                                                ▼
                                    ┌────────────────────────────────────────────────────────┐
                                    │                     FastAPI Server                     │
                                    │        JWT Auth & RBAC (Admin / Moderator Guard)       │
                                    │        Dual Routing (/api/* and /*) & Rate Limiter     │
                                    └─────┬─────────────────────┬──────────────────────┬─────┘
                                          │                     │                      │
                                          ▼                     ▼                      ▼
                     ┌──────────────────────────────┐  ┌──────────────────┐  ┌──────────────────┐
                     │   SQLAlchemy 5-Table DB      │  │ NetworkX Graph   │  │   Fairness &     │
                     │  (users, predictions, queue, │  │ Centrality Engine│  │ Dialect Auditor  │
                     │  moderator_actions, fairness)│  │ (rolling 7-day)  │  │  (AAVE vs SAE)   │
                     │   [PostgreSQL or SQLite]     │  └──────────────────┘  └──────────────────┘
                     └──────────────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────────────────────────────────────┐
                     │             Unified Context-Aware Inference Pipeline                    │
                     │                                                                         │
                     │  ┌───────────────────────┐             ┌─────────────────────────────┐  │
                     │  │   Target Resolution   │             │   Multi-Source Collector    │  │
                     │  │ (@mentions, replies)  │             │ (Conv, Behav, Temp, Graph)  │  │
                     │  └───────────┬───────────┘             └──────────────┬──────────────┘  │
                     │              │                                        │                 │
                     │              ▼                                        ▼                 │
                     │  ┌───────────────────────┐             ┌─────────────────────────────┐  │
                     │  │ Base Model: Toxic-BERT│             │ Context Model: DistilBERT   │  │
                     │  │ (unitary/toxic-bert)  │             │ (context-distilbert-v1)     │  │
                     │  │   Outputs P_toxic     │             │ 5 Auxiliary Head Flags      │  │
                     │  └───────────┬───────────┘             └──────────────┬──────────────┘  │
                     │              │                                        │                 │
                     │              └───────────────────┬────────────────────┘                 │
                     │                                  ▼                                      │
                     │                   ┌─────────────────────────────┐                       │
                     │                   │ 33-Dim Feature Assembly &   │                       │
                     │                   │ Robust Standardization      │                       │
                     │                   └──────────────┬──────────────┘                       │
                     │                                  ▼                                      │
                     │                   ┌─────────────────────────────┐                       │
                     │                   │ Learned Fusion Network      │                       │
                     │                   │ (PyTorch FusionMLP v1)      │                       │
                     │                   │   Outputs P_final (0 to 1)  │                       │
                     │                   └──────────────┬──────────────┘                       │
                     │                                  ▼                                      │
                     │                   ┌─────────────────────────────┐                       │
                     │                   │ 3-Band Decision Engine      │                       │
                     │                   │  < 40%: Auto-Approve (Safe) │                       │
                     │                   │  40%-70%: Uncertain -> HITL │                       │
                     │                   │  > 70%: Cyberbullying       │                       │
                     │                   └──────────────┬──────────────┘                       │
                     │                                  │                                      │
                     │              ┌───────────────────┴───────────────────┐                  │
                     │              ▼                                       ▼                  │
                     │  ┌──────────────────────┐               ┌───────────────────────────┐   │
                     │  │  Token LIME Weights  │               │ 5-Pillar Context Rationale│   │
                     │  └──────────────────────┘               └───────────────────────────┘   │
                     └─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Features & Innovations

### 1. Multi-Tier AI Detection & Learned Score Fusion
- **Base Toxicity Classifier ($P_{\text{toxic}}$)**: Powered by HuggingFace `unitary/toxic-bert` for deep contextual token representation.
- **Contextual Sub-Model ($P_{\text{context}}$)**: A fine-tuned `DistilBERT` transformer (`context-distilbert-v1`, PyTorch weights) trained with multi-task auxiliary heads to predict whether content is **targeted**, **repeated**, **escalating**, **sarcastic**, or **threatening**.
- **Learned Fusion Engine (FusionMLP)**: Rather than arbitrary heuristic weights, CyberShield uses a trained multi-layer perceptron (`fusion-mlp-v1`) that consumes a **33-dimensional normalized feature vector** (ingesting base toxicity, contextual probabilities, conversation turns, temporal velocities, past offenses, and graph centrality).
- **Fast Keyword Fallback Mode**: Allows full local demonstration without GPU/PyTorch overhead via regex pattern matching and simulated attributions.

### 2. Multi-Source Retrospective Context Collector
Extracts 4 distinct dimensions of context before computing final toxicity:
- **Conversational Context**: Analyzes thread depth, reply time deltas, and parent message authors.
- **Behavioral Attacker Profiling**: Tracks lifetime attacks sent, attack ratios against specific targets, distinct target counts, and prior moderator flags.
- **Temporal Burst Velocity**: Measures attack frequency across rolling windows ($V_{\text{1h}}, V_{\text{6h}}, V_{\text{24h}}, V_{\text{7d}}$) and calculates escalation rates.
- **Social Graph Features**: Extracts directed in-degree and out-degree centralities, clustering coefficients, and mutual interaction edge weights.

### 3. Human-in-the-Loop (HITL) Review Queue
- **Parameterized 3-Band Decision Policy**:
  - **Safe ($P_{\text{final}} < 0.40$)**: Automatically marked safe and approved.
  - **Uncertain ($0.40 \le P_{\text{final}} \le 0.70$)**: Automatically escalated to the **Review Queue** with machine-generated escalation rationales.
  - **Cyberbullying ($P_{\text{final}} > 0.70$)**: Automatically flagged or hidden.
- **Dedicated Review Queue Panel**: Moderators inspect borderline posts, review contextual evidence, and log decisions (`Approved`, `Hidden`, `Review`) with notes.
- **Active Learning Export**: Exposes `/api/feedback/dataset` to continuously retrain and calibrate future model versions with verified human decisions.

### 4. Interactive Social Network Threat Graph (NetworkX)
- Dynamically compiles interaction graphs from database logs using a **rolling 7-day window**.
- Computes directed network centrality metrics to automatically assign behavioral roles:
  - **Abuser**: Sent $\ge 3$ attacks in the rolling window.
  - **Victim**: Received $\ge 3$ attacks (and sent $< 3$).
  - **Aggressive Participator**: Sent $\ge 3$ and received $\ge 3$ attacks (mutual hostility).
  - **Neutral**: Default for incidental participants.
- Interactive SVG network visualizer highlights victim-attacker links, displays edge weights, and reveals conflict cliques.

### 5. Algorithmic Fairness & Dialect Bias Auditing
- **Dialect Parity Audit (AAVE vs SAE)**: Evaluates false positive rate (FPR) parity on African American Vernacular English (AAVE) proxies (Blodgett et al.) compared to Standard American English (SAE).
- **Identity Parity Audit**: Tracks model fairness across protected identity terms (Borkan et al.).
- Interactive Recharts bar charts demonstrate that our Context-Aware Fusion system drastically suppresses false positive spikes commonly seen in standard classifiers.

### 6. Rigorous 4-Tier Evaluation & ACL 2020 CheckList Suite
- **4 Architectural Evaluation Tiers**:
  1. *Tier 1*: Stock Toxic-BERT Baseline (Text Only)
  2. *Tier 2*: Toxic-BERT + Rule-Based Additive Clamping Context Layer
  3. *Tier 3*: Toxic-BERT + Fine-tuned DistilBERT Context Sub-Model
  4. *Tier 4*: Full Learned Context-Aware Fusion (Collector + DistilBERT + FusionMLP)
- **CheckList Behavioral Testing (Ribeiro et al., ACL 2020)**:
  - **Minimum Functionality Tests (MFT)**: Basic sanity check capabilities.
  - **Invariance Tests (INV)**: Stability under perturbation of benign names, handle references, and punctuation.
  - **Directional Tests (DIR)**: Monotonic shifts upon adding explicit negations or sarcastic qualifiers.
- **Feature Ablation Study**: Empirically quantifies accuracy degradation when removing conversation, behavioral, temporal, or graph pillars.

### 7. Enterprise Security, JWT Auth & RBAC
- **Role-Based Access Control**: Separate privileges for **Admin** (system-wide configuration, fairness audits, user management) and **Moderator** (feed review and queue resolution).
- **Security Protections**: Passwords encrypted with Bcrypt, stateless JWT tokens with 60-minute expiry, and in-memory IP rate limiting (max 5 failed logins per 5 minutes).
- **1-Click Demo Logins**: Built-in modal presets for seamless live demonstrations.

### 8. Bi-directional WebSocket Streaming
- High-throughput WebSocket endpoint (`/ws/feed`) with origin verification.
- Ingestion simulator streaming from a curated benchmark dataset with dynamic speed controls (1s – 5s per comment), pause, resume, and real-time frontend push.

---

## 📁 Repository Structure

```
cybbulldet/
├── docker-compose.yml              # Multi-container orchestration (PostgreSQL, Backend, Frontend)
├── README.md                       # Comprehensive system documentation
│
├── backend/
│   ├── main.py                     # FastAPI server, dual routing, and endpoints
│   ├── auth.py                     # JWT token handling, bcrypt hashing, and RBAC guards
│   ├── db.py                       # SQLAlchemy engine, migrations, seeds, and CRUD queries
│   ├── stream.py                   # WebSocket connection manager and feed streaming worker
│   ├── evaluate.py                 # 4-Tier evaluation, feature ablation, and benchmarks
│   ├── checklist_test.py           # Ribeiro et al. (ACL 2020) MFT, INV, DIR test cases
│   ├── fairness.py                 # Blodgett et al. AAVE proxy & Borkan et al. identity audits
│   ├── graph.py                    # NetworkX directed graph metrics, centrality, and cliques
│   ├── fusion.py                   # Classical additive clamping & feature calculations
│   ├── utils.py                    # Dynamic target resolution (@mentions, replies)
│   ├── tweets.csv                  # 100-comment curated benchmark evaluation dataset
│   │
│   ├── context/                    # Multi-source retrospective context collector
│   │   ├── collector.py            # Master collector aggregating all 4 pillars
│   │   ├── conversation.py         # Thread depth, parent message, and latency features
│   │   ├── behavioral.py           # Historical offenses, target repetition, flag counts
│   │   ├── temporal.py             # Velocity windows (1h, 6h, 24h, 7d), escalation rates
│   │   └── graph_features.py       # In/out centralities and interaction weights
│   │
│   ├── models/                     # Deep learning model architectures and schemas
│   │   ├── db_models.py            # SQLAlchemy 5-table schema models
│   │   ├── toxic_model.py          # HuggingFace unitary/toxic-bert wrapper
│   │   ├── context_model.py        # Fine-tuned DistilBERT multi-task sub-model
│   │   ├── fusion_model.py         # PyTorch FusionMLP network & 33-dim schema definition
│   │   ├── tokenizer.py            # Shared tokenization utilities
│   │   └── weights/                # Serialized model weights & feature scalers
│   │       ├── context_submodel.pt # Fine-tuned DistilBERT checkpoint (~265MB)
│   │       ├── fusion_mlp.pt       # Trained FusionMLP checkpoint
│   │       ├── feature_scaler.json # Feature normalization parameters
│   │       ├── context_model_metadata.json
│   │       └── fusion_model_metadata.json
│   │
│   ├── inference/                  # Production inference orchestrator
│   │   ├── pipeline.py             # End-to-end processing pipeline
│   │   └── decision_engine.py      # Configurable 3-band decision policy
│   │
│   ├── explainability/             # Explainability engine
│   │   ├── explainer.py            # Master multi-pillar explainer
│   │   └── contextual_factors.py   # Contextual contribution rationales
│   │
│   └── training/                   # Model training and dataset synthesis scripts
│       ├── generate_context_dataset.py
│       ├── train_context_model.py
│       └── train_fusion_model.py
│
└── frontend/
    ├── package.json                # Dependencies (React 19, Tailwind v4, Recharts, Lucide)
    ├── vite.config.js              # Vite bundler configuration & backend API proxies
    ├── src/
    │   ├── App.jsx                 # Main application layout, routing, and heartbeats
    │   ├── context/
    │   │   └── AuthContext.jsx     # Global JWT authentication and user role state
    │   ├── components/
    │   │   ├── Dashboard.jsx       # Live WebSocket feed, text sandbox, and LIME cards
    │   │   ├── ReviewQueue.jsx     # HITL queue for resolving uncertain cases (40%-70%)
    │   │   ├── FairnessPanel.jsx   # Demographic parity & dialect audit charts
    │   │   ├── NetworkGraph.jsx    # Interactive SVG directed social threat graph
    │   │   ├── Analytics.jsx       # Daily trends, alignment pie chart, 4-tier benchmark
    │   │   ├── LoginModal.jsx      # Role selection modal with 1-click demo logins
    │   │   └── Sidebar.jsx         # Navigation drawer, queue badge, and AI mode toggle
    │   └── tests/                  # Vitest unit test suite
```

---

## 🛠️ Installation & Setup

### Option A: Docker Compose (Recommended - Zero Configuration)

To run the entire system (PostgreSQL 15 database, FastAPI backend, and React frontend) in isolated Docker containers:

```bash
docker compose up --build
```

- **Frontend Application**: `http://localhost:80` (or `http://localhost:5173`)
- **Backend API Docs**: `http://localhost:8000/docs`
- **PostgreSQL Database**: Port `5432`

---

### Option B: Manual Local Setup

#### Prerequisites
- [Python 3.10+](https://www.python.org/downloads/)
- [Node.js 18+](https://nodejs.org/)

#### Step 1: Backend Setup
1. Open a terminal in the root folder and navigate to `backend`:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   # Windows:
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux:
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   python main.py
   ```
   *The backend will boot on `http://127.0.0.1:8000`. SQLite tables `moderation.db` will automatically initialize, run legacy migrations, and seed baseline accounts and records.*

#### Step 2: Frontend Setup
1. Open a second terminal and navigate to `frontend`:
   ```bash
   cd frontend
   ```
2. Install frontend packages:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   *The frontend dashboard will be available at `http://localhost:5173`.*

---

## 🔐 Demo Credentials (RBAC Quick Login)

The system automatically initializes standard demo accounts upon first startup. Click **Login** in the top-right header or sidebar to authenticate:

| Role | Email Address | Password | Permissions |
| :--- | :--- | :--- | :--- |
| **Admin** | `admin@cybershield.org` | `AdminPass123!` | Full Access (Dashboard, Review Queue, Network Graph, Analytics, Fairness Audits, System Status) |
| **Moderator** | `mod@cybershield.org` | `ModPass123!` | Moderation Access (Dashboard, Review Queue Resolution, Network Graph) |

*Tip: The login modal includes **1-Click Quick Preset** buttons for instant demo access without manual typing.*

---

## 🌟 Examiner Demonstration Walkthrough

When presenting CyberShield to evaluators, project examiners, or reviewers, follow this structured demo narrative:

### 1. Dual AI Model Modes & Text Sandbox Analysis
- Navigate to the **Dashboard** and inspect the **Sandbox Text Toxicity Analyzer**.
- Enter challenging linguistic cases:
  - *Negation Test*: `"Don't worry, you aren't stupid at all"` $\to$ Stock BERT models often trigger on "stupid", while CyberShield's context layer resolves it to **Safe**.
  - *Positive Slang Test*: `"You absolutely killed that performance, sick beat!"` $\to$ Recognized as positive praise, preventing false positive flags.
  - *Sarcasm Test*: `"Wow, you are so remarkably intelligent, said no one ever"` $\to$ Sarcastic qualifier triggers auxiliary heads and lifts probability into **Cyberbullying**.
- Click **"Inspect Score Fusion & LIME Weights"** on any card to view token-level attributions and individual feature delta adjustments.

### 2. Live WebSocket Feed & Dynamic Target Resolution
- Click **Resume Feed** to activate the streaming simulator.
- Use the **Speed Slider** to adjust ingestion velocity (1.0s to 5.0s per comment).
- Observe that target recipients are resolved dynamically (first `@mention`, parent reply author, or `@anonymous`).
- Posts whose fused confidence falls into the $[40\%, 70\%]$ band will automatically display an **Escalated to Human Review Queue** banner.

### 3. Human-in-the-Loop Review Queue
- Click on the **Review Queue** tab in the sidebar (indicated by a live pending badge count).
- Review borderline cases that the model refused to make autonomous assumptions on.
- Inspect the contextual triggers (e.g. *repeated interactions*, *rapid temporal bursts*, or *sarcastic qualifier presence*).
- Submit decisions (**Approve** or **Hide**) with optional moderator notes to log entries to the audit trail.

### 4. Interactive Social Network Threat Graph (NetworkX)
- Click on the **Bullying Graph** tab.
- Inspect the directed social graph built from rolling weekly database interactions.
- Hover over nodes to isolate communication edges and display attack counts.
- View dynamically computed NetworkX centrality metrics:
  - Out-degree centrality (attack frequency ratio)
  - In-degree centrality (victimization frequency ratio)
  - Clustering coefficient
  - Automatic behavioral roles: **Abuser**, **Victim**, **Aggressive Participator**, or **Neutral**.

### 5. Algorithmic Fairness & Dialect Parity Audit
- Log in as **Admin** and click on the **Fairness Audit** tab.
- Review the **Blodgett et al. AAVE Dialect Proxy** false positive audit.
- Demonstrate that while standard classifiers penalize non-standard dialects with inflated false positive rates, our Context-Aware system eliminates dialect disparity.

### 6. Analytics & 4-Tier Model Evaluation Report
- Click on the **Analytics Panel**.
- Review **Daily Classification Trends** and **AI/Human Alignment** (measuring how often human moderators agreed with AI decisions).
- Click **Run Benchmark** to execute the live evaluation across the **4 Architectural Tiers** and the **ACL 2020 CheckList Test Suite** (Minimum Functionality, Invariance, and Directional shifts).

---

## 📊 Evaluation & Benchmark Results

Evaluated on our curated 100-comment adversarial benchmark dataset containing obvious bullying, safe banter, negations, reclaimed positive slang, and sarcasm:

| Classifier Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Keyword Lexicon Baseline** (Regex matching) | 76.0% | 69.1% | 84.4% | 76.0% |
| **Stock Toxic-BERT Baseline** (Text Only) | 87.0% | 92.1% | 77.8% | 84.3% |
| **Context-Aware Hybrid System** (Toxic-BERT + Additive Clamping) | 93.0% | 97.5% | 86.7% | 91.8% |
| **Full Learned Context-Aware Fusion** (Collector + DistilBERT + FusionMLP) | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

### CheckList Behavioral Testing (Ribeiro et al., ACL 2020)
- **Minimum Functionality Tests (MFT)**: 100% Pass Rate (Sanity checks on overt threats and obvious safe posts).
- **Invariance Tests (INV)**: 100% Pass Rate (Robustness under benign name substitutions and punctuation changes).
- **Directional Tests (DIR)**: 100% Pass Rate (Consistent probability reduction on negations and consistent escalation on sarcastic collocations).

---

## 📡 REST API & WebSocket Reference

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `POST` | `/api/auth/login` | Authenticate and obtain JWT bearer token | No (Rate Limited) |
| `GET` | `/api/auth/me` | Fetch active user profile and assigned role | Yes |
| `POST` | `/api/auth/register` | Register new moderator or admin account | No |
| `POST` | `/api/predict` | Classify text using full end-to-end inference pipeline | No |
| `GET` | `/api/posts` | Retrieve all persistent classification records | No |
| `POST` | `/api/posts/{id}/action` | Update human moderation action (`Approved`/`Hidden`) | No |
| `GET` | `/api/queue` | Fetch items in the uncertain review band ($40\% - 70\%$) | Yes (Admin/Mod) |
| `POST` | `/api/queue/{id}/resolve`| Submit human decision for an escalated queue item | Yes (Admin/Mod) |
| `GET` | `/api/graph` | Compile NetworkX directed graph analysis and roles | No |
| `GET` | `/api/analytics` | Retrieve daily trends, alignment, and abuser metrics | No |
| `GET` | `/api/fairness` | Run demographic & AAVE dialect parity audit | No |
| `GET` | `/api/evaluation` | Execute 4-Tier evaluation and CheckList test suite | No |
| `POST` | `/api/toggle-mode` | Toggle between Toxic-BERT and Keyword Fallback | No |
| `GET` | `/api/status` | Hardware status, model availability, and active mode | No |
| `WS` | `/ws/feed` | Real-time WebSocket feed subscription & controls | Handshake Origin |

---

## ⚙️ Environment Configuration

Configuration variables can be defined in a `.env` file in the root or `backend` folder:

```env
# Database Configuration (PostgreSQL or SQLite)
DATABASE_URL=sqlite:///moderation.db

# Security & JWT
JWT_SECRET_KEY=cyber-production-secure-key-32-bytes!
ENV=development

# Model Execution Defaults
USE_REAL_MODEL=True
USE_REAL_MODEL_ON_BOOT=False

# Decision Engine Thresholds
DECISION_THRESHOLD_SAFE=0.40
DECISION_THRESHOLD_BULLYING=0.70

# CORS Allowed Origins
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000,http://localhost:80
```

---

## 👥 Authors & Academic Context

- **Project**: Context-Aware Cyberbullying Detection & Explainability System (CyberShield v2.0)
- **Degree**: Bachelor of Technology (B.Tech) - Final Year Capstone Project
- **Lead Developer**: Bharath Tummalapalli
- **License**: MIT Academic License
