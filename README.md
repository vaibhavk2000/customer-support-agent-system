# 🤖 Customer Support Multi-Agent System (Capstone)

A production-ready, multi-agent AI system built with Python and Ollama to handle customer support workflows autonomously while adhering to strict company policies, tool integration, memory retrieval, guardrail validations, and automated evaluation suites.

---

## 🏗️ System Architecture

```text
User Request ──► API / CLI Entry ──► Router Agent
                                          │
                                          ▼
                                   Research Agent ◄──► Internal KB Memory
                                          │
                                          ▼
                                   Support Agent  ◄──► Tools (Order Lookup)
                                          │
                                          ▼
                                   Reviewer Agent
                                          │
                                          ▼
                                  Guardrails & Validation
                                          │
                                          ▼
                                  Evaluation Pipeline
                                          │
                                          ▼
                              Final Response + Trace Log

```

---

## ✨ Key Features

* **Multi-Agent Orchestration:** Router, Research, Support, Reviewer, and Evaluation agents working in sequence.
* **Internal Memory Retrieval:** Cosine similarity retrieval over internal policy documents.
* **Tool Integration:** Mock Order Lookup tool for real-time order status checks.
* **Safety & Guardrails:** Prompt injection detection, forced human escalation rules, and output leakage validation.
* **Evaluation Pipeline:** LLM-as-a-Judge rubric evaluator and automated test suite execution.
* **Local Inference:** Fully private, local execution using [Ollama](https://ollama.com/) (`llama3.1`).

---

## 📁 Repository Structure

```text
customer-support-agent-system/
├── .gitignore          # Excludes temporary cache and logs
├── app.py              # Main multi-agent workflow & orchestration logic
├── knowledge_base.txt  # Internal policy knowledge base for memory retrieval
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
└── test_cases.json     # Test suite for evaluation harness

```

---

## 🚀 Setup & Execution Steps

### Step 1: Install Ollama & Pull the Model

Ensure [Ollama](https://ollama.com/) is installed and running on your system, then pull the target model:

```bash
ollama pull llama3.1

```

### Step 2: Clone the Repository

```bash
git clone [https://github.com/vaibhavk2000/customer-support-agent-system.git](https://github.com/vaibhavk2000/customer-support-agent-system.git)
cd customer-support-agent-system

```

### Step 3: Set Up Environment & Dependencies

Create a virtual environment (optional but recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

### Step 4: Run the Application

Launch the Interactive CLI:

```bash
python app.py

```

---

## 🧪 Usage Options

When you run `python app.py`, you will see three menu choices:

1. **Run live support request:** Test real-time queries against the system (e.g., refund queries, order lookup, password reset, or injection attacks).
2. **Run evaluation test suite:** Run all predefined test scenarios in `test_cases.json` through the evaluation pipeline and display pass/fail benchmarks.
3. **Exit:** Terminate the application.

---

## 🛡️ Tested Guardrails

The application includes built-in protective layers that trigger automatically:

* **Prompt Injection:** Captures prompt override attempts and yields safe policy responses.
* **Escalation Rules:** Automatically flags legal threats, fraud, or chargebacks for human handoff.
* **Output Validation:** Blocks short outputs, internal policy text leaks, or false refund guarantees.
