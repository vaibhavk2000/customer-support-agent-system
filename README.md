# 🤖 Customer Support Multi-Agent System (Capstone)

A production-ready, multi-agent AI system built to handle customer support workflows autonomously while adhering to strict company policies, tool integration, guardrail validations, and evaluation suites.

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
