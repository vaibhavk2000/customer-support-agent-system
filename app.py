import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"   # Change if needed


# ============================================================
# LLM CALL
# ============================================================

def call_llm(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


# ============================================================
# SIMPLE RETRIEVAL / MEMORY
# ============================================================

def load_chunks(file_path: str) -> List[str]:
    text = Path(file_path).read_text(encoding="utf-8")
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def cosine_similarity(a: Counter, b: Counter) -> float:
    common = set(a.keys()) & set(b.keys())
    numerator = sum(a[word] * b[word] for word in common)

    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return numerator / (norm_a * norm_b)


def retrieve_relevant_chunks(query: str, chunks: List[str], top_k: int = 4) -> List[str]:
    query_vec = Counter(tokenize(query))
    scored = []

    for chunk in chunks:
        chunk_vec = Counter(tokenize(chunk))
        score = cosine_similarity(query_vec, chunk_vec)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k] if score > 0]


# ============================================================
# TOOLS
# ============================================================

MOCK_ORDERS = {
    "1001": {"status": "processing", "refundable": True},
    "1002": {"status": "shipped", "refundable": False},
    "1003": {"status": "delivered", "refundable": True},
    "1004": {"status": "delayed", "refundable": True},
}


def extract_order_id(text: str) -> str:
    match = re.search(r"\b(100[1-4])\b", text)
    return match.group(1) if match else ""


def lookup_order(order_id: str) -> str:
    if not order_id:
        return "No valid order ID provided."
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return f"Order {order_id} was not found."
    return f"Order {order_id} status: {order['status']}. Refundable flag: {order['refundable']}."


def calculator(expression: str) -> str:
    allowed = re.fullmatch(r"[0-9\.\+\-\*\/\%\(\) ]+", expression)
    if not allowed:
        return "Error: unsupported expression."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# AGENT PROMPTS
# ============================================================

ROUTER_PROMPT = """
You are a Router Agent.

Classify the request into one of these categories:
- refund
- order_status
- account_access
- troubleshooting
- escalation
- general_support

Return only JSON in this format:
{"category":"...", "needs_order_lookup": true/false, "needs_escalation": true/false}
""".strip()


RESEARCH_PROMPT = """
You are a Research Agent.

Your job:
- Use the provided policy context only.
- Extract the most relevant support rules.
- Summarize them in bullet points.
- Do not answer the customer directly.
- Do not invent policy.
""".strip()


SUPPORT_PROMPT = """
You are a Support Agent.

Your job:
- Draft a customer-facing support response.
- Use the research notes and tool results.
- Be professional, empathetic, concise, and action-oriented.
- Never invent order status, refunds, or policy.
- If escalation is required, clearly say that a human specialist will assist.
- Do not expose internal policy wording.
""".strip()


REVIEWER_PROMPT = """
You are a Reviewer Agent.

Your job:
- Review the draft response for:
  - accuracy
  - clarity
  - policy compliance
  - tone
- Provide:
  1. Review Notes
  2. Final Response
- Improve the response if needed.
""".strip()


EVALUATOR_PROMPT = """
You are an Evaluation Agent.

Score the response from 1 to 5 on:
- accuracy
- clarity
- policy_compliance
- helpfulness

Return only JSON in this format:
{
  "accuracy": 0,
  "clarity": 0,
  "policy_compliance": 0,
  "helpfulness": 0,
  "overall_pass": true,
  "notes": "..."
}
""".strip()


# ============================================================
# AGENT FUNCTIONS
# ============================================================

def router_agent(user_input: str) -> Dict[str, Any]:
    raw = call_llm(ROUTER_PROMPT, f"Customer request: {user_input}")
    try:
        parsed = json.loads(raw)
        return parsed
    except json.JSONDecodeError:
        return {
            "category": "general_support",
            "needs_order_lookup": False,
            "needs_escalation": False,
        }


def research_agent(user_input: str, kb_chunks: List[str], router_output: Dict[str, Any]) -> str:
    relevant_chunks = retrieve_relevant_chunks(user_input + " " + router_output.get("category", ""), kb_chunks, top_k=4)
    context = "\n\n".join(f"- {chunk}" for chunk in relevant_chunks) if relevant_chunks else "No relevant context found."

    user_prompt = f"""
Customer request:
{user_input}

Router output:
{json.dumps(router_output)}

Relevant policy context:
{context}

Create concise internal research notes in bullet points.
""".strip()

    return call_llm(RESEARCH_PROMPT, user_prompt)


def support_agent(user_input: str, router_output: Dict[str, Any], research_notes: str, tool_results: str) -> str:
    user_prompt = f"""
Customer request:
{user_input}

Router output:
{json.dumps(router_output)}

Research notes:
{research_notes}

Tool results:
{tool_results}

Draft the customer response.
""".strip()

    return call_llm(SUPPORT_PROMPT, user_prompt)


def reviewer_agent(user_input: str, draft_response: str) -> str:
    user_prompt = f"""
Customer request:
{user_input}

Draft response:
{draft_response}

Review and improve it.
""".strip()

    return call_llm(REVIEWER_PROMPT, user_prompt)


def evaluator_agent(user_input: str, final_response: str) -> Dict[str, Any]:
    raw = call_llm(
        EVALUATOR_PROMPT,
        f"""
Customer request:
{user_input}

Final response:
{final_response}
""".strip()
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "accuracy": 3,
            "clarity": 3,
            "policy_compliance": 3,
            "helpfulness": 3,
            "overall_pass": False,
            "notes": "Evaluation parser fallback triggered."
        }


# ============================================================
# GUARDRAILS
# ============================================================

def detect_prompt_injection(user_input: str) -> bool:
    suspicious_patterns = [
        "ignore previous instructions",
        "reveal system prompt",
        "show hidden instructions",
        "ignore policy",
        "override instructions",
    ]
    lowered = user_input.lower()
    return any(p in lowered for p in suspicious_patterns)


def requires_escalation(user_input: str, router_output: Dict[str, Any]) -> bool:
    escalation_terms = ["legal", "fraud", "chargeback", "lawsuit", "dispute"]
    lowered = user_input.lower()
    if router_output.get("needs_escalation"):
        return True
    return any(term in lowered for term in escalation_terms)


def validate_output(final_text: str) -> Tuple[bool, List[str]]:
    errors = []

    if len(final_text.strip()) < 40:
        errors.append("Response is too short.")
    if "internal policy" in final_text.lower():
        errors.append("Leaked internal wording.")
    if "guaranteed refund" in final_text.lower():
        errors.append("Unsupported refund promise.")

    return (len(errors) == 0, errors)


# ============================================================
# ORCHESTRATION
# ============================================================

def run_workflow(user_input: str, kb_chunks: List[str]) -> Dict[str, Any]:
    workflow_log: List[Dict[str, Any]] = []
    start_time = time.time()

    state: Dict[str, Any] = {
        "user_input": user_input,
        "router_output": {},
        "research_notes": "",
        "tool_results": "",
        "draft_response": "",
        "review_output": "",
        "final_response": "",
        "evaluation": {},
        "guardrail_flags": [],
        "latency_seconds": 0.0,
    }

    # Guardrail 1: prompt injection detection
    if detect_prompt_injection(user_input):
        state["guardrail_flags"].append("prompt_injection_detected")
        state["final_response"] = (
            "I’m unable to follow instructions that attempt to override system behavior. "
            "Please describe your support issue directly, and I’ll help within policy."
        )
        state["latency_seconds"] = round(time.time() - start_time, 2)
        return state

    # Step 1: Router
    state["router_output"] = router_agent(user_input)
    workflow_log.append({"step": "router", "output": state["router_output"]})

    # Step 2: Research
    state["research_notes"] = research_agent(user_input, kb_chunks, state["router_output"])
    workflow_log.append({"step": "research", "output": state["research_notes"]})

    # Step 3: Tools
    tool_outputs = []

    if state["router_output"].get("needs_order_lookup"):
        order_id = extract_order_id(user_input)
        order_result = lookup_order(order_id)
        tool_outputs.append(f"Order Lookup: {order_result}")

    state["tool_results"] = "\n".join(tool_outputs) if tool_outputs else "No tool used."
    workflow_log.append({"step": "tools", "output": state["tool_results"]})

    # Step 4: Escalation override
    if requires_escalation(user_input, state["router_output"]):
        state["guardrail_flags"].append("escalation_required")
        state["draft_response"] = (
            "I’m sorry you’re dealing with this. Your request needs review by a human support specialist. "
            "I’m escalating this so the right team can assist you directly."
        )
    else:
        # Step 5: Draft response
        state["draft_response"] = support_agent(
            user_input,
            state["router_output"],
            state["research_notes"],
            state["tool_results"]
        )

    workflow_log.append({"step": "support", "output": state["draft_response"]})

    # Step 6: Review
    state["review_output"] = reviewer_agent(user_input, state["draft_response"])
    workflow_log.append({"step": "reviewer", "output": state["review_output"]})

    # Extract final response
    if "Final Response" in state["review_output"]:
        state["final_response"] = state["review_output"].split("Final Response", 1)[-1].strip(": \n")
    else:
        state["final_response"] = state["review_output"]

    # Step 7: Output validation
    is_valid, validation_errors = validate_output(state["final_response"])
    if not is_valid:
        state["guardrail_flags"].append("output_validation_failed")
        state["final_response"] = (
            "Thanks for your request. I want to make sure this is handled correctly, "
            "so I’m routing it for a more careful review by support."
        )
        workflow_log.append({"step": "validation", "errors": validation_errors})
    else:
        workflow_log.append({"step": "validation", "status": "passed"})

    # Step 8: Evaluation
    state["evaluation"] = evaluator_agent(user_input, state["final_response"])
    workflow_log.append({"step": "evaluation", "output": state["evaluation"]})

    state["latency_seconds"] = round(time.time() - start_time, 2)
    state["workflow_log"] = workflow_log
    return state


# ============================================================
# EVALUATION PIPELINE
# ============================================================

def run_test_suite(kb_chunks: List[str]) -> List[Dict[str, Any]]:
    test_cases = json.loads(Path("test_cases.json").read_text(encoding="utf-8"))
    results = []

    for test in test_cases:
        output = run_workflow(test["input"], kb_chunks)
        final_response = output["final_response"].lower()

        must_include_ok = all(
            phrase.lower() in final_response for phrase in test["expected_must_include"]
        )

        results.append({
            "name": test["name"],
            "input": test["input"],
            "must_include_pass": must_include_ok,
            "overall_eval_pass": output["evaluation"].get("overall_pass", False),
            "guardrail_flags": output.get("guardrail_flags", []),
            "latency_seconds": output.get("latency_seconds", 0.0),
            "final_response": output["final_response"],
            "evaluation": output["evaluation"],
        })

    return results


# ============================================================
# REPORTING & PERSISTENCE
# ============================================================

def save_run_to_file(result: Dict[str, Any], filename: str = "latest_run_report.json") -> None:
    Path(filename).write_text(json.dumps(result, indent=2), encoding="utf-8")


def save_trace_log(result: Dict[str, Any], filename: str = "trace_log.json") -> None:
    trace = {
        "user_input": result["user_input"],
        "router_output": result["router_output"],
        "guardrail_flags": result["guardrail_flags"],
        "latency_seconds": result["latency_seconds"],
        "evaluation": result["evaluation"],
        "workflow_log": result.get("workflow_log", []),
    }
    Path(filename).write_text(json.dumps(trace, indent=2), encoding="utf-8")


def print_workflow_result(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("FINAL RESPONSE")
    print("=" * 80)
    print(result["final_response"])

    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    print(json.dumps(result["evaluation"], indent=2))

    print("\n" + "=" * 80)
    print("GUARDRAILS")
    print("=" * 80)
    print(result["guardrail_flags"])

    print("\n" + "=" * 80)
    print("LATENCY")
    print("=" * 80)
    print(f'{result["latency_seconds"]} seconds')


def print_test_report(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("TEST SUITE RESULTS")
    print("=" * 80)

    for row in results:
        print(f"\nTest: {row['name']}")
        print(f"Must-Include Pass: {row['must_include_pass']}")
        print(f"Overall Eval Pass: {row['overall_eval_pass']}")
        print(f"Guardrails: {row['guardrail_flags']}")
        print(f"Latency: {row['latency_seconds']} seconds")
        print(f"Evaluation: {json.dumps(row['evaluation'])}")


# ============================================================
# MAIN APP
# ============================================================

def main() -> None:
    kb_chunks = load_chunks("knowledge_base.txt")

    print("\nProduction-Ready Support Agent System")
    print("1. Run live support request")
    print("2. Run evaluation test suite")
    print("3. Exit\n")

    while True:
        choice = input("Choose an option: ").strip()

        if choice == "1":
            user_input = input("\nEnter support request: ").strip()
            result = run_workflow(user_input, kb_chunks)
            print_workflow_result(result)
            save_run_to_file(result)
            save_trace_log(result)
            print("\nSaved run report and trace log to file.\n")

        elif choice == "2":
            results = run_test_suite(kb_chunks)
            print_test_report(results)

        elif choice == "3":
            print("Goodbye.")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.\n")


if __name__ == "__main__":
    main()
