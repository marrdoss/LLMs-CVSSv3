"""
Script 2: Run CVE Severity Evaluation via Ollama
=================================================
Sends all CVE descriptions to Llama 3.1 8B and Mistral 7B.
Models predict EXACT CVSS score (0.0–10.0).
Severity label is derived automatically from predicted score.

Two prompt types: zero-shot and few-shot.
Three runs each for consistency analysis.

Total: 80 CVEs × 2 models × 2 prompt types × 3 runs = 960 prompts
All results saved to: cve_results.csv

Requirements:
    pip install ollama pandas tqdm

Usage:
    python 2_run_evaluation.py
"""

import ollama
import pandas as pd
import re
import time
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────
INPUT_FILE  = "cve_dataset.csv"
OUTPUT_FILE = "cve_results.csv"

MODELS = {
    "llama3.1": "llama3.1",
    "mistral":  "mistral",
}

RUNS_PER_CONDITION = 3
TEMPERATURE        = 0.7   # Non-zero → consistency analysis is meaningful

# CVSS v3.1 severity thresholds
SEVERITY_THRESHOLDS = [
    (9.0, 10.0, "Critical"),
    (7.0, 8.9,  "High"),
    (4.0, 6.9,  "Medium"),
    (0.1, 3.9,  "Low"),
    (0.0, 0.0,  "None"),
]

# ── Prompt Templates ──────────────────────────────────────────

ZERO_SHOT_PROMPT = """\
You are a cybersecurity expert trained in CVSS v3.1 scoring.

Given the vulnerability description below, predict the CVSS v3.1 base score.
The score must be a number between 0.0 and 10.0 with one decimal place.

When estimating, consider:
- Attack Vector: Can it be exploited remotely (higher) or locally (lower)?
- Attack Complexity: Is the attack simple (higher) or complex (lower)?
- Privileges Required: None (higher) vs High (lower)
- User Interaction: None (higher) vs Required (lower)
- Impact on Confidentiality, Integrity, Availability: High (higher) vs None (lower)

CVSS v3.1 severity ranges for reference:
- Critical : 9.0 – 10.0
- High     : 7.0 – 8.9
- Medium   : 4.0 – 6.9
- Low      : 0.1 – 3.9

Return your answer in EXACTLY this format (no extra text):
Predicted Score: [number between 0.0 and 10.0]
Reason: [1-2 sentences]

Vulnerability Description:
{description}
"""

FEW_SHOT_PROMPT = """\
You are a cybersecurity expert trained in CVSS v3.1 scoring.

Given the vulnerability description below, predict the CVSS v3.1 base score.
The score must be a number between 0.0 and 10.0 with one decimal place.

Here are examples to guide your scoring:

---
Example 1 (Low severity):
Description: A local authenticated user can read log files belonging to another user due to insecure file permissions on a shared directory.
Predicted Score: 2.3
Reason: Requires local access and authentication. Impact is limited to low confidentiality disclosure with no integrity or availability impact.

---
Example 2 (Medium severity):
Description: An input validation flaw in the web application allows authenticated remote users to inject SQL commands, potentially exposing database records.
Predicted Score: 6.5
Reason: Network exploitable but requires authentication. Significant confidentiality impact but limited by the need for valid credentials.

---
Example 3 (High severity):
Description: A heap-based buffer overflow in the network daemon allows a remote unauthenticated attacker to execute arbitrary code with service-level privileges.
Predicted Score: 8.8
Reason: Remote unauthenticated exploitation with code execution represents high risk. Privileges are limited to the service account.

---
Example 4 (Critical severity):
Description: A remote code execution vulnerability in the authentication component requires no credentials and no user interaction. Full system compromise is possible with complete impact on confidentiality, integrity, and availability.
Predicted Score: 9.8
Reason: Unauthenticated remote code execution with no user interaction and complete system impact satisfies all critical severity criteria.

---
CVSS v3.1 severity ranges for reference:
- Critical : 9.0 – 10.0
- High     : 7.0 – 8.9
- Medium   : 4.0 – 6.9
- Low      : 0.1 – 3.9

Return your answer in EXACTLY this format (no extra text):
Predicted Score: [number between 0.0 and 10.0]
Reason: [1-2 sentences]

Vulnerability Description:
{description}
"""

# ── Helpers ───────────────────────────────────────────────────

def build_prompt(prompt_type: str, description: str) -> str:
    if prompt_type == "zero_shot":
        return ZERO_SHOT_PROMPT.format(description=description)
    elif prompt_type == "few_shot":
        return FEW_SHOT_PROMPT.format(description=description)
    raise ValueError(f"Unknown prompt type: {prompt_type}")


def parse_score(response_text: str) -> float | None:
    """Extract predicted CVSS score from model response."""
    # Look for 'Predicted Score: X.X'
    match = re.search(
        r"predicted score\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        response_text.lower(),
    )
    if match:
        score = float(match.group(1))
        if 0.0 <= score <= 10.0:
            return round(score, 1)

    # Fallback: find any float/int in range 0-10
    numbers = re.findall(r"\b([0-9](?:\.[0-9]+)?|10(?:\.0+)?)\b", response_text)
    for num in numbers:
        score = float(num)
        if 0.0 <= score <= 10.0:
            return round(score, 1)

    return None


def score_to_severity(score: float | None) -> str:
    """Convert CVSS score to severity label using v3.1 thresholds."""
    if score is None:
        return "Unknown"
    for low, high, label in SEVERITY_THRESHOLDS:
        if low <= score <= high:
            return label
    return "Unknown"


def average_score(scores: list) -> float | None:
    """Average of valid scores across runs."""
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 1)


def is_score_consistent(scores: list, tolerance: float = 1.0) -> bool:
    """Consistent if all valid scores are within ±tolerance of each other."""
    valid = [s for s in scores if s is not None]
    if len(valid) < 2:
        return True
    return (max(valid) - min(valid)) <= tolerance


def is_label_consistent(scores: list) -> bool:
    """Stricter: consistent only if all 3 runs produce the SAME severity label.
    This catches cases like 6.0 vs 7.0 which are within score tolerance
    but fall in different severity classes (Medium vs High).
    """
    labels = [score_to_severity(s) for s in scores if s is not None]
    if len(labels) < 2:
        return True
    return len(set(labels)) == 1


def query_ollama(model: str, prompt: str) -> str:
    """Send a prompt to a local Ollama model and return the response text."""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": TEMPERATURE},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"\n  [ERROR] Ollama call failed for {model}: {e}")
        return "ERROR"


# ── Main ──────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 60)
    print("  CVE CVSS Score Prediction — Ollama Local Models")
    print("=" * 60)

    # Load dataset
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"\n[ERROR] {INPUT_FILE} not found. Run 1_build_dataset.py first.")
        return

    print(f"\n[+] Loaded {len(df)} CVEs from {INPUT_FILE}")

    prompt_types  = ["zero_shot", "few_shot"]
    total_calls   = len(df) * len(MODELS) * len(prompt_types) * RUNS_PER_CONDITION
    print(f"[+] Total prompts to run : {total_calls}")
    print(f"[+] Temperature          : {TEMPERATURE}")
    print(f"[+] Models               : {list(MODELS.keys())}")
    print(f"[+] Task                 : Predict exact CVSS score (0.0–10.0)")
    print("\nStarting evaluation — this will run overnight on Intel Mac...\n")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CVEs"):
        cve_id       = row["CVE_ID"]
        description  = row["Description"]
        official_sev = row["Official_Severity"]
        official_cvss = float(row["Official_CVSS"])

        result_row = {
            "CVE_ID":             cve_id,
            "Official_Severity":  official_sev,
            "Official_CVSS":      official_cvss,
            "Description":        description,
        }

        for model_key, model_name in MODELS.items():
            for pt in prompt_types:
                run_scores = []

                for run_num in range(1, RUNS_PER_CONDITION + 1):
                    prompt   = build_prompt(pt, description)
                    response = query_ollama(model_name, prompt)
                    score    = parse_score(response)
                    run_scores.append(score)

                    # Store raw score per run
                    result_row[f"{model_key}_{pt}_run{run_num}_score"] = score
                    result_row[f"{model_key}_{pt}_run{run_num}_label"] = (
                        score_to_severity(score)
                    )

                    time.sleep(0.1)

                # Final score = average of 3 runs
                final_score    = average_score(run_scores)
                final_label    = score_to_severity(final_score)

                # Two consistency metrics:
                # 1. Score consistency  — all runs within ±1.0 (loose)
                # 2. Label consistency  — all runs produce same severity label (strict)
                score_consistent = is_score_consistent(run_scores, tolerance=1.0)
                label_consistent = is_label_consistent(run_scores)

                label_correct  = (
                    final_label.lower() == official_sev.lower()
                    if final_label != "Unknown" else False
                )
                score_error    = (
                    round(abs(final_score - official_cvss), 1)
                    if final_score is not None else None
                )

                result_row[f"{model_key}_{pt}_final_score"]      = final_score
                result_row[f"{model_key}_{pt}_final_label"]      = final_label
                result_row[f"{model_key}_{pt}_score_error"]      = score_error
                result_row[f"{model_key}_{pt}_label_correct"]    = label_correct
                result_row[f"{model_key}_{pt}_score_consistent"] = score_consistent
                result_row[f"{model_key}_{pt}_label_consistent"] = label_consistent

        results.append(result_row)

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n[✓] Results saved to : {OUTPUT_FILE}")
    print(f"[✓] Total rows       : {len(results_df)}")
    print("\nRun 3_analyze_results.py to see full analysis.")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
