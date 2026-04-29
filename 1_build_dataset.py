"""
Script 1: Build CVE Dataset from NVD API
=========================================
Uses curl via temp file to bypass Python urllib/SSL issues.
Fetches 80 CVEs published in 2025, stratified by severity.

Usage:
    python3 1_build_dataset.py
"""

import subprocess
import json
import os
import tempfile
import pandas as pd
import time

# ── Configuration ─────────────────────────────────────────────
NVD_API_URL      = "https://services.nvd.nist.gov/rest/json/cves/2.0"
TARGET_PER_CLASS = 100
SEVERITY_CLASSES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
OUTPUT_FILE      = "cve_dataset.csv"

SEVERITY_RANGES = {
    "LOW":      (0.1, 3.9),
    "MEDIUM":   (4.0, 6.9),
    "HIGH":     (7.0, 8.9),
    "CRITICAL": (9.0, 10.0),
}

# ── Fetch using curl → temp file ──────────────────────────────
def fetch(severity, max_results=500):
    """Write curl output to temp file — avoids subprocess stdout issues."""
    url = (
        f"{NVD_API_URL}"
        f"?cvssV3Severity={severity}"
        f"&pubStartDate=2025-01-01T00:00:00.000"
        f"&pubEndDate=2025-12-31T23:59:59.999"
        f"&resultsPerPage={max_results}"
        f"&startIndex=0"
    )

    # Write to temp file instead of capturing stdout
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.close()

    cmd = f'curl -s --max-time 30 "{url}" -o "{tmp.name}"'
    result = subprocess.run(cmd, shell=True, timeout=35)

    try:
        with open(tmp.name, "r") as f:
            content = f.read()
        os.unlink(tmp.name)

        if not content.strip():
            print(f"  [ERROR] Empty response for {severity}")
            return []

        data = json.loads(content)
        return data.get("vulnerabilities", [])
    except Exception as e:
        print(f"  [ERROR] {e}")
        return []

# ── Extract ───────────────────────────────────────────────────
def extract(vuln, severity):
    try:
        cve    = vuln["cve"]
        cve_id = cve["id"]

        # Get English description
        desc = next(
            (d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"),
            None
        )
        if not desc or len(desc) < 50:
            return None

        # Skip if CVE ID in description
        if cve_id.lower() in desc.lower():
            return None

        # Get CVSS v3.1 — prefer Primary NVD score
        metrics   = cve.get("metrics", {})
        cvss_list = metrics.get("cvssMetricV31", metrics.get("cvssMetricV30", []))
        if not cvss_list:
            return None

        primary = next((c for c in cvss_list if c.get("type") == "Primary"), cvss_list[0])
        score   = float(primary["cvssData"]["baseScore"])

        # Validate score range matches severity class
        lo, hi = SEVERITY_RANGES[severity]
        if not (lo <= score <= hi):
            return None

        return {
            "CVE_ID":            cve_id,
            "Description":       desc.strip(),
            "Official_CVSS":     score,
            "Official_Severity": severity.capitalize(),
            "Published":         cve.get("published", "")[:10],
        }
    except Exception:
        return None

# ── Main ──────────────────────────────────────────────────────
def build():
    print("=" * 55)
    print("  CVE Dataset Builder — NVD 2025")
    print("=" * 55)

    all_rows = []

    for sev in SEVERITY_CLASSES:
        print(f"\n[+] Fetching {sev} CVEs...")
        raw = fetch(sev)
        print(f"    Raw from NVD     : {len(raw)}")

        rows = []
        for v in raw:
            info = extract(v, sev)
            if info:
                rows.append(info)
            if len(rows) >= TARGET_PER_CLASS:
                break

        print(f"    Usable           : {len(rows)}")
        all_rows.extend(rows)
        time.sleep(1.0)

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("\n[ERROR] No CVEs collected.")
        return

    df_out = (
        df.groupby("Official_Severity")
        .apply(lambda x: x.head(TARGET_PER_CLASS))
        .reset_index(drop=True)
    )

    df_out.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 55)
    print(f"  Saved to         : {OUTPUT_FILE}")
    print(f"  Total CVEs       : {len(df_out)}")
    print(f"  CVSS range       : {df_out['Official_CVSS'].min()} – {df_out['Official_CVSS'].max()}")
    print(f"  Mean CVSS        : {df_out['Official_CVSS'].mean():.2f}")
    print("\n  Distribution:")
    print(df_out["Official_Severity"].value_counts().to_string())
    print("=" * 55)

if __name__ == "__main__":
    build()
