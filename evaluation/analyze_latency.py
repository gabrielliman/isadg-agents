import os
import re
import glob
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Reads every log: pipeline.log (routing, normalization, ISAD(G)) and
# vision_ocr.log (OCR latencies, whose transcriptions come from the checkpoints)
LOG_DIR = os.path.join(ROOT, "results", "logs")

# The published run logged in Portuguese ("Latência Visão ..."); newer runs log in English.
AGENT_NAMES = {
    "Visão": "Vision (OCR)", "Vision": "Vision (OCR)",
    "Estruturador": "Structurer", "Structurer": "Structurer",
    "Normalização": "Normalization", "Normalization": "Normalization",
    "ISAD(G)": "ISAD(G) (RAG)",
}

def analyze_latency():
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
    if not log_files:
        print(f"Error: no .log files found in '{LOG_DIR}'.")
        return

    latencies = defaultdict(list)

    # Example target: "Latency Vision [Page 10]: 4.56s"
    # Group 1 captures the agent name, group 2 the elapsed time
    pattern = re.compile(r"(?:Latência|Latency)\s+(.*?)\s+\[.*?\]:\s*([\d\.]+)s")

    for log_file in log_files:
        print(f"Reading log '{os.path.basename(log_file)}'...")
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    agent = match.group(1).strip()
                    latencies[AGENT_NAMES.get(agent, agent)].append(float(match.group(2)))
    print()

    if not latencies:
        print("No latency metrics found in the log files.")
        return

    print("=" * 65)
    print("LATENCY METRICS PER AGENT (seconds)")
    print("=" * 65)
    print(f"{'AGENT/TASK':<20} | {'CALLS':<10} | {'MEAN':<10} | {'STD DEV'}")
    print("-" * 65)

    for agent, times in latencies.items():
        n_calls = len(times)
        mean = statistics.mean(times)
        # Standard deviation only when there is more than one sample
        stdev = statistics.stdev(times) if n_calls > 1 else 0.0

        print(f"{agent:<20} | {n_calls:<10} | {mean:<9.2f}s | ± {stdev:.2f}s")

    print("=" * 65)

if __name__ == "__main__":
    analyze_latency()
