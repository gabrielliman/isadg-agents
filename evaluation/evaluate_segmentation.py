import os
import json
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS_DIR = os.path.join(ROOT, "results", "isadg_records")
GROUND_TRUTH_DIR = os.path.join(ROOT, "data", "segmentation_ground_truth")
CHECKPOINT_DIR = os.path.join(ROOT, "results", "transcription_checkpoints")

def load_ground_truth_intervals(txt_path):
    """
    Reads the ground-truth txt and builds the intervals (start page -> end page).
    The end page is inferred from the start page of the next article.
    """
    splits = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',', 1)
            if len(parts) >= 1:
                try:
                    page_num = int(parts[0].strip())
                    title = parts[1].strip() if len(parts) > 1 else "Untitled"
                    splits.append((page_num, title))
                except ValueError:
                    continue

    # Sort to guarantee ordering
    splits.sort(key=lambda x: x[0])

    intervals = []
    for i in range(len(splits)):
        start = splits[i][0]
        title = splits[i][1]

        # This article ends on the page before the next one starts
        if i < len(splits) - 1:
            end = splits[i+1][0] - 1
        else:
            end = "End of PDF"

        intervals.append({"start": start, "end": end, "title": title[:40] + "..." if len(title) > 40 else title})

    return intervals

def load_predicted_intervals(records_dir, edition_year, max_gt_page):
    """
    Reads the generated JSON records for the given year and extracts their page intervals,
    filtering strictly by the file prefix and capping at the size of the original PDF.
    """
    intervals = []

    for filename in os.listdir(records_dir):
        if not filename.endswith(".json"):
            continue

        # GUARD 1: the year must be in the PDF NAME (prefix), not somewhere in the article title
        # e.g. for '1881' accept 'per717703_1881_...'
        name_parts = filename.split('_')
        if not any(edition_year in part for part in name_parts[:2]):
            continue

        path = os.path.join(records_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                start_page = data.get("pagina_inicio")
                end_page = data.get("pagina_fim", "?")
                title = data.get("area_identificacao", {}).get("titulo", filename)

                if start_page is not None and str(start_page).isdigit():
                    start_int = int(start_page)

                    # GUARD 2: ignore any start page beyond the PDF itself
                    if start_int > max_gt_page:
                        continue

                    intervals.append({
                        "start": start_int,
                        "end": int(end_page) if str(end_page).isdigit() else end_page,
                        "title": title[:40] + "..." if len(title) > 40 else title
                    })
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Sort by start page
    intervals.sort(key=lambda x: x["start"])

    # FILL EMPTY GAPS BETWEEN INTERVALS
    filled = []

    # 1. Gap at the beginning (before the first detected article)
    if intervals and isinstance(intervals[0]["start"], int) and intervals[0]["start"] > 1:
        filled.append({
            "start": 1,
            "end": intervals[0]["start"] - 1,
            "title": "[Inferred Empty Interval - Start]"
        })

    # 2. Gaps in between
    for i in range(len(intervals)):
        filled.append(intervals[i])

        if i < len(intervals) - 1:
            current_end = intervals[i]["end"]
            next_start = intervals[i+1]["start"]

            # Numeric end with missing pages between it and the next start
            if isinstance(current_end, int) and isinstance(next_start, int):
                if current_end < next_start - 1:
                    filled.append({
                        "start": current_end + 1,
                        "end": next_start - 1,
                        "title": "[Inferred Empty Interval]"
                    })

    # 3. Gap at the end (between the last article and the end of the PDF)
    if filled:
        last_end = filled[-1]["end"]
        if isinstance(last_end, int) and last_end < max_gt_page:
            filled.append({
                "start": last_end + 1,
                "end": max_gt_page,
                "title": "[Inferred Empty Interval - End]"
            })

    return filled

def compute_metrics(true_starts, predicted_starts):
    """Computes Precision, Recall and F1-Score."""
    true_positives = true_starts.intersection(predicted_starts)
    false_positives = predicted_starts - true_starts
    false_negatives = true_starts - predicted_starts

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1_score

if __name__ == "__main__":
    gt_files = glob.glob(os.path.join(GROUND_TRUTH_DIR, "*.txt"))

    if not gt_files:
        print(f"No .txt files found in {GROUND_TRUTH_DIR}.")
        exit()

    for txt_path in sorted(gt_files):
        edition_year = os.path.basename(txt_path).replace(".txt", "")
        print("="*80)
        print(f"EDITION EVALUATION: {edition_year}")
        print("="*80)

        # 1. Ground-truth intervals
        gt_intervals = load_ground_truth_intervals(txt_path)
        true_set = set([item["start"] for item in gt_intervals])

        # Last page of the edition, taken from its transcription checkpoint so that no
        # interval is inferred beyond the end of the PDF
        checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, f"*_{edition_year}_*_transcribed.json"))
        if checkpoints:
            with open(checkpoints[0], 'r', encoding='utf-8') as f:
                max_page = max(int(k) for k in json.load(f))
        elif isinstance(gt_intervals[-1]["end"], int):
            max_page = gt_intervals[-1]["end"]
        else:
            max_page = gt_intervals[-1]["start"] + 10
        if gt_intervals[-1]["end"] == "End of PDF":
            gt_intervals[-1]["end"] = max_page

        # 2. Predicted intervals (agent)
        predicted_intervals = load_predicted_intervals(RECORDS_DIR, edition_year, max_page)
        predicted_set = set([item["start"] for item in predicted_intervals])

        # 3. Side-by-side interval comparison
        print(f"\n--- INTERVAL MAP ({edition_year}) ---")
        print(f"{'GROUND TRUTH':<40} | {'PREDICTED (AGENTIC PIPELINE)':<40}")
        print("-" * 80)

        max_rows = max(len(gt_intervals), len(predicted_intervals))
        for i in range(max_rows):
            gt_str = ""
            if i < len(gt_intervals):
                p = gt_intervals[i]
                gt_str = f"Pages {p['start']} to {p['end']}"

            pred_str = ""
            if i < len(predicted_intervals):
                g = predicted_intervals[i]

                # Visual marker for a hit or miss on the start page
                status = "[OK]" if g["start"] in true_set else "[FALSE POSITIVE]"
                pred_str = f"Pages {g['start']} to {g['end']} {status}"

            print(f"{gt_str:<40} | {pred_str:<40}")

        # 4. Final metrics
        if not true_set:
            print(f"\nGround truth for {edition_year} is empty or invalid.")
            continue

        tp, fp, fn, precision, recall, f1 = compute_metrics(true_set, predicted_set)

        print(f"\n--- ROUTING METRICS ({edition_year}) ---")
        print(f"True Positives (correct splits)       : {tp}")
        print(f"False Positives (hallucinated splits) : {fp}")
        print(f"False Negatives (missed splits)       : {fn}")
        print(f"F1-Score: {f1:.4f} (Precision: {precision:.2f} | Recall: {recall:.2f})\n")
