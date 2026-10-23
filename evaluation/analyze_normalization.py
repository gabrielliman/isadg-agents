import os
import re
import difflib
import csv
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "results", "raw_texts")
NORM_DIR = os.path.join(ROOT, "results", "normalized_texts")
OUTPUT_CSV = os.path.join(ROOT, "results", "metrics", "normalization_tokens.csv")

def clean_text(text):
    """Removes line breaks and collapses whitespace to focus on content."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    return re.sub(r'\s+', ' ', text).strip()

def extract_words(text):
    """Extracts lowercase words without attached punctuation.
    Serves as a quick approximation of the lexical token count."""
    return re.findall(r'\b\w+\b', text.lower())

def analyze_files():
    if not os.path.exists(RAW_DIR) or not os.path.exists(NORM_DIR):
        print("Error: artifact folders not found.")
        return

    raw_files = {f.replace('_raw.txt', ''): f for f in os.listdir(RAW_DIR) if f.endswith('_raw.txt')}
    norm_files = {f.replace('_norm.txt', ''): f for f in os.listdir(NORM_DIR) if f.endswith('_norm.txt')}

    all_base_names = set(raw_files.keys()).union(set(norm_files.keys()))

    missing_files = []
    rows = []
    lexical_changes = Counter()

    print("Cross-analyzing artifacts (this may take a few seconds)...\n")

    for base_name in sorted(all_base_names):
        raw_file = raw_files.get(base_name)
        norm_file = norm_files.get(base_name)

        # 1. Missing files check
        if not raw_file:
            missing_files.append(f"[MISSING RAW] {base_name}_norm.txt exists, but the raw file is missing.")
            continue
        if not norm_file:
            missing_files.append(f"[MISSING NORMALIZED] {base_name}_raw.txt exists, but the agent failed to normalize it.")
            continue

        raw_path = os.path.join(RAW_DIR, raw_file)
        norm_path = os.path.join(NORM_DIR, norm_file)

        try:
            with open(raw_path, 'r', encoding='utf-8') as fr, open(norm_path, 'r', encoding='utf-8') as fn:
                raw_text = fr.read()
                norm_text = fn.read()

            # Clean text for character counts
            raw_clean = clean_text(raw_text)
            norm_clean = clean_text(norm_text)

            raw_len = len(raw_clean)
            norm_len = len(norm_clean)
            char_delta = norm_len - raw_len

            # Word extraction for "token" counts and difflib analysis
            raw_words = extract_words(raw_clean)
            norm_words = extract_words(norm_clean)

            raw_tokens = len(raw_words)
            norm_tokens = len(norm_words)

            had_omission = False

            # 2. Change mapping and boolean omission detection
            matcher = difflib.SequenceMatcher(None, raw_words, norm_words)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():

                if tag == 'delete':
                    deleted_len = i2 - i1
                    if deleted_len > 3: # More than 3 consecutive words deleted
                        had_omission = True

                elif tag == 'replace':
                    orig_len = i2 - i1
                    new_len = j2 - j1

                    if orig_len > 10 and new_len < (orig_len * 0.4): # Drastic summarization
                        had_omission = True

                    # Feed the lexical counter only with proportional substitutions
                    elif abs(orig_len - new_len) <= 2:
                        old_span = " ".join(raw_words[i1:i2])
                        new_span = " ".join(norm_words[j1:j2])
                        lexical_changes[(old_span, new_span)] += 1

            rows.append({
                "full_name": base_name, # Full name for the CSV
                "short_name": base_name[:30] + "..." if len(base_name) > 30 else base_name,
                "raw_len": raw_len,
                "norm_len": norm_len,
                "delta": char_delta,
                "raw_tok": raw_tokens,
                "norm_tok": norm_tokens,
                "omission": "Yes" if had_omission else "No"
            })

        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    # ==========================================
    # CSV EXPORT
    # ==========================================
    if rows:
        try:
            with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["File", "Raw_Chars", "Norm_Chars", "Char_Delta", "Raw_Tokens", "Norm_Tokens", "Had_Omission"])
                for row in rows:
                    writer.writerow([
                        row['full_name'],
                        row['raw_len'],
                        row['norm_len'],
                        row['delta'],
                        row['raw_tok'],
                        row['norm_tok'],
                        row['omission']
                    ])
        except Exception as e:
            print(f"Error saving CSV file: {e}")

    # ==========================================
    # TERMINAL OUTPUT
    # ==========================================
    print("-" * 105)
    print("1. MISSING FILES REPORT")
    print("-" * 105)
    if missing_files:
        for missing in missing_files:
            print(missing)
    else:
        print("No missing files! Every raw file has its normalized counterpart.")

    print("\n" + "-" * 105)
    print("2. LENGTH AND TOKEN DIFFERENCE TABLE")
    print("-" * 105)
    header = f"{'FILE NAME (TRUNCATED)':<33} | {'RAW CHR':<9} | {'NORM CHR':<9} | {'DELTA':<7} | {'RAW TOK':<8} | {'NORM TOK':<8} | {'OMISSION?'}"
    print(header)
    print("-" * 105)

    for row in rows:
        sign = "+" if row['delta'] > 0 else ""
        print(f"{row['short_name']:<33} | {row['raw_len']:<9} | {row['norm_len']:<9} | {sign}{row['delta']:<6} | {row['raw_tok']:<8} | {row['norm_tok']:<8} | {row['omission']}")

    if rows:
        print(f"\n[+] Data exported to: {os.path.relpath(OUTPUT_CSV, ROOT)}")

    print("\n" + "-" * 105)
    print("3. TOP 30 MOST CHANGED WORDS DURING NORMALIZATION")
    print("-" * 105)
    print(f"{'ARCHAIC / RAW OCR TEXT':<30} -> {'NORMALIZED TEXT':<30} | FREQUENCY")
    print("-" * 105)
    for (old, new), frequency in lexical_changes.most_common(30):
        print(f"{old:<30} -> {new:<30} | {frequency}x")

if __name__ == "__main__":
    analyze_files()
