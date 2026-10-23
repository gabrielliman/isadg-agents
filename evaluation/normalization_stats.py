import csv
import statistics
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(ROOT, "results", "metrics", "normalization_tokens.csv")

def safe_mean(data):
    return statistics.mean(data) if data else 0.0

def safe_stdev(data):
    return statistics.stdev(data) if len(data) > 1 else 0.0

def normalization_statistics():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run analyze_normalization.py first.")
        return

    invalid_tok_in = []
    invalid_tok_out = []

    valid_tok_in = []
    valid_tok_out = []
    valid_delta = []

    print(f"Reading data from {os.path.relpath(INPUT_CSV, ROOT)}...\n")

    with open(INPUT_CSV, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_tok = int(row["Raw_Tokens"])
            norm_tok = int(row["Norm_Tokens"])
            delta = int(row["Char_Delta"])

            if row["Had_Omission"] == "Yes":
                invalid_tok_in.append(raw_tok)
                invalid_tok_out.append(norm_tok)
            else:
                valid_tok_in.append(raw_tok)
                valid_tok_out.append(norm_tok)
                valid_delta.append(delta)

    print("=" * 65)
    print("INTRALINGUAL NORMALIZATION STATISTICS")
    print("=" * 65)

    print("\n[!] INVALID NORMALIZATIONS (omission / severe summarization)")
    print("-" * 65)
    print(f"Number of files       : {len(invalid_tok_in)}")
    if invalid_tok_in:
        print(f"Input tokens (mean)   : {safe_mean(invalid_tok_in):.2f} (± {safe_stdev(invalid_tok_in):.2f})")
        print(f"Output tokens (mean)  : {safe_mean(invalid_tok_out):.2f} (± {safe_stdev(invalid_tok_out):.2f})")

    print("\n[+] VALID NORMALIZATIONS (structure preserved)")
    print("-" * 65)
    print(f"Number of files       : {len(valid_tok_in)}")
    if valid_tok_in:
        print(f"Input tokens (mean)   : {safe_mean(valid_tok_in):.2f} (± {safe_stdev(valid_tok_in):.2f})")
        print(f"Output tokens (mean)  : {safe_mean(valid_tok_out):.2f} (± {safe_stdev(valid_tok_out):.2f})")
        print(f"Length change (mean)  : {safe_mean(valid_delta):.2f} characters (± {safe_stdev(valid_delta):.2f})")

    print("\n" + "=" * 65)

if __name__ == "__main__":
    normalization_statistics()
