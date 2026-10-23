import os
import re
import difflib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "results", "raw_texts")
MANUAL_DIR = os.path.join(ROOT, "data", "transcription_ground_truth")

def extract_words(text):
    """
    Lowercases the text and extracts lexical units only.
    This keeps a different comma or period from counting as an OCR error on the word.
    """
    return re.findall(r'\b\w+\b', text.lower())

def evaluate_ocr_accuracy():
    if not os.path.exists(MANUAL_DIR) or not os.path.exists(RAW_DIR):
        print(f"Error: folders '{MANUAL_DIR}' or '{RAW_DIR}' not found.")
        return

    manual_files = [f for f in os.listdir(MANUAL_DIR) if f.endswith('.txt')]

    if not manual_files:
        print(f"No ground-truth .txt files found in {MANUAL_DIR}.")
        return

    print("=" * 80)
    print("VISUAL TRANSCRIPTION ACCURACY (GROUND TRUTH VS VLM)")
    print("=" * 80)

    for manual_file in sorted(manual_files):
        # Strip the '_manual' suffix to find the matching raw file
        base_name = manual_file.replace('_manual.txt', '').replace('.txt', '')

        raw_path = os.path.join(RAW_DIR, f"{base_name}_raw.txt")
        manual_path = os.path.join(MANUAL_DIR, manual_file)

        if not os.path.exists(raw_path):
            print(f"[WARNING] Ground truth '{manual_file}' has no matching raw file.")
            continue

        try:
            with open(manual_path, 'r', encoding='utf-8') as fm, open(raw_path, 'r', encoding='utf-8') as fr:
                manual_text = fm.read()
                raw_text = fr.read()
        except Exception as e:
            print(f"Error reading files for {base_name}: {e}")
            continue

        manual_words = extract_words(manual_text)  # Ground truth
        raw_words = extract_words(raw_text)        # Model prediction

        total_manual_words = len(manual_words)

        if total_manual_words == 0:
            print(f"[!] File {manual_file} is empty.")
            continue

        # Sequential comparison (Word Error Rate basis)
        matcher = difflib.SequenceMatcher(None, manual_words, raw_words)

        substituted = 0
        inserted = 0
        deleted = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Count the larger side of the changed block as errors
                substituted += max((i2 - i1), (j2 - j1))
            elif tag == 'insert':
                # Model invented words that are not in the ground truth
                inserted += (j2 - j1)
            elif tag == 'delete':
                # Model dropped words that are in the ground truth
                deleted += (i2 - i1)

        total_errors = substituted + inserted + deleted

        # Accuracy could go negative if the model hallucinates more words than the original has.
        # Clamp to zero for readability.
        accuracy = max(0.0, 100 - ((total_errors / total_manual_words) * 100))

        print(f"\nDOCUMENT: {base_name}")
        print("-" * 80)
        print(f"Words in ground truth (manual)    : {total_manual_words}")
        print(f"Words in OCR (VLM)                : {len(raw_words)}")
        print(f"Total errors (edit distance)      : {total_errors}")
        print(f"  -> Substituted words (spelling) : {substituted}")
        print(f"  -> Hallucinated words (inserted): {inserted}")
        print(f"  -> Omitted words (deleted)      : {deleted}")
        print("-" * 80)
        print(f"TRANSCRIPTION ACCURACY (word level): {accuracy:.2f}%")
        print("=" * 80)

if __name__ == "__main__":
    evaluate_ocr_accuracy()
