import csv
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def calculate_corpus_bleu_scores(csv_filepath):
    """
    Calculates corpus BLEU scores (1 to 4) from a CSV file.

    The CSV file is expected to have at least three columns:
    - Ignored first column (e.g., image name)
    - Second column: original_captions (references, separated by '|')
    - Third column: generated_caption (hypothesis)
    """
    list_of_references = []
    hypotheses = []
    
    # Using a smoothing function to handle cases where n-grams might not be present
    # This is important especially for BLEU-3 and BLEU-4 with smaller datasets
    smoothie = SmoothingFunction().method1

    try:
        with open(csv_filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader) # Skip header row
            print(f"Skipped header: {header}")

            for row_number, row in enumerate(reader, 1):
                if len(row) < 3:
                    print(f"Warning: Row {row_number} has fewer than 3 columns. Skipping: {row}")
                    continue
                
                original_captions_str = row[1]
                generated_caption_str = row[2]

                if not original_captions_str or not generated_caption_str:
                    print(f"Warning: Row {row_number} has empty original or generated caption. Skipping.")
                    continue

                # Split original captions by '|' and tokenize each by space
                references_for_row = [ref.strip().split() for ref in original_captions_str.split('|')]
                # Tokenize generated caption by space
                hypothesis_for_row = generated_caption_str.strip().split()

                if not all(references_for_row) or not hypothesis_for_row:
                    print(f"Warning: Row {row_number} resulted in empty tokenized captions after processing. Skipping.")
                    print(f"Original captions string: '{original_captions_str}'")
                    print(f"Generated caption string: '{generated_caption_str}'")
                    continue
                
                list_of_references.append(references_for_row)
                hypotheses.append(hypothesis_for_row)

    except FileNotFoundError:
        print(f"Error: The file {csv_filepath} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading or processing the CSV: {e}")
        return

    if not hypotheses or not list_of_references:
        print("No valid data to calculate BLEU scores.")
        return

    print(f"\nProcessed {len(hypotheses)} entries for BLEU score calculation.")

    try:
        bleu_1 = corpus_bleu(list_of_references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu_2 = corpus_bleu(list_of_references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu_3 = corpus_bleu(list_of_references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        bleu_4 = corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

        print(f"Corpus BLEU-1: {bleu_1:.4f}")
        print(f"Corpus BLEU-2: {bleu_2:.4f}")
        print(f"Corpus BLEU-3: {bleu_3:.4f}")
        print(f"Corpus BLEU-4: {bleu_4:.4f}")

    except Exception as e:
        print(f"An error occurred during BLEU calculation: {e}")


if __name__ == "__main__":
    # Path to your CSV file
    # Make sure this path is correct and accessible from where you run the script.
    csv_file_path = r"C:\Users\DELL\Downloads\all_images_predictions_diag_v1 (1).csv"
    calculate_corpus_bleu_scores(csv_file_path) 