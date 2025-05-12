from deep_translator import GoogleTranslator
from tqdm import tqdm  # Import tqdm for the progress bar
import time
import os

# Function to translate English captions to Gujarati
def translate_captions(input_file, output_file):
    # Initialize the translator
    translator = GoogleTranslator(source='en', target='gu')

    # Read English captions from the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Check if the output file exists and count the number of lines already translated
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as outfile:
            translated_lines = outfile.readlines()
        start_line = len(translated_lines)  # Number of lines already translated
    else:
        start_line = 0  # Start from the beginning if the output file doesn't exist

    # Open the output file in append mode
    with open(output_file, 'a', encoding='utf-8') as outfile:
        # Translate each caption to Gujarati with a progress bar
        for i, line in enumerate(tqdm(lines[start_line:], desc="Translating", unit="line", initial=start_line, total=len(lines)), start_line + 1):
            retries = 3  # Number of retries for each translation
            for attempt in range(retries):
                try:
                    # Split the line into image identifier and caption
                    image_id, caption = line.strip().split('\t')  # Assuming tab-separated
                    # Translate the caption
                    translated = translator.translate(caption)
                    # Write the translated line to the output file immediately
                    outfile.write(f"{image_id}\t{translated}\n")
                    # Add a delay of 1 second between translations
                    # time.sleep(1)  # Delay to avoid hitting API rate limits
                    break  # Exit the retry loop if translation succeeds
                except Exception as e:
                    if attempt == retries - 1:  # If all retries fail
                        print(f"Error translating line: {line.strip()}. Error: {e}")
                        # Write a blank line if translation fails after retries
                        outfile.write(f"{image_id}\t\n")
                    else:
                        print(f"Retrying line {i} (attempt {attempt + 1})...")
                        time.sleep(2)  # Wait before retrying

    print(f"Translation complete. Translated captions saved to {output_file}")

# Main function
if __name__ == "__main__":
    # File paths (using raw strings to avoid unicode escape errors)
    input_file = r"C:\Users\DELL\Downloads\Flickr8k.token.txt\Flickr8k.token.txt"  # Input file with English captions
    output_file = r"C:\Users\DELL\Desktop\gujarati_captions.txt"  # Output file for Gujarati captions

    # Translate captions
    try:
        translate_captions(input_file, output_file)
    except KeyboardInterrupt:
        print("\nTranslation interrupted by user. Exiting gracefully.")