#!/usr/bin/env python
import pandas as pd
import re
import json
from sklearn.model_selection import train_test_split
import argparse
import os

def clean_text(text):
    """
    Clean the input text by:
      - Converting to lowercase
      - Removing URLs, mentions, hashtags, punctuation, and extra spaces
    """
    # Lowercase the text
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (you may choose to keep the word; here we remove the '#' symbol)
    text = re.sub(r'#', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_for_lm(row):
    """
    Format a row for language model fine-tuning.
    Creates a prompt-response pair suitable for instruction fine-tuning.
    """
    # Get the class label
    class_label = row['class']
    if class_label == 0:
        label_text = "hate speech"
    elif class_label == 1:
        label_text = "offensive language"
    else:
        label_text = "neither hate speech nor offensive language"
    
    # Create the instruction/prompt
    question = f"Analyze the following tweet for hate speech or offensive language: \"{row['tweet']}\""
    
    # Create a detailed analysis (for CoT - Chain of Thought)
    analysis = (f"This tweet contains language that can be classified as {label_text}. "
               f"The tweet has been reviewed by multiple annotators, with {row['hate_speech']} "
               f"classifying it as hate speech, {row['offensive_language']} as offensive language, "
               f"and {row['neither']} as neither.")
    
    # Create the response
    response = f"This tweet contains {label_text}."
    
    return {
        "Question": question,
        "Complex_CoT": analysis,
        "Response": response
    }

def process_data(input_csv, output_dir, test_size=0.2, val_size=0.1, random_state=42):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file; adjust quoting if necessary
    df = pd.read_csv(input_csv, quotechar='"')
    
    # Ensure required columns exist
    required_columns = ['tweet', 'class', 'hate_speech', 'offensive_language', 'neither']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the dataset.")
    
    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_columns)
    
    # Clean the tweet text
    df['clean_tweet'] = df['tweet'].apply(clean_text)
    
    # Print the class distribution
    print("Class distribution:")
    print(df['class'].value_counts())
    
    # Split the data into training+validation and test sets
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['class'], random_state=random_state
    )
    
    # Further split training+validation into training and validation sets.
    # Calculate relative validation size from the remaining data.
    val_relative_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative_size, stratify=train_val_df['class'], random_state=random_state
    )
    
    # Format data for language model fine-tuning
    train_formatted = [format_for_lm(row) for _, row in train_df.iterrows()]
    val_formatted = [format_for_lm(row) for _, row in val_df.iterrows()]
    test_formatted = [format_for_lm(row) for _, row in test_df.iterrows()]
    
    # Save the formatted data as JSON files
    train_json_path = os.path.join(output_dir, "train.json")
    val_json_path = os.path.join(output_dir, "val.json")
    test_json_path = os.path.join(output_dir, "test.json")
    
    with open(train_json_path, 'w') as f:
        json.dump(train_formatted, f, indent=2)
    
    with open(val_json_path, 'w') as f:
        json.dump(val_formatted, f, indent=2)
    
    with open(test_json_path, 'w') as f:
        json.dump(test_formatted, f, indent=2)
    
    # Also save a single JSONL file for easy loading with datasets library
    train_jsonl_path = os.path.join(output_dir, "train.jsonl")
    with open(train_jsonl_path, 'w') as f:
        for item in train_formatted:
            f.write(json.dumps(item) + '\n')
    
    print(f"Training data saved to {train_json_path} and {train_jsonl_path} ({len(train_formatted)} samples)")
    print(f"Validation data saved to {val_json_path} ({len(val_formatted)} samples)")
    print(f"Test data saved to {test_json_path} ({len(test_formatted)} samples)")

def main():
    parser = argparse.ArgumentParser(
        description="Process hate speech dataset (CSV format) for fine-tuning a language model."
    )
    parser.add_argument("--input_csv", type=str, default="data/labeled_data.csv",
                        help="Path to the input CSV file")
    parser.add_argument("--output_dir", type=str, default="data/processed_data/",
                        help="Directory to save the processed data files")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use for test set (default: 0.2)")
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="Fraction of training data to use for validation (default: 0.1)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    process_data(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

if __name__ == "__main__":
    main()
