#!/usr/bin/env python3

import pandas as pd
import re
import json
from sklearn.model_selection import train_test_split
import argparse
import os
import numpy as np
import random
from collections import Counter
import requests
from openai import OpenAI
import nltk
from nltk.corpus import wordnet

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
    Includes a 3-dimensional vector representing the three classification labels.
    """
    # Get the class label
    class_label = row['class']
    
    # Create the instruction/prompt
    question = f"{row['tweet']}"
    
    # Create a 3-dimensional vector representing the labels
    # Normalize the counts to create a probability distribution
    total_annotations = row['hate_speech'] + row['offensive_language'] + row['neither']
    if total_annotations > 0:  # Avoid division by zero
        label_vector = [
            float(row['hate_speech']) / total_annotations,
            float(row['offensive_language']) / total_annotations,
            float(row['neither']) / total_annotations
        ]
    else:
        label_vector = [0.0, 0.0, 0.0]
    
    return {
        "Question": question,
        "Label": label_vector
    }

def print_label_distribution(df, dataset_name):
    """
    Print the distribution of samples across different classes and categories.
    """
    print(f"\n=== {dataset_name} Label Distribution ===")
    
    # Class distribution (0, 1, 2)
    print("Class distribution:")
    class_counts = df['class'].value_counts().sort_index()
    for class_id, count in class_counts.items():
        class_name = "hate speech" if class_id == 0 else "offensive language" if class_id == 1 else "neither"
        print(f"  Class {class_id} ({class_name}): {count} samples ({count/len(df)*100:.2f}%)")
    
    # Average annotation counts
    print("\nAverage annotation counts per tweet:")
    print(f"  Hate speech: {df['hate_speech'].mean():.2f}")
    print(f"  Offensive language: {df['offensive_language'].mean():.2f}")
    print(f"  Neither: {df['neither'].mean():.2f}")
    
    # Dominant label analysis
    print("\nDominant label distribution:")
    # Get the dominant label for each tweet
    df['dominant_label'] = df[['hate_speech', 'offensive_language', 'neither']].idxmax(axis=1)
    dominant_counts = df['dominant_label'].value_counts()
    for label, count in dominant_counts.items():
        print(f"  {label}: {count} samples ({count/len(df)*100:.2f}%)")

def handle_imbalance(df_orig, method='random_over', random_state=42):
    """
    Apply various resampling techniques to handle class imbalance.
    
    Args:
        df_orig: pandas DataFrame with the data
        method: resampling method to use. Options:
            'random_over' - Random oversampling
            'random_under' - Random undersampling
            'class_weights' - Calculate class weights (doesn't modify data)
            'combined' - Combination of oversampling minority and undersampling majority
        random_state: Random seed for reproducibility
    
    Returns:
        rebalanced DataFrame or original DataFrame with class_weights added
    """
    print(f"\n=== Applying {method.upper()} to Handle Class Imbalance ===")
    
    # Make a copy to avoid modifying the original
    df = df_orig.copy()
    
    # Create a temporary index column to track rows
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'original_index'}, inplace=True)
    
    # Get the class distribution
    class_counts = df['class'].value_counts().sort_index()
    print(f"Original class distribution: {dict(sorted(Counter(df['class']).items()))}")
    
    if method == 'random_over':
        # Separate dataframes by class
        class_0 = df[df['class'] == 0]
        class_1 = df[df['class'] == 1]
        class_2 = df[df['class'] == 2]
        
        # Get the size of the majority class
        max_size = max(len(class_0), len(class_1), len(class_2))
        
        # Oversample minority classes
        if len(class_0) < max_size:
            class_0 = class_0.sample(max_size, replace=True, random_state=random_state)
        if len(class_2) < max_size:
            class_2 = class_2.sample(max_size, replace=True, random_state=random_state)
        
        # Combine the balanced classes
        df_balanced = pd.concat([class_0, class_1, class_2])
        # Shuffle the data
        df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
    elif method == 'random_under':
        # Separate dataframes by class
        class_0 = df[df['class'] == 0]
        class_1 = df[df['class'] == 1]
        class_2 = df[df['class'] == 2]
        
        # Get the size of the minority class
        min_size = min(len(class_0), len(class_1), len(class_2))
        
        # Undersample majority classes
        if len(class_1) > min_size:
            class_1 = class_1.sample(min_size, random_state=random_state)
        if len(class_2) > min_size:
            class_2 = class_2.sample(min_size, random_state=random_state)
        
        # Combine the balanced classes
        df_balanced = pd.concat([class_0, class_1, class_2])
        # Shuffle the data
        df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
    elif method == 'class_weights':
        # Calculate class weights - inversely proportional to class frequencies
        total_samples = len(df)
        n_classes = len(class_counts)
        
        class_weights = {}
        for class_id, count in class_counts.items():
            class_weights[class_id] = total_samples / (n_classes * count)
        
        print(f"Class weights: {class_weights}")
        
        # Add class_weight column to the DataFrame
        df['class_weight'] = df['class'].map(class_weights)
        df_balanced = df
        
    elif method == 'combined':
        # Separate dataframes by class
        class_0 = df[df['class'] == 0]  # minority class (hate speech)
        class_1 = df[df['class'] == 1]  # majority class (offensive language)
        class_2 = df[df['class'] == 2]  # medium class (neither)
        
        # Determine target sizes
        # Oversample hate speech to match "neither" class size
        target_size_minority = len(class_2)
        # Undersample offensive language to reduce but not equalize completely
        target_size_majority = int(len(class_1) * 0.5)  # Reduce by half
        
        # Oversample minority class (hate speech)
        if len(class_0) < target_size_minority:
            class_0 = class_0.sample(target_size_minority, replace=True, random_state=random_state)
        
        # Undersample majority class (offensive language)
        if len(class_1) > target_size_majority:
            class_1 = class_1.sample(target_size_majority, random_state=random_state)
        
        # Combine the adjusted classes
        df_balanced = pd.concat([class_0, class_1, class_2])
        # Shuffle the data
        df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    else:
        print(f"Warning: Unknown rebalancing method '{method}'. Using original data.")
        df_balanced = df
    
    # Print new class distribution
    print(f"New class distribution: {dict(sorted(Counter(df_balanced['class']).items()))}")
    
    return df_balanced

def augment_with_gpt4o(df, minority_class=0, n_samples=1000, api_key=None, batch_size=10):
    """
    Generate synthetic examples for minority class using OpenAI's GPT-4o model.
    
    Args:
        df: pandas DataFrame with the data
        minority_class: class ID to augment (default 0 for hate speech)
        n_samples: target number of samples to add
        api_key: OpenAI API key
        batch_size: Number of examples to generate in each API call
    
    Returns:
        DataFrame with added synthetic examples
    """
    print(f"\n=== Augmenting Dataset with GPT-4o Generated Examples ===")
    
    if not api_key:
        print("Error: OpenAI API key is required for GPT-4o augmentation")
        return df
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Get examples from minority class to use as prompts
    minority_samples = df[df['class'] == minority_class]['tweet'].sample(
        min(10, sum(df['class'] == minority_class)), 
        random_state=42
    ).tolist()
    
    # Example prompt template for GPT-4o
    system_prompt = """You are an AI research assistant helping to generate balanced training data for a hate speech detection model. 
    Your task is to generate examples that would be classified as hate speech (class 0) for training purposes only.
    The generated examples should be diverse, realistic, and clearly belonging to the hate speech category.
    The examples should be between 1-3 sentences each.
    
    Respond ONLY with a valid JSON array of strings, with each string being a synthetic example.
    Format: ["example 1", "example 2", ...]
    """
    
    user_prompt_template = """I need to generate training examples similar to these hate speech examples from my dataset:
    
    {}
    
    Please generate exactly {} synthetic examples of text that would be classified as hate speech.
    Respond ONLY with a valid JSON array of strings.
    """
    
    # Fill the prompt with examples
    example_text = "\n".join([f"- {s}" for s in minority_samples])
    
    synthetic_examples = []
    num_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
    
    try:
        for i in range(num_batches):
            current_batch_size = min(batch_size, n_samples - len(synthetic_examples))
            
            # Skip if we've already generated enough examples
            if current_batch_size <= 0:
                break
                
            print(f"Generating batch {i+1}/{num_batches} ({current_batch_size} examples)...")
            
            user_prompt = user_prompt_template.format(example_text, current_batch_size)
            
            # Call the OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            try:
                # Extract the JSON array
                json_str = response_text.strip()
                if not json_str.startswith('{'):
                    # If not a proper JSON object, try to extract array
                    match = re.search(r'\[(.*)\]', json_str, re.DOTALL)
                    if match:
                        json_str = f'{{"examples": [{match.group(1)}]}}'
                    else:
                        json_str = f'{{"examples": []}}'
                
                # Parse the JSON
                parsed_json = json.loads(json_str)
                
                # Get the examples - handle different possible JSON structures
                if "examples" in parsed_json:
                    batch_examples = parsed_json["examples"]
                elif isinstance(parsed_json, list):
                    batch_examples = parsed_json
                else:
                    # Try to find any array in the response
                    for key, value in parsed_json.items():
                        if isinstance(value, list):
                            batch_examples = value
                            break
                    else:
                        batch_examples = []
                        print(f"Warning: Could not parse examples from response: {json_str}")
                
                # Add the examples
                synthetic_examples.extend(batch_examples)
                print(f"Successfully generated {len(batch_examples)} examples in this batch")
                
            except Exception as e:
                print(f"Error parsing API response: {e}")
                print(f"Response text: {response_text}")
                continue
                
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
    
    print(f"Total examples generated: {len(synthetic_examples)}")
    
    # Create new DataFrame rows for synthetic examples
    synthetic_rows = []
    for text in synthetic_examples:
        # Ensure text is a string
        if not isinstance(text, str):
            continue
            
        synthetic_rows.append({
            'tweet': text,
            'clean_tweet': clean_text(text),
            'class': minority_class,
            'hate_speech': 3,  # Higher count for hate speech annotation
            'offensive_language': 0,
            'neither': 0,
            'original_index': -1,  # Flag as synthetic
            'dominant_label': 'hate_speech'
        })
    
    # Create DataFrame with synthetic examples
    synthetic_df = pd.DataFrame(synthetic_rows)
    
    # Combine with original DataFrame
    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
    
    print(f"Added {len(synthetic_rows)} synthetic examples. New dataset size: {len(augmented_df)}")
    return augmented_df

def eda_augmentation(df, minority_class=0, n_samples=1000, alpha=0.1):
    """
    Apply Easy Data Augmentation techniques to generate more examples.
    Techniques include: synonym replacement, random insertion, random swap, random deletion.
    """
    # Download necessary NLTK resources
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    print(f"\n=== Augmenting Dataset with EDA Techniques ===")
    
    def get_synonyms(word):
        """Get synonyms for a word."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word:
                    synonyms.add(synonym)
        return list(synonyms)
    
    def synonym_replacement(text, n):
        """Replace n words in the text with their synonyms."""
        words = nltk.word_tokenize(text)
        new_words = words.copy()
        random_word_indices = random.sample(range(len(words)), min(n, len(words)))
        
        for idx in random_word_indices:
            word = words[idx]
            synonyms = get_synonyms(word)
            if synonyms:
                new_words[idx] = random.choice(synonyms)
        
        return ' '.join(new_words)
    
    def random_deletion(text, p):
        """Randomly delete words from text with probability p."""
        words = nltk.word_tokenize(text)
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        if not new_words:
            rand_idx = random.randrange(len(words))
            return words[rand_idx]
        
        return ' '.join(new_words)
    
    def random_swap(text, n):
        """Randomly swap n pairs of words in the text."""
        words = nltk.word_tokenize(text)
        new_words = words.copy()
        
        for _ in range(min(n, len(words) // 2)):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    # Get samples from the minority class
    minority_samples = df[df['class'] == minority_class]['tweet'].tolist()
    
    # Decide how many examples to generate
    n_to_generate = min(n_samples, max(0, n_samples - len(minority_samples)))
    
    # Apply EDA techniques to generate new examples
    augmented_samples = []
    
    for _ in range(n_to_generate):
        # Pick a random sample from minority class
        original_text = random.choice(minority_samples)
        
        # Randomly choose an augmentation technique
        technique = random.choice(['synonym', 'deletion', 'swap'])
        
        if technique == 'synonym':
            n_words = max(1, int(len(original_text.split()) * alpha))
            augmented_text = synonym_replacement(original_text, n_words)
        elif technique == 'deletion':
            augmented_text = random_deletion(original_text, alpha)
        else:  # swap
            n_words = max(1, int(len(original_text.split()) * alpha))
            augmented_text = random_swap(original_text, n_words)
        
        augmented_samples.append({
            'tweet': augmented_text,
            'clean_tweet': clean_text(augmented_text),
            'class': minority_class,
            'hate_speech': 3,  # Higher count for hate speech annotation
            'offensive_language': 0,
            'neither': 0,
            'original_index': -1,  # Flag as synthetic
            'dominant_label': 'hate_speech'
        })
    
    # Create DataFrame with augmented examples
    augmented_df = pd.DataFrame(augmented_samples)
    
    # Combine with original DataFrame
    result_df = pd.concat([df, augmented_df], ignore_index=True)
    
    print(f"Added {len(augmented_samples)} augmented examples using EDA. New dataset size: {len(result_df)}")
    return result_df

def process_data(input_csv, output_dir, test_size=0.2, val_size=0.1, random_state=42, 
                 balance_method='random_over', apply_balance=True, augmentation=None,
                 augment_samples=1000, api_key=None):
    """
    Process the hate speech dataset and handle class imbalance.
    
    Args:
        input_csv: Path to the input CSV file
        output_dir: Directory to save processed data
        test_size: Fraction of data for testing
        val_size: Fraction of remaining data for validation
        random_state: Random seed for reproducibility
        balance_method: Method to handle class imbalance
        apply_balance: Whether to apply balancing
        augmentation: Augmentation method ('gpt4o', 'eda', or None)
        augment_samples: Number of samples to generate with augmentation
        api_key: OpenAI API key for GPT-4o
    """
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
    
    # Print dataset overview
    print("\n=== Dataset Overview ===")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Print label distribution for the entire dataset
    print_label_distribution(df, "Full Dataset")
    
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
    
    # Print label distribution for each split
    print_label_distribution(train_df, "Training Set (Before Augmentation/Balancing)")
    print_label_distribution(val_df, "Validation Set")
    print_label_distribution(test_df, "Test Set")
    
    # Apply augmentation if requested (before balancing)
    original_train_size = len(train_df)
    if augmentation:
        if augmentation == 'gpt4o':
            if not api_key:
                print("Warning: OpenAI API key is required for GPT-4o augmentation. Skipping.")
            else:
                train_df = augment_with_gpt4o(
                    train_df, 
                    minority_class=0, 
                    n_samples=augment_samples,
                    api_key=api_key
                )
        elif augmentation == 'eda':
            train_df = eda_augmentation(
                train_df,
                minority_class=0,
                n_samples=augment_samples
            )
        else:
            print(f"Warning: Unknown augmentation method '{augmentation}'. Skipping augmentation.")
        
        print_label_distribution(train_df, "Training Set (After Augmentation)")
    
    # Handle class imbalance if requested (after augmentation)
    if apply_balance:
        train_df = handle_imbalance(train_df, method=balance_method, random_state=random_state)
        
        # Print the balanced training set distribution
        print_label_distribution(train_df, "Training Set (After Balancing)")
    
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
    
    print(f"\n=== Output Files ===")
    print(f"Training data saved to {train_json_path} and {train_jsonl_path} ({len(train_formatted)} samples)")
    print(f"Validation data saved to {val_json_path} ({len(val_formatted)} samples)")
    print(f"Test data saved to {test_json_path} ({len(test_formatted)} samples)")
    
    # If class weights were calculated, save them separately
    if balance_method == 'class_weights' and apply_balance:
        # Extract unique class weights
        class_weights = {int(class_id): weight for class_id, weight in 
                        train_df[['class', 'class_weight']].drop_duplicates().values}
        
        # Save class weights
        weights_path = os.path.join(output_dir, "class_weights.json")
        with open(weights_path, 'w') as f:
            json.dump(class_weights, f, indent=2)
        
        print(f"Class weights saved to {weights_path}")
    
    # Save summary statistics
    summary = {
        "dataset_stats": {
            "total_samples": len(df),
            "training_original": original_train_size,
            "training_after_augmentation": len(train_df) if augmentation else original_train_size,
            "training_after_balancing": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "class_distribution": {
            "original": dict(sorted(Counter(df['class']).items())),
            "training_final": dict(sorted(Counter(train_df['class']).items())),
            "validation": dict(sorted(Counter(val_df['class']).items())),
            "test": dict(sorted(Counter(test_df['class']).items())),
        },
        "processing_params": {
            "test_size": test_size,
            "val_size": val_size,
            "random_state": random_state,
            "balance_method": balance_method,
            "apply_balance": apply_balance,
            "augmentation": augmentation,
            "augment_samples": augment_samples if augmentation else 0,
        }
    }
    
    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processing summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Process hate speech dataset for language model fine-tuning')
    
    parser.add_argument('--input_csv', type=str, default="../data/labeled_data.csv", 
                        help='Path to the input CSV file')
    parser.add_argument('--output_dir', type=str, default="../data/llm_processed_data/", 
                        help='Directory to save the processed data')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1, 
                        help='Fraction of data to use for validation')
    parser.add_argument('--random_state', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--balance_method', type=str, default='random_over',
                        choices=['random_over', 'random_under', 'class_weights', 'combined', 'none'],
                        help='Method to handle class imbalance')
    parser.add_argument('--no_balance', action='store_false', dest='apply_balance',
                        help='Skip class balancing')
    parser.add_argument('--augmentation', type=str, default=None,
                        choices=['gpt4o', 'eda', None],
                        help='Method to augment minority class data')
    parser.add_argument('--augment_samples', type=int, default=10,
                        help='Number of samples to generate with augmentation')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key (for GPT-4o augmentation)')
    
    args = parser.parse_args()
    
    # If balance_method is 'none', set apply_balance to False
    if args.balance_method == 'none':
        args.apply_balance = False
    
    process_data(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        balance_method=args.balance_method,
        apply_balance=args.apply_balance,
        augmentation=args.augmentation,
        augment_samples=args.augment_samples,
        api_key=args.api_key
    )

if __name__ == "__main__":
    main()