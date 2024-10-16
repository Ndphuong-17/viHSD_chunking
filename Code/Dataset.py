import pandas as pd
from sklearn.model_selection import train_test_split
import torch 
import ast
import re


def split_path(path, test_index, train_path, dev_path, test_path):
    # Load the data
    data = pd.read_csv(path) # Replace 'data.csv' with your file path

        # Get a list of unique sentence IDs
    unique_sentence_ids = data['sentence_id'].unique()[:test_index] 

    # Split sentence IDs into train (70%), temp (30%)
    train_ids, temp_ids = train_test_split(unique_sentence_ids, test_size=0.30, random_state=42)

    # Further split the temp IDs into test (15%) and dev (15%)
    test_ids, dev_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

    # Create subsets for each split using the sentence IDs
    train_data = data[data['sentence_id'].isin(train_ids)]
    test_data = data[data['sentence_id'].isin(test_ids)]
    dev_data = data[data['sentence_id'].isin(dev_ids)]


    # Save the splits into separate CSV files
    train_data[:-1].to_csv(train_path[:-4] + '_test.csv', index=False)
    test_data[:-1].to_csv(test_path[:-4] + '_test.csv', index=False)
    dev_data[:-1].to_csv(dev_path[:-4] + '_test.csv', index=False)

    train_path = train_path[:-4] + '_test.csv'
    test_path = test_path[:-4] + '_test.csv'
    dev_path = dev_path[:-4] + '_test.csv'

    print(f"Training set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print(f"Development set: {len(dev_data)} samples")

    return train_path, dev_path, test_path



def prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Remove NaN values
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(f"Columns: {df.columns}")

    try:
        texts = df['Text'].tolist()
    except KeyError:
        print("Text column not found.")
        return [], []

    # Parse the 'Tag' column as lists of floats
    binary_spans = []
    for tag in df['Tag'].tolist():
        # Replace spaces with commas to create a valid list representation
        tag = tag.replace(' ', ',')
        tag = tag.replace('[', '[ ').replace(']', ' ]')  # Add space after '[' and before ']' for better readability
        try:
            parsed_tag = ast.literal_eval(tag)  # Safely evaluate the string to a list of floats
            binary_spans.append([float(i) for i in parsed_tag])  # Convert to float
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing tag '{tag}': {e}")
            binary_spans.append([])  # Append an empty list in case of error

    return texts, binary_spans


import torch
import re

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, max_sentences=4):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_sentences = max_sentences  # Max number of sentences to consider

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Convert the list to a tensor of type float (for multilabel)
        label = torch.tensor(label, dtype=torch.float)

        # Split the text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Tokenize each sentence
        sentence_embeddings = []
        for sentence in sentences[:self.max_sentences]:  # Limit to max_sentences
            encoded = self.tokenizer(
                sentence,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)  # (max_length)
            attention_mask = encoded['attention_mask'].squeeze(0)  # (max_length)
            sentence_embeddings.append((input_ids, attention_mask))

        # Pad sentences if there are fewer than max_sentences
        while len(sentence_embeddings) < self.max_sentences:
            # Create dummy tensors filled with 0
#             dummy_input = torch.zeros((self.max_length,), dtype=torch.long)
#             dummy_mask = torch.zeros((self.max_length,), dtype=torch.long)
            sentence_embeddings.append((input_ids, attention_mask))

        # Stack the tokenized sentences into tensors
        input_ids = torch.stack([item[0] for item in sentence_embeddings])  # Shape: (num_sentences, max_length)
        attention_mask = torch.stack([item[1] for item in sentence_embeddings])  # Shape: (num_sentences, max_length)


        return {
            'input_ids': input_ids,          # Tensor of shape (max_sentences, max_length)
            'attention_mask': attention_mask,  # Tensor of shape (max_sentences, max_length)
            'label': label                    # Label for the text
        }


def create_dataloader(data_path, batch_size, tokenizer, max_len, shuffle=True):
    dataset = TextDataset(*prepare_data(data_path), tokenizer, max_len)
    # return texts
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
