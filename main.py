import argparse
import os
import json
import torch
import tensorflow as tf
from transformers import (
AutoModelForSequenceClassification,
AutoTokenizer, AutoModel
)
from Code.Dataset import split_path, create_dataloader
from Code.Model import setup_model, MultiTaskModel, train, test
from tqdm import tqdm

def main(args):
    # Clear previous sessions and GPU memory
    tf.keras.backend.clear_session()
    torch.cuda.empty_cache()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the tokenizer and input model
    classes = ['Anger','Disgust','Enjoyment','Fear','Other','Sadness','Surprise']
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    embedding_model = AutoModel.from_pretrained(args.model)
    classification_model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(classes)  # Number of classes
    )
    # clear_output()

    # Adjust the token embeddings size if needed
    embedding_model.resize_token_embeddings(len(tokenizer))


    # Handle test index and data splitting
    if args.test_index is not None and args.test_index > 3:
        train_path, dev_path, test_path = split_path(
            args.test_path, args.test_index, args.train_path, args.dev_path, args.test_path
        )
    elif args.test_index is not None:
        print("Test index out of range. Please provide a valid integer index greater than 3.")
        train_path, dev_path, test_path = args.train_path, args.dev_path, args.test_path
    else:
        train_path, dev_path, test_path = args.train_path, args.dev_path, args.test_path

    # Create data loaders
    train_dataloader = create_dataloader(
        train_path, batch_size=args.batch_size, tokenizer=tokenizer, max_len=args.max_len
    )
    dev_dataloader = create_dataloader(
        dev_path, batch_size=args.batch_size, tokenizer=tokenizer, max_len=args.max_len, shuffle=args.shuffle
    )
    test_dataloader = create_dataloader(
        test_path, batch_size=args.batch_size, tokenizer=tokenizer, max_len=args.max_len
    )

    # Get the first batch of data from the DataLoader
    first_batch = next(iter(test_dataloader))

    # Access input_ids, attention_mask, and labels
    input_ids = first_batch['input_ids']
    attention_mask = first_batch['attention_mask']
    labels = first_batch['label']

    # Print to check
    print(f"Input IDs: {input_ids.size()}")
    print(f"Attention Mask: {attention_mask.size()}")
    print(f"Labels: {labels.shape}")

    # Set up the model and training components
    model, criterion_span, optimizer_spans, device, num_epochs = setup_model(
        model_class = MultiTaskModel, 
        embedding_model = embedding_model, 
        classification_model = classification_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs
    )

    # Train the model
    train(
        model=model,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        criterion=criterion_span,
        optimizer=optimizer_spans,
        device=device,
        num_epochs=num_epochs
    )

    # Test the model and save results
    span_preds, span_targets = test(
        model=model,
        test_dataloader=test_dataloader,
        device=device
    )

    # Save test results to a JSON file
    test_results = {
        "predictions": span_preds.tolist(),
        "targets": span_targets.tolist()
    }
    with open(args.output_json, 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"Test results saved to {args.output_json}")

    # Save the trained model
    model_save_path = os.path.join(args.output_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a multi-task model.")

    # Paths for data
    parser.add_argument('--train_path', type=str, required=True, help="Path to the training data")
    parser.add_argument('--dev_path', type=str, required=True, help="Path to the development data")
    parser.add_argument('--test_path', type=str, required=True, help="Path to the test data")
    parser.add_argument('--test_index', type=int, default=None, help="Index for test split, if applicable")

    # Model training parameters
    parser.add_argument('--model', type=str, default="vinai/phobert-base", help="Pretrained language model")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument('--max_len', type=int, default=64, help="Maximum sequence length for tokenization")
    parser.add_argument('--shuffle', type=bool, default=False, help="Shuffle data during evaluation")

    # Model hyperparameters
    parser.add_argument('--lr', type=float, default=5e-6, help="Learning rate for the optimizer")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for the optimizer")
    parser.add_argument('--num_epochs', type=int, default=2, help="Number of training epochs")

    # Output paths
    parser.add_argument('--output_json', type=str, default="test_results.json", help="Path to save test results JSON")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save the trained model")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
