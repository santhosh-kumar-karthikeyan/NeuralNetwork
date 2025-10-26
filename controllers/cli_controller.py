"""
CLI Controller for handling commands - Two main workflows.
"""
import os
import pickle
import argparse
from typing import Optional
from sklearn.model_selection import train_test_split
from singleLayerPerceptron import SingleLayerPerceptron
from utils.data_utils import load_csv

MODEL_FILE = "./assets/model.pkl"

def load_model(model_path: Optional[str] = None):
    """Load a model from disk. If model_path is None, use default."""
    path = model_path if model_path else MODEL_FILE
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def save_model(model, model_path: Optional[str] = None):
    """Save a model to disk. If model_path is None, use default."""
    path = model_path if model_path else MODEL_FILE
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def workflow_train(args):
    """
    Workflow 1: Train a model with optional evaluation.
    If --evaluate flag is set, split data and show classification report without training output.
    Otherwise, show all training details.
    """
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV file '{args.csv}' not found.")
        return
    
    print(f"\n[INFO] Loading data from {args.csv}...")
    df = load_csv(args.csv)
    
    print(f"\n[CONFIG] Training configuration:")
    print(f"   * Target column: {args.target}")
    print(f"   * Activation function: {args.activation}")
    print(f"   * Learning rate: {args.lr}")
    print(f"   * Threshold: {args.threshold}")
    print(f"   * Max epochs: {args.epochs}")
    
    # If evaluation is requested, split the data
    if args.evaluate:
        test_size = args.test_split / 100.0
        print(f"   * Train/Test split: {100 - args.test_split}%/{args.test_split}%")
        print()
        
        # Split data and reset indices
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        print(f"[DATA] Data split: {len(train_df)} training samples, {len(test_df)} test samples")
        print(f"[TRAIN] Training on training set (intermediate output suppressed)...\n")
        
        # Train on training set (silent)
        model = SingleLayerPerceptron(
            train_df, 
            args.target, 
            threshold=args.threshold, 
            learning_rate=args.lr, 
            max_epochs=args.epochs,
            activation=args.activation,
            verbose=False  # Silent training
        )
        model.fit()
        
        # Evaluate on test set
        X_test = test_df[[c for c in test_df.columns if c != args.target]].to_numpy()
        y_test = test_df[args.target].to_numpy()
        
        print(f"\n[REPORT] Classification Report (Test Set - {len(y_test)} samples):")
        print("=" * 70)
        report_str = model.classification_report(X_test, y_test)
        print(report_str)
        print("=" * 70)
    else:
        print()
        print(f"[TRAIN] Starting training with full training output...\n")
        
        # Train on full data with verbose output
        model = SingleLayerPerceptron(
            df, 
            args.target, 
            threshold=args.threshold, 
            learning_rate=args.lr, 
            max_epochs=args.epochs,
            activation=args.activation,
            verbose=True
        )
        model.fit()
    
    # Save model
    save_model(model, args.model)
    print(f"\n[SUCCESS] Model trained and saved to {args.model if args.model else MODEL_FILE}")

def workflow_predict(args):
    """
    Workflow 2: Make predictions using a trained model.
    """
    print(f"\n[INFO] Loading model from {args.model if args.model else MODEL_FILE}...")
    model = load_model(args.model)
    if not model:
        print(f"[ERROR] Model file not found. Train a model first using 'slp train'.")
        return
    
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV file '{args.csv}' not found.")
        return
    
    print(f"[INFO] Loading data from {args.csv}...")
    df = load_csv(args.csv)
    X = df.to_numpy()
    
    print(f"[PREDICT] Making predictions for {len(X)} samples...\n")
    preds = model.predict(X)
    
    print("Predictions:")
    print("-" * 50)
    for i, p in enumerate(preds):
        print(f"  Sample {i:4d}: {p:>6}")
    print("-" * 50)
    print(f"[SUCCESS] Total predictions made: {len(preds)}\n")

def setup_parser():
    """Set up argument parser with two main workflows."""
    parser = argparse.ArgumentParser(
        description="SingleLayerPerceptron CLI - Two-Workflow Neural Network Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflows:

  [1] TRAIN: Train a model (with optional evaluation)
    slp train --csv data.csv --target y --activation binary --lr 1 --epochs 10
    
    With evaluation (trains on 70%, tests on 30%):
    slp train --csv data.csv --target y --evaluate --test-split 30
    
    Custom train/test split:
    slp train --csv data.csv --target y --evaluate --test-split 20

  [2] PREDICT: Make predictions with a trained model
    slp predict --csv test.csv
    slp predict --csv test.csv --model custom_model.pkl
        """,
        prog="slp"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Choose a workflow", required=False)

    # TRAIN workflow
    train_parser = subparsers.add_parser(
        "train", 
        help="Train a new model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Train with optional evaluation using train/test split"
    )
    train_parser.add_argument("--csv", required=True, help="Path to training CSV file")
    train_parser.add_argument("--target", required=True, help="Target column name")
    train_parser.add_argument(
        "--activation", 
        choices=["binary", "bipolar", "sigmoid", "threshold"],
        default="binary",
        help="Activation function (default: binary)"
    )
    train_parser.add_argument("--threshold", type=float, default=0, help="Activation threshold (default: 0)")
    train_parser.add_argument("--lr", type=float, default=1, help="Learning rate (default: 1)")
    train_parser.add_argument("--epochs", type=int, default=10, help="Maximum epochs (default: 10)")
    train_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model on test set (suppresses training output, shows only report)"
    )
    train_parser.add_argument(
        "--test-split",
        type=float,
        default=30,
        help="Test set percentage for evaluation (default: 30)"
    )
    train_parser.add_argument("--model", default=MODEL_FILE, help=f"Model output path (default: {MODEL_FILE})")

    # PREDICT workflow
    predict_parser = subparsers.add_parser(
        "predict", 
        help="Make predictions with trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Load a trained model and make predictions on new data"
    )
    predict_parser.add_argument("--csv", required=True, help="Path to data CSV file (features only)")
    predict_parser.add_argument("--model", default=None, help=f"Path to trained model (default: {MODEL_FILE})")

    return parser

def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command == "train":
        workflow_train(args)
    elif args.command == "predict":
        workflow_predict(args)
    else:
        parser.print_help()
        print("\n[INFO] Use 'slp train --help' or 'slp predict --help' for workflow details")