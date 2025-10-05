#!/usr/bin/env python3
"""
Entry point for the CLI.
"""
from controllers.cli_controller import main

if __name__ == "__main__":
    main()
import sys
import os
import pickle
import pandas as pd
import numpy as np
from typing import Optional
import argparse
from singleLayerPerceptron import SingleLayerPerceptron
from tabulate import tabulate

MODEL_FILE = "model.pkl"


def load_model() -> Optional[SingleLayerPerceptron]:
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            return pickle.load(f)
    return None


def save_model(model: SingleLayerPerceptron):
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)


def train(args):
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' not found.")
        return
    df = pd.read_csv(args.csv)
    model = SingleLayerPerceptron(df, args.target, threshold=args.threshold, learning_rate=args.lr,
                                  max_epochs=args.epochs)
    model.fit()
    save_model(model)
    print(f"Model trained and saved to {MODEL_FILE}")


def predict(args):
    model = load_model()
    if not model:
        print(f"Error: Model file '{MODEL_FILE}' not found. Train first.")
        return
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' not found.")
        return
    df = pd.read_csv(args.csv)
    X = df.to_numpy()
    preds = model.predict(X)
    for i, p in enumerate(preds):
        print(f"Row {i}: {p}")


def report(args):
    model = load_model()
    if not model:
        print(f"Error: Model file '{MODEL_FILE}' not found. Train first.")
        return
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' not found.")
        return
    df = pd.read_csv(args.csv)
    X = df[[c for c in df.columns if c != args.target]].to_numpy()
    y = df[args.target].to_numpy()
    report_str = model.classification_report(X, y)
    print("Classification Report:")
    print(report_str)


def main():
    parser = argparse.ArgumentParser(description="SingleLayerPerceptron CLI", prog="slp")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--csv", required=True, help="Path to training CSV")
    train_parser.add_argument("--target", required=True, help="Target column name")
    train_parser.add_argument("--threshold", type=float, default=0, help="Activation threshold")
    train_parser.add_argument("--lr", type=float, default=1, help="Learning rate")
    train_parser.add_argument("--epochs", type=int, default=10, help="Max epochs")

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--csv", required=True, help="Path to prediction CSV (features only)")

    # Report subcommand
    report_parser = subparsers.add_parser("report", help="Generate classification report")
    report_parser.add_argument("--csv", required=True, help="Path to test CSV")
    report_parser.add_argument("--target", required=True, help="Target column name")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "report":
        report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
