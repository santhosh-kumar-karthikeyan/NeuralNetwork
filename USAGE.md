================================================================================
  QUICK USAGE GUIDE - Single Layer Perceptron CLI
================================================================================

This document provides quick examples for the two main workflows.

================================================================================
  WORKFLOW 1: TRAINING
================================================================================

--- Step 1: Prepare Training Data ---

Create a CSV file with features and target column.
Example file: my_data.csv

   x1,x2,target
   0,0,0
   0,1,0
   1,0,0
   1,1,1

--- Step 2a: Train with Full Output (See All Epochs) ---

Command:
   ./dist/slp train --csv my_data.csv --target target --epochs 5

This will:
   - Show configuration
   - Print all 5 epochs with detailed tables
   - Save model to ./assets/model.pkl

--- Step 2b: Train with Evaluation Report ---

Command:
   ./dist/slp train --csv my_data.csv --target target --evaluate

This will:
   - Split data: 70% train, 30% test
   - Train silently
   - Show classification report on test set
   - Save model to ./assets/model.pkl

--- Step 2c: Train with Custom Parameters ---

Command:
   ./dist/slp train \
     --csv my_data.csv \
     --target target \
     --activation sigmoid \
     --lr 0.5 \
     --epochs 20 \
     --model my_model.pkl

This will:
   - Use sigmoid activation function
   - Learning rate 0.5
   - Train for 20 epochs
   - Save to my_model.pkl

================================================================================
  WORKFLOW 2: PREDICTION
================================================================================

--- Step 1: Prepare Feature Data ---

Create a CSV file with ONLY features (no target column).
Example file: test_data.csv

   x1,x2
   0,0
   0,1
   1,0
   1,1

--- Step 2a: Make Predictions (Default Model) ---

Command:
   ./dist/slp predict --csv test_data.csv

This will:
   - Load model from ./assets/model.pkl
   - Make predictions for each row
   - Print predictions to console

Output:
   [INFO] Loading model from ./assets/model.pkl...
   [INFO] Loading data from test_data.csv...
   [PREDICT] Making predictions for 4 samples...
   
   Predictions:
   --------------------------------------------------
     Sample    0:      0
     Sample    1:      1
     Sample    2:      0
     Sample    3:      1
   --------------------------------------------------
   [SUCCESS] Total predictions made: 4

--- Step 2b: Make Predictions (Custom Model) ---

Command:
   ./dist/slp predict --csv test_data.csv --model my_model.pkl

This will:
   - Load model from my_model.pkl
   - Make predictions for each row

================================================================================
  COMPLETE EXAMPLE: AND Gate
================================================================================

AND gate learns: output is 1 only when both inputs are 1

1. Create training data (and_training.csv):

   x1,x2,y
   0,0,0
   0,1,0
   1,0,0
   1,1,1

2. Train the model:

   ./dist/slp train --csv and_training.csv --target y --epochs 10

   Output shows:
   [INFO] Loading data from and_training.csv...
   [CONFIG] Training configuration:
      * Target column: y
      * Activation function: binary
      * Learning rate: 1
      * Threshold: 0
      * Max epochs: 10
   
   [TRAIN] Starting training with full training output...
   
   Inferred number of features: 2
   Inferred target mode: binary
   Selected activation function: binary
   EPOCH 1/10
     x1    x2    Net input    Predicted    Δw1    Δw2    Δbias    w1    w2    bias
   ----  ----  -----------  -----------  -----  -----  -------  ----  ----  ------
      0     0            0            0      0      0        0     0     0       0
      0     1            0            0      0      0        0     0     0       0
      1     0            0            0      0      0        0     0     0       0
      1     1            0            0      1      1        1     1     1       1
   ================================================================================
   
   [More epochs follow...]
   
   [SUCCESS] Model trained and saved to ./assets/model.pkl

3. Create test data (and_test.csv):

   x1,x2
   0,0
   0,1
   1,0
   1,1

4. Make predictions:

   ./dist/slp predict --csv and_test.csv

   Output:
   [INFO] Loading model from ./assets/model.pkl...
   [INFO] Loading data from and_test.csv...
   [PREDICT] Making predictions for 4 samples...
   
   Predictions:
   --------------------------------------------------
     Sample    0:      0
     Sample    1:      0
     Sample    2:      0
     Sample    3:      1
   --------------------------------------------------
   [SUCCESS] Total predictions made: 4

5. Verify: Expected output for AND gate is [0, 0, 0, 1] ✓

================================================================================
  ACTIVATION FUNCTIONS
================================================================================

Use --activation flag to choose:

1. binary (default for 0/1 targets)
   ./dist/slp train --csv data.csv --target y --activation binary

2. bipolar (default for -1/1 targets)
   ./dist/slp train --csv data.csv --target y --activation bipolar

3. sigmoid (smooth, differentiable)
   ./dist/slp train --csv data.csv --target y --activation sigmoid

4. threshold (hard cutoff)
   ./dist/slp train --csv data.csv --target y --activation threshold

================================================================================
  PARAMETERS EXPLAINED
================================================================================

--csv FILE
   Path to CSV file (required)
   For training: must include target column
   For prediction: features only (no target)

--target COLUMN
   Name of target column in training CSV
   Example: if CSV has "y" column, use --target y

--activation FUNC
   Activation function: binary, bipolar, sigmoid, threshold
   Default: binary

--threshold VALUE
   Threshold for activation (for binary/bipolar/threshold functions)
   Default: 0

--lr VALUE
   Learning rate (eta) - controls step size during learning
   Default: 1
   Smaller values: slower learning, more stable
   Larger values: faster learning, may overshoot

--epochs NUM
   Maximum number of training iterations
   Default: 10
   More epochs may improve accuracy but takes longer

--model PATH
   Path to save/load model
   Default: ./assets/model.pkl

--evaluate
   When training, split data into train/test and show report
   No argument needed, just add the flag

--test-split PERCENT
   Test set percentage when using --evaluate
   Default: 30
   Example: --test-split 20 means 80% train, 20% test

================================================================================
  ERROR MESSAGES
================================================================================

[ERROR] CSV file not found
   Solution: Check file path, use absolute path if needed

[ERROR] Model file not found. Train a model first
   Solution: Run training first: ./dist/slp train ...

[ERROR] Shape mismatch / operands could not be broadcast
   Solution: Prediction CSV must have same number of features as training
            (and no target column)

[ERROR] Column 'y' not found
   Solution: Check CSV header matches --target value

Invalid value for '--epochs'
   Solution: Must be a number: --epochs 10

================================================================================
  TIPS AND TRICKS
================================================================================

- Test your data first with small --epochs value (3-5) to see if it works

- Use --evaluate flag for quick validation of model quality

- Save models with meaningful names: --model and_gate_model.pkl

- Keep training and test data in same format (same column names)

- For small datasets, don't use --evaluate (too little test data)

- Use learning rate --lr between 0.1 and 1.0 for best results

- Sigmoid activation is useful for non-linear problems

- Check CPU output is ASCII-compatible before printing to dot matrix

================================================================================
