================================================================================
  SINGLE LAYER PERCEPTRON NEURAL NETWORK CLI
================================================================================

A command-line tool for training and using a Single Layer Perceptron with
support for multiple activation functions and train/test evaluation.

Compatible with dot matrix printers and ASCII-only terminals.

================================================================================
  INSTALLATION
================================================================================

1. Extract the project to your system
2. Navigate to the project directory:
   cd /path/to/NeuralNetwork

3. The executable 'slp' is located in the dist/ directory

For development installation:
   pip install -e .
   # Then use: slp [command] [options]

================================================================================
  QUICK START
================================================================================

--- TRAINING A MODEL ---

Basic training with verbose output (shows all epochs):
   ./dist/slp train --csv data.csv --target y

Training with custom activation function:
   ./dist/slp train --csv data.csv --target y --activation sigmoid --epochs 20

Training with evaluation (train/test split):
   ./dist/slp train --csv data.csv --target y --evaluate --test-split 30

--- MAKING PREDICTIONS ---

Predict using the default trained model:
   ./dist/slp predict --csv features.csv

Predict using a custom model:
   ./dist/slp predict --csv features.csv --model my_model.pkl

================================================================================
  DETAILED USAGE
================================================================================

--- WORKFLOW 1: TRAINING ---

Command: slp train [options]

Options:
  --csv CSV_FILE               [REQUIRED] Path to training CSV file
  --target COLUMN_NAME         [REQUIRED] Name of target column
  --activation FUNC            Activation function
                               Options: binary, bipolar, sigmoid, threshold
                               Default: binary
  --threshold VALUE            Activation threshold for binary/bipolar
                               Default: 0
  --lr LEARNING_RATE           Learning rate (eta)
                               Default: 1
  --epochs NUM                 Maximum number of training epochs
                               Default: 10
  --model MODEL_PATH           Output path for trained model
                               Default: ./assets/model.pkl
  --evaluate                   Enable evaluation mode (see below)
  --test-split PERCENTAGE      Test set percentage for evaluation
                               Default: 30

TRAINING MODES:

1. Standard Training (Verbose)
   ./dist/slp train --csv data.csv --target y --epochs 10
   
   - Shows all training epochs with detailed iteration tables
   - Trains on full dataset
   - Best for understanding how the perceptron learns
   
2. Evaluation Training (Automatic Split)
   ./dist/slp train --csv data.csv --target y --evaluate --test-split 25
   
   - Automatically splits data: 75% training, 25% test
   - Suppresses intermediate training output
   - Shows classification report on test set only
   - Best for quick model assessment
   - Default split: 70% training, 30% test

INPUT FILE FORMAT (CSV):

Must include all features and the target column in one CSV:

   Example: data.csv
   -----------
   x1,x2,x3,target
   0,0,1,0
   0,1,0,0
   1,0,1,1
   1,1,1,1
   
   Command:
   ./dist/slp train --csv data.csv --target target

OUTPUT:

The trained model is saved to: ./assets/model.pkl
(or custom path if --model is specified)

Example output from training:

   [INFO] Loading data from data.csv...
   
   [CONFIG] Training configuration:
      * Target column: y
      * Activation function: binary
      * Learning rate: 1.0
      * Threshold: 0
      * Max epochs: 3
   
   [TRAIN] Starting training with full training output...
   
   Inferred number of features: 2
   Inferred target mode: binary
   Selected activation function: binary
   EPOCH 1/3
     x1    x2    Net input    Predicted    Δw1    Δw2    Δbias
   ----  ----  -----------  -----------  -----  -----  -------
      0     0            0            0      0      0        0
      0     1            0            0      0      0        0
      1     0            0            0      0      0        0
      1     1            0            0      1      1        1
   ================================================================================
   EPOCH 2/3
   ...
   
   [SUCCESS] Model trained and saved to ./assets/model.pkl

Example output from evaluation:

   [INFO] Loading data from data.csv...
   
   [CONFIG] Training configuration:
      * Target column: y
      * Train/Test split: 75%/25%
   
   [DATA] Data split: 3 training samples, 1 test samples
   [TRAIN] Training on training set (intermediate output suppressed)...
   
   [REPORT] Classification Report (Test Set - 1 samples):
   ======================================================================
                      precision    recall  f1-score   support
                  0       1.00      1.00      1.00         1
                  1       0.00      0.00      0.00         0
        accuracy                           1.00         1
       macro avg       0.50      0.50      0.50         1
    weighted avg       1.00      1.00      1.00         1
   ======================================================================
   
   [SUCCESS] Model trained and saved to ./assets/model.pkl

ACTIVATION FUNCTIONS:

1. binary (default)
   - Output: 0 or 1
   - Uses threshold to determine output
   - Best for: Binary classification

2. bipolar
   - Output: -1 or 1
   - Inferred automatically if target has negative values
   - Best for: Problems with -1 and +1 labels

3. sigmoid
   - Output: continuous value between 0 and 1
   - Smooth, differentiable function
   - Best for: Probabilistic interpretation

4. threshold
   - Output: 0 or 1 (or -1 if bipolar mode)
   - Hard threshold at specified value
   - Best for: Strict decision boundaries

================================================================================
  WORKFLOW 2: PREDICTION
================================================================================

Command: slp predict [options]

Options:
  --csv CSV_FILE              [REQUIRED] Path to feature CSV file (no target)
  --model MODEL_PATH          Path to trained model
                              Default: ./assets/model.pkl

IMPORTANT: Prediction CSV must contain ONLY features, no target column!

INPUT FILE FORMAT (CSV):

Example: test_features.csv (features only, no target column)
   x1,x2
   0,0
   0,1
   1,0
   1,1

OUTPUT:

The predictions are printed to console:

   [INFO] Loading model from ./assets/model.pkl...
   [INFO] Loading data from test_features.csv...
   [PREDICT] Making predictions for 4 samples...
   
   Predictions:
   --------------------------------------------------
     Sample    0:      0
     Sample    1:      1
     Sample    2:      0
     Sample    3:      1
   --------------------------------------------------
   [SUCCESS] Total predictions made: 4

================================================================================
  COMPLETE EXAMPLES
================================================================================

--- EXAMPLE 1: Train on AND Logic ---

1. Create training data file (and_data.csv):
   x1,x2,y
   0,0,0
   0,1,0
   1,0,0
   1,1,1

2. Train the model:
   ./dist/slp train --csv and_data.csv --target y --activation binary

3. Create prediction data (and_test.csv):
   x1,x2
   0,0
   0,1
   1,0
   1,1

4. Make predictions:
   ./dist/slp predict --csv and_test.csv

Expected output: 0, 0, 0, 1

--- EXAMPLE 2: Train with Evaluation Report ---

1. Create training data with multiple samples:
   ./dist/slp train --csv dataset.csv --target label --evaluate --test-split 20

This will:
   - Split data: 80% training, 20% testing
   - Train silently on 80% of data
   - Show classification report on 20% test set
   - Save model to ./assets/model.pkl

--- EXAMPLE 3: Use Different Activation Function ---

Train with sigmoid activation:
   ./dist/slp train --csv data.csv --target y --activation sigmoid --epochs 50

Train with bipolar activation:
   ./dist/slp train --csv data.csv --target y --activation bipolar

Train with threshold function:
   ./dist/slp train --csv data.csv --target y --activation threshold --threshold 0.5

================================================================================
  TROUBLESHOOTING
================================================================================

Problem: "CSV file not found"
Solution: Use absolute path or make sure file is in current directory
   ./dist/slp train --csv /full/path/to/data.csv --target y

Problem: "Model file not found"
Solution: Train a model first before predicting
   ./dist/slp train --csv data.csv --target y

Problem: Prediction has wrong number of features
Solution: Ensure prediction CSV has same number of columns as training
   Training: had 2 features (x1, x2)
   Prediction CSV must also have exactly 2 columns

Problem: Shape mismatch in predictions
Solution: Prediction CSV should NOT contain target column
   Wrong: x1,x2,y (remove y column)
   Right: x1,x2

Problem: Evaluation report shows all zeros
Solution: With small datasets (4-5 samples), train/test split is too small
   Use larger dataset or don't use --evaluate flag

================================================================================
  PROJECT STRUCTURE
================================================================================

NeuralNetwork/
|-- cli.py                          Main entry point
|-- singleLayerPerceptron.py        Core perceptron implementation
|-- Activation/
|   |-- __init__.py
|   +-- activation.py               Activation functions
|-- controllers/
|   |-- __init__.py
|   +-- cli_controller.py           CLI logic and workflows
|-- utils/
|   |-- __init__.py
|   +-- data_utils.py               Data loading utilities
|-- dist/
|   +-- slp                         Standalone executable
+-- assets/
    +-- model.pkl                   Saved trained model

================================================================================
  TECHNICAL NOTES
================================================================================

- The tool uses sklearn's train_test_split with random_state=42 for
  reproducible results in evaluation mode.

- Models are saved using Python's pickle format for quick serialization.

- The perceptron uses the learning rule:
  w_new = w_old + eta * (target - predicted) * features

- Bias update: bias_new = bias_old + eta * (target - predicted)

- The tool automatically infers whether to use binary or bipolar mode based
  on the target column values (checks for negative values).

- All output is ASCII-compatible for compatibility with dot matrix printers
  and legacy terminals.

================================================================================
  SUPPORT
================================================================================

For issues or questions, check:
1. Ensure CSV format is correct (headers in first row)
2. Verify target column name matches CSV header
3. Check file permissions on input CSV files
4. Ensure training CSV has both features and target column
5. Ensure prediction CSV has only features (no target)

================================================================================
  LICENSE
================================================================================

See LICENSE file for details.

================================================================================
