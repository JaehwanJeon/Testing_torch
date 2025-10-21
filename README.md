Deep Learning Model for Hysteresis Prediction
File Structure
.
├── BilinearTesting_main.py, BoucWenTesting_main.py, BWBNTesting_main.py, IMKTesting_main.py, ROTesting_main.py
│   ├── Main files for testing each model
│   ├── Each file imports and runs EQ_generation.py, Preprocessing.py, Training.py, and ResultAnalysis.py
│   ├── Run each by setting generate_EQ, preprocess, train, and analyze_result to True
│   └── Deep learning model parameters can also be changed in these files
│
├── EQ_generation.py
│   └── Function for Earthquake (EQ) generation
├── Preprocessing.py
│   └── Function for data preprocessing
├── Training.py
│   └── Function for model training
├── ResultAnalysis.py
│   └── Function for result analysis
│
└── backend.py
    └── Contains various backend functions (e.g., loss function, proposed LSTM model, etc.)

How to Use
 * Run by setting generate_EQ, preprocess, train, and analyze_result to True or False.
 * Deep learning model parameters can be modified between the preprocess and train functions (e.g., nn_size = 64, alpha = 0.2, window_size = 1000, etc.).
