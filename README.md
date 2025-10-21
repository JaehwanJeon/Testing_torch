Deep Learning Model for Hysteresis Prediction

This document outlines the file structure and usage instructions for the deep learning model designed for hysteresis prediction.


📁 File Structure
The project is organized into modular Python scripts to manage earthquake generation, data preprocessing, model training, and results analysis for various hysteretic models. The datasets used will be made available upon request. Please contact the authors for access.

.

├── BilinearTesting_main.py, BoucWenTesting_main.py, BWBNTesting_main.py, Xu_OP_4_main.py, ROTesting_main.py, Link_main.py

│   ├── Main files for testing each specific hysteretic model (e.g., Bilinear, Bouc-Wen, Menegotto-Pinto, Ramberg-Osgood, and experimental Link dataset.)

│   ├── These files drive the entire process: they import and execute the functions from EQ_generation.py, Preprocessing.py, Training.py, and ResultAnalysis.py.

│   ├── Execution flow is controlled by setting boolean flags (generate_EQ, preprocess, train, analyze_result).

│   └── Deep learning model parameters (e.g., layer sizes, learning rate) are typically defined and modifiable within these files.

│

├── EQ_generation.py

│   └── Contains the function(s) responsible for generating the **Earthquake (EQ)** input data.

├── Preprocessing.py

│   └── Contains the function(s) for **data preprocessing**, such as normalization, scaling, and sequence windowing.

├── Training.py

│   └── Contains the function(s) for **model training**, including compiling and fitting the deep learning model.

├── ResultAnalysis.py

│   └── Contains the function(s) for **analyzing and visualizing the results** (e.g., plots, metrics).

│

└── backend.py
    └── A library containing **various backend functions**, such as custom loss functions, the definition of the proposed **LSTM** or other neural network architectures, and utility functions.


🛠️ How to Use

The main execution and workflow control is managed by modifying settings within the testing files (e.g., BoucWenTesting_main.py).

 * Workflow Control: In the selected main testing file, set the boolean flags—generate_EQ, preprocess, train, and analyze_result—to True or False to control which parts of the workflow execute.
 
 * Parameter Modification: Deep learning model parameters can be modified directly within the main testing file, specifically between the call to the preprocess function and the train function.
 
   * Examples of modifiable parameters:
   
     * nn_size = 64 (Neural network layer size)

     * alpha = 0.2 (A model-specific parameter, e.g., for regularization or a non-linear activation)

     * window_size = 1000 (The size of the input sequence/window for the time-series prediction)
