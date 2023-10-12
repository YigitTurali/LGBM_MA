# LGBM_MA Repository

This repository contains the implementation of the LGBM_MA model. Below is a detailed guide on how to use, contribute, and understand the codebase.

## Paper Reference:

**Title:** [Gradient Boosting With Moving-Average Terms for Nonlinear Sequential Regression](https://ieeexplore.ieee.org/abstract/document/10233101/)

**Abstract:** 
The paper investigates sequential nonlinear regression and introduces a novel gradient boosting algorithm. This algorithm is inspired by the well-known linear auto-regressive-moving-average (ARMA) models and exploits the residuals, i.e., prediction errors, as additional features. The main idea is to utilize the state information from early time steps contained in the residuals to enhance the performance in a nonlinear sequential regression/prediction framework. By exploiting the changes in the previous time steps through residual terms, the algorithm aims to achieve improved predictive accuracy in the context of boosting.

## Table of Contents

- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Contributing](#contributing)
- [License](#license)

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YigitTurali/LGBM_MA.git

2. **Navigate to the Directory**:
    ```bash
    cd LGBM_MA
3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
4. **Run the Code**
Detailed instructions on specific scripts or models will be provided below in the Code Explanation section.
  
## Code Explanation
### 1. `Hybrid_Model.py`
This file contains the implementation of the LGBM-MA model, which combines GBM and MA terms to achieve better predictive performance.

### 2. `MLP_DataLoaders.py`
This script is responsible for loading the data required for the MLP (Multi-Layer Perceptron) model. It ensures that the data is in the correct format and is ready for training and evaluation.

### 3. `MLP_Model.py`
In this file, the MLP model is defined. It includes the architecture, training, and evaluation methods for the model.

### 4. `Pipelines.py`
This script contains various data preprocessing pipelines. These pipelines are essential for preparing the data for different models and ensuring that it's in the right format.

### 5. `Single_LightGBM.py`
As the name suggests, this file contains the implementation of the LightGBM model. It includes the model's definition, training, and evaluation methods.

### 6. `Synthetic_Dataset_Prep.py`
This script is used to prepare synthetic datasets. It contains various utilities and methods to generate and process synthetic data.

### 7. `letter_paper.py`
This file contains the implementation related to the letter paper dataset. It includes data loading, preprocessing, and model training for this specific dataset.

### 8. `letter_paper_real_data.py`
Similar to the `letter_paper.py` file, this script deals with the real data from the letter paper dataset.

### 9. `m4_prep.py`
This script is dedicated to the preparation of the M4 dataset. It includes utilities and methods to load, preprocess, and prepare the M4 dataset for model training.

### 10. `main.py`
This is the main script where the entire workflow is orchestrated. It calls various utilities and models defined in other files and executes the project's main logic.

### 11. `synthectic_data2.py`
Another utility script for synthetic data generation and processing.

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) to get started.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more information or questions, please contact [Yigit Turali](https://github.com/YigitTurali).
