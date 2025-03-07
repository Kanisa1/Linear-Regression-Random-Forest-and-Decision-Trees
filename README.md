# Optimization Using Gradient Descent: Linear Regression

This project demonstrates how to build a simple linear regression model to predict sales based on TV marketing expenses. The project explores three different approaches to solving this problem:

1. **Linear Regression using NumPy**.
2. **Linear Regression using Scikit-Learn**.
3. **Linear Regression using Gradient Descent from scratch**.

Additionally, the project compares the performance of Linear Regression with two other machine learning models: **Random Forest** and **Decision Trees**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Approaches](#approaches)
4. [Setup](#setup)
5. [Running the Code](#running-the-code)
6. [Results](#results)
7. [Dependencies](#dependencies)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

---

## Project Overview

The goal of this project is to predict sales based on TV marketing expenses using linear regression. The project explores three different approaches to linear regression:

1. **NumPy**: Using `np.polyfit` to fit a linear regression model.
2. **Scikit-Learn**: Using the `LinearRegression` class from Scikit-Learn.
3. **Gradient Descent**: Implementing gradient descent from scratch to optimize the linear regression model.

Finally, the project compares the performance of Linear Regression with Random Forest and Decision Trees using Root Mean Squared Error (RMSE) as the evaluation metric.

---

## Dataset

The dataset used in this project is `tvmarketing.csv`, which contains two fields:

- **TV**: TV marketing expenses (independent variable).
- **Sales**: Sales amount (dependent variable).

The dataset is loaded using `pandas`, and the relationship between TV expenses and sales is visualized using a scatter plot.

---

## Approaches

### 1. Linear Regression with NumPy
- The `np.polyfit` function is used to fit a linear regression model.
- Predictions are made using the equation \( Y = mX + b \).

### 2. Linear Regression with Scikit-Learn
- The `LinearRegression` class from Scikit-Learn is used to fit the model.
- Predictions are made using the `predict` method.

### 3. Gradient Descent
- Gradient descent is implemented from scratch to optimize the linear regression model.
- The cost function and its partial derivatives are defined.
- The model parameters (slope and intercept) are updated iteratively.

### 4. Comparison with Random Forest and Decision Trees
- Random Forest and Decision Tree models are created using Scikit-Learn.
- The RMSE for each model is calculated and compared.
- The models are ranked based on their RMSE values.

---

## Setup

### Prerequisites
- Python 3.x
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

3. Download the dataset (`tvmarketing.csv`) and place it in the `data` folder.

---

## Running the Code

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Summative_Assignment.ipynb
   ```

2. Run the cells in the notebook sequentially to:
   - Load the dataset.
   - Perform linear regression using NumPy and Scikit-Learn.
   - Implement gradient descent from scratch.
   - Compare Linear Regression with Random Forest and Decision Trees.

3. The output will include:
   - Predictions for TV marketing expenses.
   - RMSE values for each model.
   - A ranked list of models based on RMSE.

---

## Results

### Model Performance
The models are ranked based on their RMSE values:

1. **Random Forest**: Lowest RMSE (best performance).
2. **Linear Regression**: Moderate RMSE.
3. **Decision Tree**: Highest RMSE (worst performance).

### Example Output
```
Root Mean Square Error (Linear Regression): 3.1234
Root Mean Square Error (Random Forest): 2.9876
Root Mean Square Error (Decision Tree): 3.4567

Model Rank and Associated RMSE:
Random Forest: 2.9876
Linear Regression: 3.1234
Decision Tree: 3.4567
```

---

## Dependencies

- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.
- **Scikit-Learn**: For machine learning models and evaluation.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset: [tvmarketing.csv](data/tvmarketing.csv)
- Scikit-Learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- NumPy Documentation: [https://numpy.org/doc/](https://numpy.org/doc/)
