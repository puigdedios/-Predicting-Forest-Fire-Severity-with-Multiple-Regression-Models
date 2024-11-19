## How to Run the Code

To run the code in a Colab notebook, use the following commands:

!wget https://raw.githubusercontent.com/puigdedios/-Predicting-Forest-Fire-Severity-with-Multiple-Regression-Models/main/forest_fires_1.py !python forest_fires_1.py



## Models Used

The following regression models were trained and evaluated:
- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**
- **Decision Tree Regressor**
- **Gradient Boosting Regressor**
- **Support Vector Regression (SVR)**
- **K-Nearest Neighbors (KNN)**

These models were assessed using two performance metrics:
- **Test RMSE** (Root Mean Squared Error): Measures how well the model performs on unseen data.
- **Test R²** (R-squared): Indicates how well the model fits the data.
- **Cross-Validation RMSE**: Provides a better estimate of model performance by evaluating it across different subsets of the dataset.

## Model Performance

The performance of each model is summarized below:

| Model                 | Test RMSE      | Test R²         | Cross-Validation RMSE |
|-----------------------|----------------|-----------------|-----------------------|
| **Linear Regression**  | 108.39         | 0.003           | 2162.38               |
| **Random Forest**      | 109.78         | -0.022          | 2877.67               |
| **XGBoost**            | 111.71         | -0.059          | 4720.38               |
| **Decision Tree**      | 95.36          | 0.229           | 6903.12               |
| **Gradient Boosting**  | 107.81         | 0.014           | 4808.74               |
| **Support Vector Regression** | 110.12 | -0.029          | 2189.86               |
| **K-Nearest Neighbors**| 107.13         | 0.026           | 2480.02               |

### Model Insights:
- **Decision Tree** showed the best performance in terms of R² (0.229), but it had a significantly higher Cross-Validation RMSE compared to other models, indicating potential overfitting.
- **Linear Regression** performed poorly in terms of R² (0.003), but it showed relatively lower RMSE values compared to some other models.
- **Random Forest**, **XGBoost**, and **Gradient Boosting** showed moderate performance, with the Random Forest model performing slightly worse than the others on both test RMSE and Cross-Validation RMSE.

## Project Structure

The project includes the following key components:

- **Data Preprocessing**: 
  - Label encoding for categorical features.
  - Scaling of numerical features using `StandardScaler` to improve model performance.
  
- **Model Evaluation**: 
  - Each model was trained on the training set, then evaluated using both the test set and cross-validation.
  
- **Visualization**: 
  - Bar plots of RMSE and R² values for each model.
  - Scatter plots comparing actual vs. predicted values for each model.

## Installation

To run this project, you need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- xgboost
- ucimlrepo

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost ucimlrepo
```

## Usage

1. Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/forest-fire-prediction.git
```

2. Navigate to the project directory:

```bash
cd forest-fire-prediction
```

3. Run the Python script to train the models and evaluate their performance:

```bash
python forest_fire_prediction.py
```

This will output the RMSE and R² values for each model, followed by visualizations comparing the model performances and actual vs. predicted values.

## Results

- **RMSE**: Lower RMSE values indicate better model accuracy.
- **R²**: A higher R² indicates a better model fit, with values closer to 1 being ideal.
- **Cross-validation RMSE**: Provides a more reliable performance measure by evaluating models across different data subsets.

Bar plots for RMSE and R² comparisons are generated, along with scatter plots for each model's actual vs. predicted values.

## Contributing

Feel free to fork this repository, submit issues, or contribute to improving the models and visualizations. Contributions to enhance model performance or add new features are always welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.