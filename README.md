# 🏠 Housing Data Analysis and Machine Learning
- This project involves two main tasks: classification and regression. We explored and analyzed two different datasets, visualized the data, prepared it for modeling, and trained machine learning models to solve classification and regression problems.

## Objectives 🎯
- Classification: Predict the grade of houses based on various features. 🏷️
- Regression: Predict the median house value based on demographic and geographic features. 🏡

## Functionality ⚙️
- Data Exploration: Perform statistical and descriptive analysis to understand the data. 🔍
- Data Visualization: Create visualizations to identify patterns and relationships in the data. 🎨
- Data Preparation: Clean and preprocess the data for modeling. 🧹
- Model Training: Train machine learning models for classification and regression tasks. 🏋️‍♂️
- Model Evaluation: Evaluate the performance of the models using various metrics. 📊

## Tools Used 🛠️
- Python: Programming language used for data analysis and modeling. 🐍
- Pandas: Library for data manipulation and analysis. 🐼
- Matplotlib & Seaborn: Libraries for data visualization. 📈
- Scikit-learn: Library for machine learning. 🤖

## Development Process 🚀
- Exploratory Data Analysis (EDA) 🔍
  - Dataset Description 📋
   - The dataset contains information about houses, including variables such as total area, number of rooms, number of bedrooms, number of washrooms, roof type, roof area, lawn area, number of floors, API, ANB, expected price, and grade.
  - Statistical and Descriptive Analysis 📊
   - We performed statistical and descriptive analysis to better understand the distribution and characteristics of the data.
- Data Visualization 🎨
  - Distribution of Expected Price 💰
   - We created a histogram to visualize the distribution of the expected prices of the houses.
  - Relationship between Total Area and Expected Price 📏
   - A scatter plot was used to visualize the relationship between the total area of the houses and their expected prices.
  - Boxplot of Expected Price by Grade 🏷️
   - A boxplot was created to compare the expected price of houses across different grades.
  - Number of Houses by Grade 🏡
   - A count plot was created to show the number of houses in each grade category.
  - Relationship between Number of Bedrooms and Expected Price 🛏️
   - A boxplot was used to visualize the relationship between the number of bedrooms and the expected price.
  - Relationship between Number of Washrooms and Expected Price 🚿
   - A boxplot was created to show the relationship between the number of washrooms and the expected price.
  - Relationship between Number of Floors and Expected Price 🏢
   - A boxplot was used to visualize the relationship between the number of floors and the expected price.
  - Relationship between Roof Area and Expected Price 🏠
   - A scatter plot was created to show the relationship between the roof area and the expected price.
  - Relationship between Lawn Area and Expected Price 🌳
   - A scatter plot was used to visualize the relationship between lawn area and expected price.
- Data Preparation 🛠️
  - Data Cleaning 🧹
   - We imputed missing values in the total_bedrooms column with the median.
   - The categorical variable ocean_proximity was converted into dummy variables.
  - Data Normalization 📏
   - Numerical variables were normalized to ensure they have a similar scale.
- Modeling 🤖
  - Data Splitting ✂️
   - The data was split into training and testing sets.
  - Model Training and Evaluation 🏋️‍♂️
   - We trained and evaluated two models for regression: Linear Regression and Random Forest.
   - We trained and evaluated three models for classification: Random Forest Classifier, Support Vector Machine (SVM), and Logistic Regression.
- Model Evaluation 🧮
  - Regression Model Metrics 📏
   - We calculated evaluation metrics: MSE, MAE, RMSE, R2, and EVS for each regression model.
  - Classification Model Metrics 📊
   - We calculated evaluation metrics: accuracy, precision, recall, and F1-score for each classification model.
  - Comparison of Metrics 📊
   - We visualized the evaluation metrics to compare the performance of the models using bar charts.
  - Feature Importance 🌟
   - We used the Random Forest model to obtain the feature importance and visualized it in a bar chart.
  - Predictions vs. Actual Values 🔍
   - Scatter plots and residual plots were created to compare the predictions with the actual values for each model.

## Results 📈
- Regression Models
  - Linear Regression and Random Forest were used to predict the median house value.
  - Metrics such as MSE, MAE, RMSE, R2, and EVS were calculated to evaluate the models.
- Classification Models
  - Random Forest Classifier, Support Vector Machine (SVM), and Logistic Regression were used to predict the grade of houses.
  - Metrics such as accuracy, precision, recall, and F1-score were calculated to evaluate the models.
    
## Conclusions 📝
- The project successfully demonstrated the process of data exploration, visualization, preparation, and modeling for both classification and regression problems. The Support Vector Machine and Logistic Regression models performed best for classification, while the Random Forest model provided valuable insights into feature importance for regression.

## 📊 Visualizations
- Histograms: Distribution of expected prices and house values.
- Scatter Plots: Relationships between various features and target variables.
- Boxplots: Comparisons of target variables across different categories.
- Heatmaps: Correlation matrices to identify relationships between features.
- Bar Charts: Comparison of model performance metrics and feature importance.

## 📈 Models and Metrics
- Regression Models: Linear Regression, Random Forest
- Metrics: MSE, MAE, RMSE, R2, EVS
- Classification Models: Random Forest Classifier, SVM, Logistic Regression
- Metrics: Accuracy, Precision, Recall, F1-Score

## 📂 Project Structure
- Datos
- Notebook

## 📬 Contact
- For any questions or suggestions, feel free to reach out via GitHub issues or contact me directly at jotaduranbon.com.

Happy coding! 😊
