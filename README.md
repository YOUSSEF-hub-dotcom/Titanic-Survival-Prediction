
# Titanic Survival Prediction 🚢

This project uses the Titanic dataset to predict passenger survival using Machine Learning.
I performed data cleaning, feature engineering (like extracting Titles from names), and applied a RandomForestClassifier with OneHotEncoding for categorical variables.

### Key Steps:
- Data Cleaning (handling missing values, dropping high-null columns).
- Feature Engineering (`Title` extracted from passenger names).
- OneHotEncoding for categorical features (`sex`, `embarked`, `Title`).
- Model Training: RandomForestClassifier.
- Accuracy achieved: **~78%**.

### Results:
- Accuracy: ~78%
- The model shows strong precision and recall balance.
- Titles and passenger class were strong indicators of survival.
