import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel(r"C:\Users\youss\Downloads\titanic.xls")
pd.set_option('display.width',None)
print(df.head(51))

print("------------------------------------")
print("==========>>> Basic Function:")
print("number of rows and column:")
print(df.shape)
print("Data Type in data:")
print(df.dtypes)
print("The name of columns:")
print(df.columns)
print("The information about data:")
print(df.info())
print("Statistical operations:")
print(df.describe().round())
print("number of frequency rows:")
print(df.duplicated().sum())
print("------------------------------------")
print("============>>> Cleaning Data:")
missing_percentage = df.isnull().mean() * 100
print("The percentage of missing values in every column:\n",missing_percentage)
print("Missing Values before cleaning :")

print(df.isnull().sum())
sns.heatmap(df.isnull())
plt.title("Missing Values Before Cleaning")
plt.show()

print("The miss Value in age column is: 0.155 % so we use fillna")
df['age'] = df['age'].fillna(df['age'].mean())
print("The miss Value in embarked column is: 20 % so we use fillna")
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
print("The miss Value in fare column is: 0.076 %  so we use fillna")
df['fare'] = df['fare'].fillna(df['fare'].mean())
print("The age,embarked,fare,cabin,body,boat,and home.dest column contain miss value ?")
print(df[['fare','embarked','age']].isnull().sum())
print("The miss value in cabin column is 77% so we use drop")
df = df.drop(columns=['cabin'])
print("The miss value in boat column is 62% so we use drop")
df = df.drop(columns=['boat'])
print("The miss value in body column is 90% so we use drop")
df = df.drop(columns=['body'])
print("The miss value in home.dest column is 43% so we use drop")
df = df.drop(columns=['home.dest'])
print('The Cabin,body,boat,and home.dest contain more miss value So We use drop to remove them')
print("Missing Values after cleaning :")
print(df.isnull().sum())

sns.heatmap(df.isnull())
plt.title('The missing value in titanic after cleaning')
plt.show()

print("------------------------------------")
print("============>>> Exploratory Data Analysis (EDA):")
# 'pclass', 'survived', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket','fare', 'embarked']

print("Analyzing the distribution of passenger ages across passenger classes")
avg_age_by_pclass = df.groupby('pclass')['age'].mean().round(1)
print(avg_age_by_pclass)

print("Examining the survival rate difference between male and female passengers")
sex_sur = df.groupby('sex')['survived'].mean()
print(sex_sur)

print("Calculating the average fare by embarkation port")
avg_emp_fare = df.groupby('embarked')['fare'].mean().round(1)
print(avg_emp_fare)

print("Checking correlation between number of siblings/spouses and survival")
corr_sib_sur = df['sibsp'].corr(df['survived'])
print(corr_sib_sur)
print("No Relationship between siblings/spouses and survival")

print("Average age of people who survived")
avg_age_sur = df.groupby('survived')['age'].mean().round(1)
print(avg_age_sur)

print("Extracting titles from names and analyzing their relation to survival")
df['Title'] = df['name'].str.extract(r',\s*([^\.]*)\.', expand=False).str.strip()
print(df['Title'])
df['Title'] = df['Title'].replace({
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs',
    'Lady': 'Rare',
    'Countess': 'Rare',
    'Capt': 'Rare',
    'Col': 'Rare',
    'Don': 'Rare',
    'Dr': 'Rare',
    'Major': 'Rare',
    'Rev': 'Rare',
    'Sir': 'Rare',
    'Jonkheer': 'Rare',
    'Dona': 'Rare'
})
print("🔹 Frequency of Titles:")
print(df['Title'].value_counts())
print("\n")

survival_by_title = df.groupby('Title')['survived'].mean().sort_values(ascending=False)
print("🔹 Survival Rate by Title:")
print(survival_by_title)

print("Investigating the impact of passenger class on average fare")
avg_pclass_fare = df.groupby('pclass')['fare'].mean().round(1)
print(avg_pclass_fare)

print("Analyzing the relationship between number of parents/children and fare")
corr_parch_fare = df['parch'].corr(df['fare'])
print(corr_parch_fare)

print("Analyzing survival rate by age groups")
def age_group(age):
    if pd.isnull(age):
        return 'Unknown'
    elif age < 18:
        return 'Child (<18)'
    elif age <= 60:
        return 'Adult (18-60)'
    else:
        return 'Elderly (>60)'

df['AgeGroup'] = df['age'].apply(age_group)

print(df[['age', 'AgeGroup']].head(15))
survival_by_agegroup = df.groupby('AgeGroup')['survived'].mean().sort_values(ascending=False)
print("🔹 Survival Rate by Age Group:")
print(survival_by_agegroup)

print("------------------------------------")
print(" Visualization :")

survival_by_title = df.groupby(['Title', 'pclass'])['survived'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x='Title', y='survived', hue='pclass', data=survival_by_title, palette="viridis")
plt.xticks(rotation=90)
plt.title("Survival Rate by Title and Passenger Class", fontsize=14, fontweight='bold')
plt.ylabel("Survival Rate")
plt.xlabel("Title")
plt.ylim(0, 1)
plt.legend(title="Passenger Class")
plt.show()


survival_by_agegroup = df.groupby(['AgeGroup', 'pclass'])['survived'].mean().reset_index()
plt.figure(figsize=(7, 5))
sns.barplot(x='AgeGroup', y='survived', hue='pclass', data=survival_by_agegroup, palette="viridis")
plt.title("Survival Rate by Age Group and Passenger Class", fontsize=14, fontweight='bold')
plt.ylabel("Survival Rate")
plt.xlabel("Age Group")
plt.ylim(0, 1)
plt.legend(title="Passenger Class")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='sex', y='survived', hue='pclass', data=df, palette="viridis")
plt.title("Survival Rate by Sex and Passenger Class", fontsize=14, fontweight='bold')
plt.ylabel("Survival Rate")
plt.xlabel("Sex")
plt.ylim(0, 1)
plt.legend(title="Passenger Class")
plt.show()

df['family_size'] = df['sibsp'] + df['parch'] + 1
plt.figure(figsize=(8, 5))
sns.barplot(x='family_size', y='survived', hue='pclass', data=df, palette="viridis")
plt.title("Survival Rate by Family Size and Passenger Class", fontsize=14, fontweight='bold')
plt.ylabel("Survival Rate")
plt.xlabel("Family Size")
plt.ylim(0, 1)
plt.legend(title="Passenger Class")
plt.show()

print("------------------Machine Learning-------------")

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'Title']

X = df[features].copy()
y = df['survived']

categorical_cols = ['sex', 'embarked', 'Title']
numeric_cols = ['pclass', 'age', 'sibsp', 'parch', 'fare']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred)) # 77%
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\\n", classification_report(y_test, y_pred))