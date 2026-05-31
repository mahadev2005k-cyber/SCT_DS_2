# SCT_DS_2 - Titanic EDA (Exploratory Data Analysis)
**Internship:** SkillCraft Technology
**Intern ID:** SCT/MAY26/6435
**Intern Name:** K Mahadev

## Task Description
Perform data cleaning and exploratory data analysis (EDA) on the 
Titanic dataset. Explore relationships between variables and 
identify patterns and trends in the data.

## Dataset
Titanic Dataset (Kaggle)
- 891 passengers
- 12 features (Age, Sex, Pclass, Fare, Survived, etc.)

## Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn

## Data Cleaning Steps
- Filled missing Age values with median (28.0)
- Filled missing Embarked values with mode (S)
- Dropped Cabin column (77% missing values)

## Charts Generated
- chart1_survival_count.png — Overall survival count and percentage
- chart2_survival_by_gender.png — Survival rate by gender
- chart3_survival_by_class.png — Survival rate by passenger class
- chart4_age_distribution.png — Age distribution by survival
- chart5_fare_and_heatmap.png — Fare distribution and correlation heatmap

## Key Findings
- Overall survival rate: 38.4%
- Women had much higher survival rate (~74%) vs men (~19%)
- 1st class passengers survived more than 3rd class
- Most passengers were aged 20–35
