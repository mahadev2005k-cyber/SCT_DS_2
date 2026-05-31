import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── STEP 1: Load the dataset ─────────────────────────────────────
df = pd.read_csv('train.csv')
print("=" * 50)
print("STEP 1: Dataset Loaded")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", df.columns.tolist())

# ── STEP 2: Basic Info ───────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: First 5 Rows")
print(df.head())

print("\nData Types:")
print(df.dtypes)

# ── STEP 3: Missing Values ───────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Missing Values")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

# ── STEP 4: Data Cleaning ────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: Cleaning Data...")

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing - 77%)
df.drop(columns=['Cabin'], inplace=True)

print("Age missing filled with median:", df['Age'].median())
print("Embarked missing filled with mode:", df['Embarked'].mode()[0])
print("Cabin column dropped")
print("Missing values after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("✅ No missing values remaining!" if df.isnull().sum().sum() == 0 else "")

# ── STEP 5: Basic Statistics ─────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5: Basic Statistics")
print(df.describe())

survival_rate = df['Survived'].mean() * 100
print(f"\nOverall Survival Rate: {survival_rate:.1f}%")

# ══════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════

sns.set_style("whitegrid")
colors_main = ['#e74c3c', '#2ecc71']

# ── CHART 1: Survival Count ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
survival_counts = df['Survived'].value_counts()
bars = ax.bar(['Did Not Survive', 'Survived'],
              [survival_counts[0], survival_counts[1]],
              color=['#e74c3c', '#2ecc71'], edgecolor='white', width=0.5)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 5,
            f'{int(h)}\n({h/len(df)*100:.1f}%)',
            ha='center', fontsize=11, fontweight='bold')
ax.set_title('Survival Count — Titanic', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Passengers')
ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0, 650)
plt.tight_layout()
plt.savefig('chart1_survival_count.png', dpi=150)
plt.close()
print("\n✅ Chart 1 saved: Survival Count")

# ── CHART 2: Survival by Gender ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

gender_survival = df.groupby('Sex')['Survived'].mean() * 100
axes[0].bar(gender_survival.index, gender_survival.values,
            color=['#3498db', '#e91e8c'], edgecolor='white', width=0.4)
for i, (idx, val) in enumerate(gender_survival.items()):
    axes[0].text(i, val + 1, f'{val:.1f}%', ha='center', fontweight='bold')
axes[0].set_title('Survival Rate by Gender', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Survival Rate (%)')
axes[0].set_ylim(0, 100)
axes[0].spines[['top','right']].set_visible(False)

gender_counts = df.groupby(['Sex','Survived']).size().unstack()
gender_counts.plot(kind='bar', ax=axes[1],
                   color=['#e74c3c','#2ecc71'], edgecolor='white')
axes[1].set_title('Survival Count by Gender', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Number of Passengers')
axes[1].set_xticklabels(['Female','Male'], rotation=0)
axes[1].legend(['Did Not Survive','Survived'])
axes[1].spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('chart2_survival_by_gender.png', dpi=150)
plt.close()
print("✅ Chart 2 saved: Survival by Gender")

# ── CHART 3: Survival by Passenger Class ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

pclass_survival = df.groupby('Pclass')['Survived'].mean() * 100
axes[0].bar(['1st Class','2nd Class','3rd Class'],
            pclass_survival.values,
            color=['#f1c40f','#3498db','#e74c3c'], edgecolor='white', width=0.5)
for i, val in enumerate(pclass_survival.values):
    axes[0].text(i, val + 1, f'{val:.1f}%', ha='center', fontweight='bold')
axes[0].set_title('Survival Rate by Class', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Survival Rate (%)')
axes[0].set_ylim(0, 80)
axes[0].spines[['top','right']].set_visible(False)

pclass_counts = df.groupby(['Pclass','Survived']).size().unstack()
pclass_counts.plot(kind='bar', ax=axes[1],
                   color=['#e74c3c','#2ecc71'], edgecolor='white')
axes[1].set_title('Survival Count by Class', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Number of Passengers')
axes[1].set_xticklabels(['1st','2nd','3rd'], rotation=0)
axes[1].legend(['Did Not Survive','Survived'])
axes[1].spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('chart3_survival_by_class.png', dpi=150)
plt.close()
print("✅ Chart 3 saved: Survival by Class")

# ── CHART 4: Age Distribution ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(df['Age'], bins=30, color='#3498db', edgecolor='white', alpha=0.85)
axes[0].axvline(df['Age'].mean(), color='red', linestyle='--',
                label=f"Mean: {df['Age'].mean():.1f}")
axes[0].axvline(df['Age'].median(), color='orange', linestyle='--',
                label=f"Median: {df['Age'].median():.1f}")
axes[0].set_title('Age Distribution of All Passengers', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].spines[['top','right']].set_visible(False)

survived = df[df['Survived']==1]['Age']
not_survived = df[df['Survived']==0]['Age']
axes[1].hist(not_survived, bins=25, alpha=0.7, color='#e74c3c', label='Did Not Survive')
axes[1].hist(survived, bins=25, alpha=0.7, color='#2ecc71', label='Survived')
axes[1].set_title('Age Distribution by Survival', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Count')
axes[1].legend()
axes[1].spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('chart4_age_distribution.png', dpi=150)
plt.close()
print("✅ Chart 4 saved: Age Distribution")

# ── CHART 5: Fare Distribution & Correlation Heatmap ─────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(df['Fare'], bins=40, color='#9b59b6', edgecolor='white', alpha=0.85)
axes[0].set_title('Fare Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Fare (£)')
axes[0].set_ylabel('Count')
axes[0].spines[['top','right']].set_visible(False)

corr_cols = ['Survived','Pclass','Age','SibSp','Parch','Fare']
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=axes[1], square=True, linewidths=0.5)
axes[1].set_title('Correlation Heatmap', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('chart5_fare_and_heatmap.png', dpi=150)
plt.close()
print("✅ Chart 5 saved: Fare & Heatmap")

print("\n" + "=" * 50)
print("✅ ALL DONE! 5 charts saved in your folder.")
print("=" * 50)