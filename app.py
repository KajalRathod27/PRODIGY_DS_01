import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Titanic training dataset (train.csv)
df = pd.read_csv('Dataset/train.csv')

# Step 2: Check the first few rows of the dataset
print(df.head())

# Step 3: Visualize the distribution of ages (Histogram)
plt.figure(figsize=(10,6))
sns.histplot(df['Age'].dropna(), kde=True, bins=20, color='blue')
plt.title('Age Distribution in the Titanic Dataset')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Step 4: Visualize the distribution of genders (Bar Chart)
plt.figure(figsize=(8,6))
sns.countplot(x='Sex', data=df, palette='Set2')
plt.title('Gender Distribution in the Titanic Dataset')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
