import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
from sklearn.impute import KNNImputer
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Get New York Airbnb data
df = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")

# Print the first rows of the dataframe for visualization
df.head()

df.shape

df.head().T

df.info()

# Start Exploratory data analysis (EDA)

# Count the number of unique values in total_df Dataframe
df.nunique()

# Eliminate irrelevant data
df.drop(['id','name','host_name', 'last_review'],axis=1, inplace=True)
df.shape

df.head().T

df.info()

# Count the number of unique values in total_df Dataframe
df.nunique()

# Check if duplicates
print(f"Duplicate values: {df.duplicated().sum():.2f}")

# Description of the numeric variables
df.describe().T

# Histogram and boxplot of the price variable
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Histogram in the first subplot
ax[0].hist(df['price'], bins=20, color='skyblue', edgecolor='black')
ax[0].set_title('Histogram of Price')
ax[0].set_xlabel('Price')
ax[0].set_ylabel('Frequency')

# Boxplot in the second subplot
ax[1].boxplot(df['price'], vert=False)
ax[1].set_title('Boxplot of Price')
ax[1].set_xlabel('Price')

# Show the graph
plt.tight_layout()
plt.show()

# Add a column to df with the log price
df['log_price'] = np.log(df['price'])

print(f"Maximum log price: {df['log_price'].max():.2f}")

# Histogram and boxplot of the log price
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Histogram in the first subplot
ax[0].hist(df['log_price'], bins=20, range=[0, 10], color='skyblue', edgecolor='black')
ax[0].set_title('Histogram of Price (log)')
ax[0].set_xlabel('Price')
ax[0].set_ylabel('Frequency')

# Boxplot in the second subplot
ax[1].boxplot(df['log_price'], vert=False)
ax[1].set_title('Boxplot of Price (log)')
ax[1].set_xlabel('Price')

# Show the graph
plt.tight_layout()
plt.show()

# Check the lines with at least one isnull value (%)
print(f"Lines with at least one isnull value in df: {df.isnull().any(axis=1).mean()*100:.2f} %")

# Visualize missing values in dataframe 
#msno.matrix(df)
#msno.bar(df) 

# Create a new dataframe changing NaN to zero in reviews_per_month column (none review per month)
df2=df
df2['reviews_per_month'].fillna(0, inplace=True)
#msno.bar(df2) 

# Check isnull values in df2 (%)
#print("isnull values in df2 %\n")
#print(df2.isnull().mean()*100)

#df2.info()
#print(df2['neighbourhood_group'].unique())
# 5 values: ['Brooklyn' 'Manhattan' 'Queens' 'Staten Island' 'Bronx']

# Change 'neighbourhood_group' column to numeric
# Initialize the label encoder
label_encoder = LabelEncoder()
# Apply label encoding to the column
df2['neighbourhood_group'] = label_encoder.fit_transform(df2['neighbourhood_group'])

#print(df2['neighbourhood'].unique())
# 221 values: ['Kensington' 'Midtown' 'Harlem' 'Clinton Hill' 'East Harlem' ... 'Willowbrook']
# Change 'neighbourhood' column to numeric
# Calculate the frequency of each category
frequency_encoding = df2['neighbourhood'].value_counts().to_dict()
# Map the frequencies to the original DataFrame
df2['neighbourhood'] = df2['neighbourhood'].map(frequency_encoding)

#print(df2['room_type'].unique())
# 3 values: ['Private room' 'Entire home/apt' 'Shared room']
# Change 'room_type' column to numeric
df2["room_type"] = df["room_type"].replace(
       { "Private room": 1, "Entire home/apt": 2, "Shared room": 3 })

#df2.info()

# Calculate influence
# Split x and y variables
X = df2.drop(['price','log_price'], axis=1)
y = df2['log_price']

# OLS regression model
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Calculate the Cook distance
cooks_distance = model.get_influence().cooks_distance

# Scatter plot of the Cook distance
#plt.scatter(df2.price, cooks_distance[0], s=100)
#plt.xlabel('x')
#plt.ylabel('Cook Distance')
#plt.show()

# Cook distance threshold
t_point = 4/len(df2)

# Remove any houses that exceed the 4/n threshold
print(len(np.where(cooks_distance[0]>t_point)[0]))
print(np.where(cooks_distance[0]>0.5))
df3 = df2.drop(np.where(cooks_distance[0]> t_point)[0])

