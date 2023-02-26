from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import re
import numpy as np

df = pd.read_csv("bengaluru_house_prices.csv")

# print(df.head())

# drop some columns from dataset

df.drop(['area_type', 'availability', 'bath', 'society', 'balcony'],
        axis='columns', inplace=True)
# print(df.head())

# Making the

# grouping by frequency
# fq = df.groupby('location').size()/len(df)
# # mapping values to dataframe
# df.loc[:, "{}".format('location_')] = df['location'].map(fq)
# # drop original column.
# df = df.drop(['location'], axis=1)

# fq.plot.bar(stacked=True)


# Import label encoder

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df['location'] = label_encoder.fit_transform(df['location'])
print(df.head(10))

df['size'].fillna('0', inplace=True)
unique_size = df['size'].unique()
# print(unique_size)
# print(df.head())

for i in unique_size:
    result = re.findall(r'\d+', i)
    df['size'] = df['size'].replace(i, result[0])
    # print(result)
df['size'] = df['size'].astype('int')
update_unique_size = df['size'].unique()
# print(update_unique_size)
df['size'] = df['size'].replace(0, df['size'].mean())
# print(df['size'].unique())
# pd.set_option('display.max_rows', None)
# print(df['total_sqft'].unique())

for i in (df['total_sqft'].unique()):
    result1 = re.findall(r'\d+', i)
    df['total_sqft'] = df['total_sqft'].replace(i, result1[0])
    # print(result1)
df['total_sqft'] = df['total_sqft'].astype('float')
# print(df['total_sqft'].isnull().sum())
# print(df.info())
df['location'].fillna(df['location'].mean(), inplace=True)
# print(df.info())

# df['size'].astype('int')
# df['total_sqft'].astype('int')
print(df.info())
X = df[['size', 'total_sqft', 'location']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

accuracy = r2_score(y_test, y_pred)
print(accuracy)
