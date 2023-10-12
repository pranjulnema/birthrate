from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


nRowsRead = 1000 
df1 = pd.read_csv('nchs-birth-rates-for-females-by-age-group-united-states.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'nchs-birth-rates-for-females-by-age-group-united-states.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

#For plotting Histogram

def plotHistogram(df, nHistogramShown, nHistogramPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nHistRow = int((nCol + nHistogramPerRow - 1) / nHistogramPerRow)
    plt.figure(num=None, figsize=(int(4*nHistogramPerRow), int(5*nHistRow)), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nHistogramShown)):
        plt.subplot(nHistRow, nHistogramPerRow, i+1)
        df.iloc[:,i].hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.show()


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna(axis=1)
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

# Assuming you have numeric columns in your dataset, you can select them for correlation analysis
numeric_columns = df1.select_dtypes(include=['number'])

# Compute the correlation matrix
correlation_matrix = numeric_columns.corr()

# Create a heatmap for the correlation matrix
df1.nunique()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

plt.figure(figsize=(12, 4))
plt.bar(df1['Age Group'], df1['Birth Rate'])
plt.xlabel('Age Group')
plt.ylabel('Birth Rate')
plt.title('Birth Rates by Age Group')
plt.xticks(rotation=30)
plt.show()
plotHistogram(df1, 10, 5)
plotScatterMatrix(df1, 6, 15)
sns.boxplot(data=df1)
plt.show()
















