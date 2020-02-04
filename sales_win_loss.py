# In this tutorial, we will use scikit-learn to build a predictive model 
# to tell us which sales campaign will result in a loss and which will 
# result in a win.

#import necessary modules
import pandas as pd

#store the url in a variable
url = "https://community.watsonanalytics.com/wp-content/uploads/2015/04/WA_Fn-UseC_-Sales-Win-Loss.csv"

# Read in the data with 'read_csv()'
sales_data = pd.read_csv(url)

# Using .head() method to view the first few records of the data set
sales_data.head()
