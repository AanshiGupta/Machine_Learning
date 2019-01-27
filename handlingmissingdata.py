import pandas as pd
import numpy as np
import statistics as s


dataset = pd.read_csv(r"C:\Users\SIDDHANT\Desktop\ML\dis_data.csv")
dataset.isnull().sum()


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values

#using dropping the null fields
dataset['Salary'] = dataset['Salary'].dropna()
dataset['Salary'].isnull().sum()
a = s.stdev(dataset['Salary'])



#######
import pandas as pd
import numpy as np


dataset = pd.read_csv(r'C:\Users\SIDDHANT\Desktop\Book1.csv')
dataset.isnull()


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values

#using the mean strategy
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'nan', strategy = 'mean', axis = 0)
dataset['Salary'] = imputer.fit_transform(dataset['Salary'])

s.stdev(dataset['Salary'])

######
import pandas as pd
import numpy as np


dataset = pd.read_csv(r'C:\Users\SIDDHANT\Desktop\Book1.csv')
dataset.isnull()


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values


#using the median strategy
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
y[:,1] = imputer.fit_transform(y[:,1])

s.stdev(dataset['Salary'])

#######
import pandas as pd
import numpy as np


dataset = pd.read_csv(r'C:\Users\SIDDHANT\Desktop\Book1.csv')
dataset.isnull()


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values



#using the most frequent strategy
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
y[:,1] = imputer.fit_transform(y[:,1])

s.stdev(dataset['Salary'])

#######import pandas as pd
import numpy as np


dataset = pd.read_csv(r'C:\Users\SIDDHANT\Desktop\Book1.csv')
dataset.isnull()


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values


#using the global constant
dataset.fillna(0)

s.stdev(dataset['Salary'])



