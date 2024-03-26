## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/03084f05-2b35-4d9c-b151-56de31474b23)
# Ordinal Encoding
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/77a68e08-0940-41f5-b84f-863c9512764e)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/d98b36d9-5cd2-420d-b98d-5a2fa00daf18)
# Label Encoder
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/13247a01-8641-43b2-b8d0-8f4444a8c808)
# OneHot Encoder
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/b67a533c-846c-4d56-9612-67a865863b3c)
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/611db96c-21f1-42cd-bcea-5c46099c2001)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/6c239398-4386-472c-ba5b-601d421aeb0c)
# Binary Encoder
```
pip install --upgrade category_encoders
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/1dfa98ed-08c3-4964-8412-bcceb56ff852)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/5df2524e-6ad7-474e-b509-f826344a52eb)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/9d0a6fab-0ef6-40a6-a795-4adfd86770a4)
# Target Encoder
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/6888b12c-f4eb-41a1-a48f-423f690baa55)
# Data Transformation
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/content/Data_to_Transform.csv')
df
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/0b2c97e6-0839-4eba-9cd9-1f4953ad3516)
```
df.skew()
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/34ad8a84-d333-48cb-a9b6-83241f80d217)
```
np.reciprocal(df["Moderate Positive Skew"])

```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/88392f52-e663-4954-b051-ffb4d7133097)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/8b3b3927-0c63-43de-b0a1-d7a3a78e8001)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/0807861f-3b10-40a1-9c66-a7330c2faafc)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/6da33fde-9db1-4fd0-b529-5ad3cc812d97)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/5eda2471-9938-42af-b666-b9a2d204507f)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/f7aeb4fe-79b2-4355-8e6f-896686763a3c)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/d7996436-e074-4629-96f8-3b5ab6620a1a)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/d8c16946-f785-4112-9cd6-544f51afba16)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/ff0e4073-c402-4947-b279-83797f248d12)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/75234bd6-4523-4e48-a2d8-872806ccab3b)
```

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/8d385d37-5321-4add-a416-e8dd504a4520)
```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/9d1668ec-6a60-44d7-8fd4-3119a4ec1ff7)
```

dt=pd.read_csv("/content/titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/f1dd0aee-b9b9-4672-ae04-95d3dbdce8d0)
```
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![image](https://github.com/SJananisenthilkumar/EXNO-3-DS/assets/144871139/a838c634-689a-4c63-bd0c-3282d86b86a0)

# RESULT:
         Finally, perform Feature Encoding and Transformation process is executed successfully.

       
