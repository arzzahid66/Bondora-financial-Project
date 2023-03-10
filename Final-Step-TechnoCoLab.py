#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
import pickle

sns.set()
warnings.filterwarnings('ignore')

from sklearn.feature_selection import mutual_info_regression, SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor, RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, scale, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_percentage_error, mean_squared_error, roc_auc_score, log_loss, precision_recall_fscore_support, mean_absolute_error, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from imblearn.over_sampling import RandomOverSampler, SMOTE
from eli5.sklearn import PermutationImportance
from pprint import pprint

from xgboost import XGBRegressor, XGBClassifier


# In[2]:


data=pd.read_csv('Bondora_raw.csv')


# In[3]:


data.head().T


# In[4]:


data.info()


# In[5]:


drop_cols=['ReportAsOfEOD',
 'LoanId',
 'LoanNumber',
 'ListedOnUTC','DateOfBirth',
 'BiddingStartedOn','UserName', 'LanguageCode','LoanApplicationStartedDate','FirstPaymentDate','IncomeFromPrincipalEmployer', 'IncomeFromPension',
       'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare',
       'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther','LoanApplicationStartedDate','ApplicationSignedHour',
       'ApplicationSignedWeekday','ActiveScheduleFirstPaymentReached','ModelVersion','WorseLateCategory','PlannedPrincipalTillDate',"NextPaymentNr",'ProbabilityOfDefault',
          'ExpectedLoss', 'LossGivenDefault', 'ExpectedReturn']
print(len(drop_cols))
data.drop(drop_cols,axis=1,inplace=True)


# In[6]:


data.head()


# In[7]:


data.isnull().sum()


# In[8]:


nan=data.isnull().sum()[data.isnull().sum()>0]
print(nan/len(data))
nan=nan/len(data)
drop_cols=list(nan[nan>0.5].index)
print(len(drop_cols))


# In[9]:


data.columns 


# In[10]:


date_cols=["LoanDate","MaturityDate_Original","MaturityDate_Last",'LastPaymentOn']
data.drop(date_cols,axis=1,inplace=True)


# In[11]:


status_current = data.loc[data["Status"]=="Current"]
status_without_current = data.drop(data.index[data['Status']=='Current'],inplace=True)


# In[12]:


data["DefaultLoan"]=np.nan
defaultNAN=data.loc[data["DefaultDate"].isnull()]
defaultPRESENT=data.DefaultDate.drop(defaultNAN.index)
data["DefaultLoan"][defaultNAN.index]=0
data["DefaultLoan"][defaultPRESENT.index]=1


# In[13]:


data['DefaultLoan'].value_counts()


# In[14]:


data


# In[15]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


# In[16]:


data['Education']= le.fit_transform(data['Education'])
data['EmploymentStatus']= le.fit_transform(data['EmploymentStatus'])
data['Gender']= le.fit_transform(data['Gender'])
data['HomeOwnershipType']= le.fit_transform(data['HomeOwnershipType'])
data['MaritalStatus']= le.fit_transform(data['MaritalStatus'])
data['VerificationType']= le.fit_transform(data['VerificationType'])
data['UseOfLoan']= le.fit_transform(data['UseOfLoan'])
data['OccupationArea']= le.fit_transform(data['OccupationArea'])
data['Country']= le.fit_transform(data['Country'])
data['NewCreditCustomer']= le.fit_transform(data['NewCreditCustomer'])
data['County']= le.fit_transform(data['County'])
data['City']= le.fit_transform(data['City'])
data['EmploymentDurationCurrentEmployer']= le.fit_transform(data['EmploymentDurationCurrentEmployer'])
data['RecoveryStage']= le.fit_transform(data['RecoveryStage'])
data['Rating']= le.fit_transform(data['Rating'])
data['DefaultLoan']= le.fit_transform(data['DefaultLoan'])
data['Restructured']= le.fit_transform(data['Restructured'])
data['CreditScoreEsMicroL']= le.fit_transform(data['CreditScoreEsMicroL'])


# In[17]:


# data.loc[data['NewCreditCustomer'] == False,'NewCreditCustomer'] = 'Existing_credit_customer'
# data.loc[data['NewCreditCustomer'] == True,'NewCreditCustomer'] = 'New_credit_Customer'

# data.loc[data['Restructured'] == False,'Restructured']='No'
# data.loc[data['Restructured'] == True,'Restructured']='Yes'

# data.loc[data['RecoveryStage']==1,'RecoveryStage']='Collection'
# data.loc[data['RecoveryStage']==0,'RecoveryStage']='Recovery'


# In[17]:


numerical_cols=[col for col in data.columns if data[col].dtype!=object]
categorical_cols=[col for col in data.columns if data[col].dtype==object]
print("No of Numerical features =",len(numerical_cols))
print("No of Categorical features =",len(categorical_cols))


# In[18]:


for i in categorical_cols:
    data[i] = data[i].fillna(data[i].mode()[0])


# In[19]:


for i in numerical_cols:
    data[i] = data[i].fillna(data[i].median())


# In[20]:


data.isnull().sum()


# In[21]:



def drop_outliers_zscore(dfcopy:pd.DataFrame, cols, threshold:int=3, inplace:bool=False):
    
    if inplace:
        global data
    else:
        data = dfcopy.copy()

    def drop_col(df_, col):
        
        mean, std = np.mean(df_[col]), np.std(df_[col])
        if std==0:
            std = 0.1
        df_['is_outlier'] = df_[col].apply(lambda x : np.abs((x - mean) / std) > threshold)
        outliers_idx = df_.loc[df_['is_outlier']].index
        df_ = df_.drop(outliers_idx, axis=0)
        
        data = df_.drop('is_outlier', axis=1)
        return data

    
    if type(cols) == str:
        data = drop_col(data, cols)
    elif type(cols) == list:
        for col in cols:
            data = drop_col(data, col)
    else :
        raise ValueError('Pass neither list nor string in {Cols}')
    
    if inplace:
        
        dfcopy = data
    else:
        return data
    
THRESHOLD = 4
num_outlier_records =  data.shape[0] - drop_outliers_zscore(data,numerical_cols , threshold=THRESHOLD).shape[0]
print(f'Number of Outliers {num_outlier_records}, with threshold, with threshold {THRESHOLD}')

drop_outliers_zscore(data, numerical_cols, threshold=THRESHOLD, inplace=True)


# In[22]:


get_ipython().run_cell_magic('time', '', "start='\\033[1m'\nend= '\\033[0;0m'")


# In[23]:


print(start+"Shape of the Dataset:"+end,data.shape,"\n")


# In[24]:


data=data[data['Age']>10]
data=data[data['AppliedAmount']>0]
data=data[data['DebtToIncome']<100]
data=data[data['CreditScoreEeMini']!=0]
data=data[data['PrincipalBalance']>=0]


# In[25]:


data.Status.unique()


# In[26]:


data_drop=['ContractEndDate','County','City','NrOfDependants', 'EmploymentPosition','WorkExperience','MonthlyPaymentDay','PlannedInterestTillDate','CurrentDebtDaysPrimary', 'DebtOccuredOn',
       'CurrentDebtDaysSecondary', 'DebtOccuredOnForSecondary','PrincipalOverdueBySchedule', 'PlannedPrincipalPostDefault',
       'PlannedInterestPostDefault', 'EAD1', 'EAD2', 'PrincipalRecovery',
       'InterestRecovery', 'RecoveryStage', 'StageActiveSince','EL_V0', 'Rating_V0', 'EL_V1', 'Rating_V1', 'Rating_V2', 'Status','Restructured', 'ActiveLateCategory','CreditScoreEsEquifaxRisk', 'CreditScoreFiAsiakasTietoRiskGrade',
       'CreditScoreEeMini','PrincipalWriteOffs',
       'InterestAndPenaltyWriteOffs', 'PrincipalBalance',
       'InterestAndPenaltyBalance', 'NoOfPreviousLoansBeforeLoan',
       'AmountOfPreviousLoansBeforeLoan', 'PreviousRepaymentsBeforeLoan','GracePeriodStart',
       'GracePeriodEnd', 'NextPaymentDate', 'NrOfScheduledPayments',
       'ReScheduledOn', 'PrincipalDebtServicingCost',
       'InterestAndPenaltyDebtServicingCost', 'ActiveLateLastPaymentCategory','DefaultDate']
print(len(data_drop))
data.drop(data_drop,axis=1,inplace=True)


# In[27]:


data.head(2)


# In[28]:


data.info()


# In[29]:


#Preferred loan amount(AppliedAmount,Amount,)
# Preferred emi(LoanDuration,MonthlyPayment)
# Preferred Roi(Interest)
#Preferred loan amount=data["Amount"]


# In[30]:


data_temp =data[['LoanDuration', 'Interest', 'Amount']]
data_temp.head()


# In[31]:


def cal_EMI(P, r, n):
  P = P.values
  r = r.values
  n = n.values
  #print(P.shape[0])
  result_1 = np.empty(0)
  result_2 = np.empty(0)
  result = np.empty(0)
  for i in range(P.shape[0]):
    #print(P[i])
    #print(r[i])
    #print(n[i])
    # EMI = P × r × (1 + r) ^ n / ((1 + r) ^ n – 1)
    #print(P[i] * (1 + r[i]))
    result_1 = np.append(result_1, P[i] * r[i] * np.power((1 + r[i]),n[i]))
    result_2 = np.append(result_2, np.power((1 + r[i]),n[i]) - 1)
    result = np.append(result, (result_1[i] / result_2[i]))

  return result


# In[32]:


data_temp['EMI'] = cal_EMI(data_temp['Amount'],data_temp['Interest'],data_temp['LoanDuration'])


# In[33]:


data['EMI'] = data_temp['EMI']


# In[34]:


data['EMI'].head()


# In[35]:


data_temp = data[['AppliedAmount', 'Interest', 'IncomeTotal', 'LiabilitiesTotal', 'LoanDuration']]
data_temp.info()


# In[36]:


data_temp[data_temp['IncomeTotal']==3665].shape


# In[37]:


# Step 1
data_temp['Ava_Inc'] = ((data_temp['IncomeTotal']-data_temp['LiabilitiesTotal'])*0.3)
data_temp['Total_Loan_Amnt'] = np.round((data['AppliedAmount'] + (data['AppliedAmount'] * data['Interest']) /100)*data['LoanDuration'])
data_temp.head()


# In[38]:


# Step 2
def eligible_loan_amnt(df):
  Ava_Inc = df['Ava_Inc'].values
  Total_Loan_Amnt = df['Total_Loan_Amnt'].values
  ELA = np.empty(0)
  for i in range(len(Ava_Inc)):
    if Total_Loan_Amnt[i] <= Ava_Inc[i]:
      ELA = np.append(ELA, Total_Loan_Amnt[i])
    else:
      ELA = np.append(ELA, Ava_Inc[i])
  return ELA


# In[39]:


data_temp['ELA'] = eligible_loan_amnt(data_temp)


# In[40]:


data_temp.head()


# In[41]:


data['ELA'] = data_temp['ELA']
data.columns


# In[42]:


data_temp = data[['Amount', 'Interest']]
data_temp.head()


# In[43]:


data_temp['InterestAmount'] = (data_temp['Amount']*(data_temp['Interest']/100))
data_temp['TotalAmount'] = (data_temp['InterestAmount'] + data_temp['Amount'])
data_temp['ROI'] = (data_temp['InterestAmount'] / data_temp['TotalAmount'])*100
data['ROI'] = data_temp['ROI']


# In[44]:


data_temp.head()


# In[45]:


data


# In[46]:


# Let's compute IQR for each numerical feature
df_IQR = data[data.select_dtypes([float, int]).columns].quantile(.75) - data[data.select_dtypes([float, int]).columns].quantile(.25)


# In[47]:


# Let's compute maximum and minimum limits
df_Max =  data[data.select_dtypes([float, int]).columns].quantile(.75) + (1.5*df_IQR)
df_Min =  data[data.select_dtypes([float, int]).columns].quantile(.25) - (1.5*df_IQR)


# In[48]:


data.select_dtypes([float, int]).columns


# In[49]:


col_IQR = data['Age'].quantile(.75) - data['Age'].quantile(.25)
col_Max = data['Age'].quantile(.75) + (1.5*col_IQR)


# In[50]:


# Loop for replacing outliers above upper bound with the upper bound value:
for column in data.select_dtypes([float, int]).columns :
   
    col_IQR = data[column].quantile(.75) - data[column].quantile(.25)
    col_Max = data[column].quantile(.75) + (1.5*col_IQR)
    data[column][data[column] > col_Max] =  col_Max


# In[51]:


# Loop for replacing outliers under lower bound with the lower bound value:
for column in data.select_dtypes([float, int]).columns :
    col_IQR = data[column].quantile(.75) - data[column].quantile(.25)
    col_Min = data[column].quantile(.25) - (1.5*col_IQR)
    data[column][data[column] < col_Min] =  col_Min


# ## Pipeline 

# In[52]:


X=data.iloc[0:,:-3]


# In[53]:


X.head(2)


# In[54]:


y=data.iloc[:,-3:]
y


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[56]:


over = RandomOverSampler(random_state=0)
smote = SMOTE()

stdscaler = StandardScaler()
scaler = MinMaxScaler()

pca = PCA(n_components=2)


# In[ ]:


LinearDiscriminantAnalysis
QuadraticDiscriminantAnalysis


# In[58]:


#LG,LOG-G,Rand_F,Ridge


# In[59]:


#Logistic Regression and Random Forest Classifier Regression Models (1. linear Reggresion 2. Ridge Regression L2 Norm).


# In[57]:


Lin_G=LinearRegression()
ridge= Ridge(alpha=0.1)
Rand_F = RandomForestRegressor(random_state=0)

stdscaler= StandardScaler


# In[67]:


steps = [('stdscaler', StandardScaler()),('pca', PCA()),('Lin_G', LinearRegression()), ('ridge', Ridge()),("Rand_F",RandomForestRegressor())]
#clf = Pipeline(["S"])


# In[99]:


model_rand_F=[('stdscaler', StandardScaler()),('pca', PCA()),("Rand_F",RandomForestRegressor())]
pipe=Pipeline(st)
pipe


# In[100]:


pipe.fit(X_train,y_train)


# In[92]:


pipe.predict(X_test)


# In[94]:


#pipe.score(X_train,y_train)
print('test accuracy = ', round(pipe.score(X_train,y_train)*100,2), "%")


# In[72]:


from sklearn import set_config
set_config(display="diagram")
steps


# In[ ]:





# In[101]:


model_lin_R=[('stdscaler', StandardScaler()),('pca', PCA()),("LIN",LinearRegression())]
pip=Pipeline(sa)
pip


# In[102]:


pip.fit(X_train,y_train)


# In[103]:


pip.predict(X_test)


# In[104]:


#pip.score(X_train,y_train)
print('test accuracy = ', round(pip.score(X_train,y_train)*100,2), "%")


# In[105]:


model_ridge=[('stdscaler', StandardScaler()),('pca', PCA()),("ridge",Ridge())]
pipe1=Pipeline(sb)

pipe1


# In[106]:


pipe1.fit(X_train,y_train)


# In[108]:


pipe1.predict(X_test)


# In[109]:


print('test accuracy = ', round(pipe1.score(X_train,y_train)*100,2), "%")


# In[118]:


import joblib 
joblib.dump(model_rand_F,"Random_Regresser")
joblib.dump(model_ridge,"ridge_Regresser")
joblib.dump(model_lin_R,"Linear_Regresser")

