
import pandas as pd
from datetime import datetime, timedelta

X1= pd.read_csv('../data/S_1.csv',header=0 )
X1.columns=['number_buy','user_id']
Y1 = pd.read_csv('../data/S_2.csv',header=0 )
Y1.columns=['pred_date_day','user_id']


Y = pd.merge(X1, Y1, how='left', on=['user_id'])


Y['number_buy'] = Y['number_buy'].astype(float)
Y['pred_date_day'] = Y['pred_date_day'].astype(float)

Y=Y.sort_values('number_buy',ascending=False,)



Y["pred_date"] =Y['pred_date_day'].apply(lambda x: datetime(2017, 5,1)+timedelta(days=( (int(x)) )))


test = pd.DataFrame()
# 排序后截取前50000个user

test['user_id'] = Y['user_id'][:50000]
test['pred_date'] = Y['pred_date'][:50000]

test.to_csv('../submit/predict.csv',index=False,header=True)

print("complete")