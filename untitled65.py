

import pandas as pd
 
 
df= pd.read_csv('Sales Data.csv')
 
df=df.drop(columns=['Order ID', 'Purchase Address'])

df=pd.get_dummies(df, drop_first=True)
X= df.drop(columns=['Sales'])
Y= df['Sales'] 
  from sklearn.model_selection import train_test_split
 
X_train, X_test, Y_train, Y_test= \
    train_test_split(X,Y, test_size=0.3, random_state=1234)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100, random_state=1234)
rfc.fit(X_train, Y_train)
Y_predict= rfc.predict(X_test)



from sklearn.metrics import classification_report, accuracy_score

accuracy= accuracy_score(Y_test, Y_predict)
cr=classification_report(Y_test, Y_pred)