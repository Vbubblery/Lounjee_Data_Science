from bson.objectid import ObjectId
import numpy as np
import pandas as pd
from pymongo import MongoClient
from keras.models import model_from_json
client = MongoClient("mongodb://178.128.161.146:27017/")
db = client['lounjee']
users = db['users']

def loadModel():
    model_architecture = 'lounjee_rs_architecture.json'
    model_weights = 'lounjee_rs_weights.h5'
    model = model_from_json(open(model_architecture).read())
    model.load_weights(model_weights)
    return model

def load_Data():
    result = []
    for item in users.find():
        location = None
        #waysToMeet = None
        experience = None
        education = None
        group = None
        skill = None
        industry = None
        interest = None
        #purpose = None
        offer = None
        lookin4 = None
        if 'location' in item and 'name' in item['location'] and item['location']['name']!=None:
            location = item['location']['name']
        if 'purposes' in item and item['purposes']['canOffer']!=None:
            offer = ','.join([x['name'] for x in item['purposes']['canOffer']])
            lookin4 = ','.join([x['name'] for x in item['purposes']['lookingFor']])
        if 'industries' in item and item['industries']!=None:
            industry = ','.join([str(x['code']) for x in item['industries']])
        if 'interests' in item and item['interests']!=None:
            interest = ','.join(np.unique([x['name'] for x in item['interests']]))
        if 'skills' in item and item['skills']!=None:
            skill = ','.join(np.unique([x['name'] for x in item['skills']]))
        if 'education' in item and item['education']!=None:
            education = [x['fieldOfStudy'] for x in item['education']]
            education = ','.join([x['name'] for x in education])
        if 'experience' in item and item['experience']!=None:
            experience = ','.join(np.unique([x['title'] for x in item['experience']]))
        result.append({"uid":str(item['_id']),"location":location,"offer":offer,"lookin4":lookin4,"industry":industry,"interest":interest,"skill":skill,"education":education,"experience":experience})
    return pd.DataFrame(data=result, index=np.arange(len(result)))
def predict(data,model,user_id):
    item = users.find_one({'_id':ObjectId(user_id)})
    location = None
    experience = None
    education = None
    group = None
    skill = None
    industry = None
    interest = None
    offer = None
    lookin4 = None
    if 'location' in item and 'name' in item['location'] and item['location']['name']!=None:
        location = item['location']['name']
    if 'purposes' in item and item['purposes']['canOffer']!=None:
        offer = ','.join([x['name'] for x in item['purposes']['canOffer']])
        lookin4 = ','.join([x['name'] for x in item['purposes']['lookingFor']])
    if 'industries' in item and item['industries']!=None:
        industry = ','.join([str(x['code']) for x in item['industries']])
    if 'interests' in item and item['interests']!=None:
        interest = ','.join(np.unique([x['name'] for x in item['interests']]))
    if 'skills' in item and item['skills']!=None:
        skill = ','.join(np.unique([x['name'] for x in item['skills']]))
    if 'education' in item and item['education']!=None:
        education = [x['fieldOfStudy'] for x in item['education']]
        education = ','.join([x['name'] for x in education])
    if 'experience' in item and item['experience']!=None:
        experience = ','.join(np.unique([x['title'] for x in item['experience']]))
    result=[]
    result = {"uid":item['_id'],"location":location,"offer":offer,"lookin4":lookin4,"industry":industry,"interest":interest,"skill":skill,"education":education,"experience":experience}
    data = data.append(result,ignore_index=True)

    ## One hot encoding
    def dum(df,name):
        dummies = df[name].str.get_dummies(sep=',').add_prefix(name+'_')
        df.drop([name],axis=1,inplace=True)
        df = df.join(dummies)
        return df
    arr = list(data)
    for val in arr:
        if val == 'uid':
            continue
        data = dum(data,val)

    uid = data['uid']
    data.drop(['uid'],axis=1,inplace=True)
    X_a = data.tail(1)
    X_a = X_a.append([data.tail(1)]*(data.shape[0]-2),ignore_index=True)
    X_b = data.iloc[0:data.shape[0]-1]
    X = [X_a,X_b]
    predy = model.predict(X)
    df = pd.DataFrame({'0':predy[:,0],'1':predy[:,1]})
    df['uid'] = uid
    df['uid'] = df['uid']
    return df.sort_values(by=['1'],ascending=False)

if __name__ == '__main__':
    model = loadModel()
    data = load_Data()
    id = '59f70c842706222d006f29c0'
    p = predict(data,model,user_id=id).round(6).head(500)
    result = p.to_json(orient='values')
    print(result)
