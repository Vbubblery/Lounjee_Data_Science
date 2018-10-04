import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder # Encode labels

from keras.models import model_from_json # Load model from local
from keras.layers import Dense, Input, Embedding, Reshape, Dropout, LeakyReLU, Flatten # Keras layer which used for building Deep Neural Networks
from keras.layers.merge import concatenate # Merge two layers
from keras.models import Model
from keras.optimizers import Adam # Optimizer which will used fro optimizer the training.
from keras.utils import np_utils

from keras import backend as Backend
from keras.callbacks import EarlyStopping
Backend.tensorflow_backend._get_available_gpus()
import tensorflow as k
config = k.ConfigProto()
config.gpu_options.allow_growth = True

from pymongo import MongoClient # Communication between Python and MongoDB
from bson.objectid import ObjectId # ObjectId object for MongoDB

# Connction to the server which run the mongodb service
client = MongoClient("mongodb://178.128.161.146:27017/")
db = client['lounjee']
# Get the collections which will use for the project
matches = db['matches']
users = db['users']

# Load the score from the match collection.
# This function will return a DataFrame which incude the usera userb and rating
#
# i.e
# rating | usera | userb
#  0 or 1    id      id
#
# usera the user who do the recommendation
# userb the user who are recommended or not to usera
# rating [0,1] 0 means do not recommend, 1 means support
def load_Score():
    # Generate a matrix of user, the number is the id of ther user, and the default value is 0
    #      0  |  1  |  2  |  3  |  4
    #  0   0     0     0     0     0
    #  1   0     0     0     0     0
    #  2   0     0     0     0     0
    #  3   0     0     0     0     0
    #  4   0     0     0     0     0
    result = []
    for item in users.find():
        result.append(str(item['_id']))
    df = pd.DataFrame(data=np.zeros((len(result),len(result))), index=result,dtype=np.int8, columns=result)
    
    # Define a function to find the user-user in the matrix and then reset the value of it.
    def find_item(df,usera,userb,value=0):
        df[userb][usera] = value
        return df[userb][usera]
    
    # Reset the value of the user-user matrix if we find the state of the connection in the match between these 2 users are one of the [accepted,accepting,postponed,inviting]
    for item in matches.find():
        try:
            if item['stateA']['type'] == 'accepted':
                find_item(df,str(item['userA']),str(item['userB']),1)
            elif item['stateA']['type'] == 'accepting':
                find_item(df,str(item['userA']),str(item['userB']),1)
            elif item['stateA']['type'] == 'postponed':
                find_item(df,str(item['userA']),str(item['userB']),1)
            elif item['stateA']['type'] == 'inviting':
                find_item(df,str(item['userA']),str(item['userB']),1)
    #         elif item['stateA']['type'] == 'reporting':
    #             find_item(df,str(item['userA']),str(item['userB']),-1)
        except Exception:
            pass
    # If the usera like the userb, also mark this connection to be 1.
    for item in users.find():
        try:
            if item['favorites']:
                for f in item['favorites']:
                    find_item(df,str(item['_id']),str(f['_id']),1)
        except Exception:
            pass
    # split the matrix to a DataFrame with
    # rating | usera | userb
    df = df[(df.T != 0).any()]
    result=[]
    for i in df.index.values:
        for idx,val in enumerate(df.loc[i]):
            result.append({'usera':i,'userb':(df.loc[i].index)[idx],'rating':val})

    return pd.DataFrame(result)

# Load the profile of users from the user collection
# This function will return a DataFrame

# i.e
# uid | skills | education | experience  |  industry .....

# uid is the id of user's
# the each column's value come from the user's profile and some of them are vector and some of them are single value
# the vector is sep by ','
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

# Treat the Data
# ratings: come from the result of the function load_Score
# data: come from the result of the function load_Data
# sampling: 1 and 2
## 1 means use random under sampling
## 2 means use random over sampling
def data_Engineering(ratings,data, sampling=1):
    ## One hot encoding which will flat the value of the feature of the data.
    def dum(df,name):
        dummies = df[name].str.get_dummies(sep=',').add_prefix(name+'_')
        df.drop([name],axis=1,inplace=True)
        dummies
        df = df.join(dummies)
        return df
    arr = list(data)
    for val in arr:
        if val == 'uid':
            continue
        data = dum(data,val)
    # Make the data balanced
    ## Random under sampling
    if sampling is 1:
        count_class_0, count_class_1 = ratings.rating.value_counts()
        df_class_0 = ratings[ratings['rating'] == 0]
        df_class_1 = ratings[ratings['rating'] == 1]
        df_class_0_under = df_class_0.sample(count_class_1)
        ratings = pd.concat([df_class_0_under, df_class_1], axis=0)
    ## Random over sampling
    elif sampling is 2:
        count_class_0, count_class_1 = ratings.rating.value_counts()
        df_class_0 = ratings[ratings['rating'] == 0]
        df_class_1 = ratings[ratings['rating'] == 1]
        df_class_1_over = df_class_1.sample(count_class_0, replace=True)
        ratings = pd.concat([df_class_0, df_class_1_over], axis=0)
    # The number of the data changed, need to reset the index of DataFrame.
    ratings.reset_index(inplace=True,drop=True)
    
    # Merge these two DataFrame (data,ratings)
    # first replace the usera with his id
    # secnod replace the userb with his id
    result = pd.merge(ratings, data, left_on='usera',right_on='uid')
    result.drop(['usera','uid'], axis=1, inplace=True)
    result = pd.merge(result, data, left_on='userb',right_on='uid')
    result.drop(['uid','userb'], axis=1, inplace=True)
    
    
    # Clean-up the useless data
    del data
#     del ratings
    import gc
    gc.collect()
    return result

def train(data,length=10000):
    # DNN model
    def EmbeddingNet_classification(n1_features,n2_features,n_users,n_latent_factors_user = 8,n_latent_factors_item = 8,k=200, alpha=0.15, dropout=0.2,lr=0.005):
        # Embedding usera
        model1_in = Input(shape=(n1_features,),name='useraInput')
        model1_out = Embedding(input_dim = n_users+1, output_dim = n_latent_factors_user)(model1_in)
        model1_out = Flatten()(model1_out)
        model1_out = Dropout(dropout)(model1_out)
        # Embedding userb
        model2_in = Input(shape=(n2_features,),name='userbInput')
        model2_out = Embedding(input_dim = n_users+1, output_dim = n_latent_factors_item)(model2_in)
        model2_out = Flatten()(model2_out)
        model2_out = Dropout(dropout)(model2_out)
        # Merge embedding of usera and embeddingof userb
        model = concatenate([model1_out, model2_out],axis=-1)
        model = LeakyReLU(alpha=alpha)(model)
        model = Dropout(dropout)(model)
        
        # Deep Learning
        model = Dense(k)(model)
        model = LeakyReLU(alpha=alpha)(model)
        model = Dropout(dropout)(model)
        model = Dense(int(k/2))(model)
        model = LeakyReLU(alpha=alpha)(model)
        model = Dropout(dropout)(model)
        model = Dense(int(k/4))(model)
        model = LeakyReLU(alpha=alpha)(model)
        model = Dropout(dropout)(model)
        model = Dense(int(k/10))(model)
        model = LeakyReLU(alpha=alpha)(model)
        model = Dense(2, activation='softmax')(model)
        
        model = Model([model1_in, model2_in], model)
        adam = Adam(lr=lr)
        model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
        return model
    i = int((data.shape[1]+1)/2)
    # Get the usera's feature from data
    usera = data.iloc[:,1:i]
    # Get the userb's feature from data
    userb = data.iloc[:,i:]
    # Number of users
    n = length
    X = [usera,userb]
    y = data.rating
    
    # Same as One Hot Encoding, but this is use for label
    # i.e.
    # label's value are 1 and 0
    #   target                          0           1
    #   0                    ->         1           0   
    #   1                               0           1
    encoder = LabelEncoder()
    encoder.fit(y)
    y = np_utils.to_categorical(encoder.transform(y), 2)
    
    # If the score do not improve after 4 times, stop the training.
    callback = [EarlyStopping(patience=4,monitor='loss')]
    
    model = EmbeddingNet_classification(n1_features=usera.shape[1],n2_features=userb.shape[1],n_users=n)
    history = model.fit(X,y,batch_size=1000,epochs=200,shuffle=True,verbose=1,callbacks = callback)
    return (model,history,X,y)

def saveModel(model):
    model_json = model.to_json()
    open('lounjee_rs_architecture.json', 'w').write(model_json)
    model.save_weights('lounjee_rs_weights.h5', overwrite=True)
def loadModel():
    model_architecture = 'lounjee_rs_architecture.json'
    model_weights = 'lounjee_rs_weights.h5'
    model = model_from_json(open(model_architecture).read())
    model.load_weights(model_weights)
    return model
    
def main():
    ratings = load_Score()
    data = load_Data()
    result = data_Engineering(ratings,data,sampling=1)
    (model,history,X,y) = train(result)
    #model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    # model.evaluate(X,y,verbose=1)
    saveModel(model)
if __name__ == "__main__":
    main()
