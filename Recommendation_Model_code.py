#This function assumes the dataset has
# Product Id and User Id as column names
#Please rename the columns if not the case

def Neural_embed_Reco_train(path,israted=False):
  #Read the data to train the model
  data=pd.read_csv(path)

  #Divide the dataset based on the Time at which 
  #Product is bought. We tend to use all except 
  #Last  month to train the model
  
  last_month=data['Month'].value_counts().sort_index(ascending=False).index[0]
  
  #We now encode the Users and Items using LabelEnocder
  from sklearn.preprocessing import LabelEncoder
  lab_item=LabelEncoder()
  data['Item_Id']=lab_item.fit_transform(data['Product Id'])

  lab_user=LabelEncoder()
  data['User_Id']=lab_user.fit_transform(data['User Id'])
 
  #We delete the unwanted cols 
  del data['User Id'],data['Product Id']
  ratings=data[['Item_Id','User_Id']]

  #We find the number of Unique users and Items 
  n_users=max(ratings['User_Id'])+1
  n_items=max(ratings['Item_Id'])+1

  #We now create a ratings matrix with corresponding User and Item values
  if(israted==False):
    ratings_matrix=np.zeros((n_users,n_items))
    for i in range(len(ratings)):
      ratings_matrix[ratings['User_Id'][i]][ratings['Item_Id'][i]]=1

    estimator=[]
    for i in range(len(data)):
      estimator.append(ratings_matrix[data['User_Id'][i]][data['Item_Id'][i]])

    data['estimator']=estimator    

  #If the dataset is rated we use that as our measure
  if(israted==True):
    ratings_matrix=np.zeros((n_users,n_items))
    for i in range(len(data)):
      ratings_matrix[data['User_Id'][i]][data['Item_Id'][i]]=data['rating'][i]
    
    #Min max normalize the rating to bring it to [0,1] range
    for i in range(len(ratings_matrix)):
      ratings_matrix[i]=(ratings_matrix[i]-min(ratings_matrix[i]))/(max(ratings_matrix[i])-min(ratings_matrix[i])) 
    
    estimator=[]
    for i in range(len(data)):
      estimator.append(ratings_matrix[data['User_Id'][i]][data['Item_Id'][i]])

    data['estimator']=estimator   

  ratings_df=pd.DataFrame(ratings_matrix)

  #Now we split the data into test and train sets
  data1=data[data['Month']>=last_month]

  usr_dict1={}
  for i in range(n_users):
    usr_dict1[i]=0

  rem=[]
  for i in (data1.index):  
    if(usr_dict1[data1['User_Id'][i]] < 3):
      usr_dict1[data1['User_Id'][i]]+=1
    else:  
      rem.append(i)
  
  #Dropping the products bought in last month except for last three products
  data1=data1.drop(rem,axis=0)    

  inds=list(data1.index)
  train=data.drop(inds,axis=0)
  
  #Now train contains list of bought items only
  #So Randomly sampling it with Items not bought also
  #So as to prevent the trained model to overfit 
  #the data

  train_usr=list(train['User_Id'].values)
  train_itm=list(train['Item_Id'].values)
  train_rat=list(train['estimator'].values)
  for i in range(n_users):
    for j in range(n_items):
      temp=np.random.randint(n_items)
      if(ratings_matrix[i][temp]==0):
        train_usr.append(i)
        train_itm.append(temp)
        #Appending not bought items with 0.001(Instead of 0)
        #This will be used in the timw when we try to
        #diffrentiate between Zeros in train and test set.
      
        train_rat.append(0.001)

  #The Final training  and Testing Dataset
  trains=pd.DataFrame(train_usr,columns=['User_Id'])
  trains['Item_Id']=train_itm
  trains['estimator']=train_rat  
  trains=trains.drop_duplicates()

  test=data1[['User_Id','Item_Id','estimator']]


  usr=trains['User_Id'].values
  itm=trains['Item_Id'].values
  est=trains['estimator'].values

  for i in range(len(usr)):
    ratings_matrix[usr[i]][itm[i]]=-1*est[i]


  usr_test=list(test['User_Id'].values)
  itm_test=list(test['Item_Id'].values)
  est_test=list(test['estimator'].values)

  for i in range(n_users):
    for j in range(n_items):
      if(ratings_matrix[i][j]>=0):
        usr_test.append(i)
        itm_test.append(j)
        est_test.append(ratings_matrix[i][j])

  tests=pd.DataFrame(usr_test,columns=['User_Id'])
  tests['Item_Id']=itm_test
  tests['estimator']=est_test  

  import matplotlib.pyplot as plt
  import seaborn as sns
  !pip install surprise
  from surprise import NormalPredictor, Reader, Dataset, accuracy, SVD, CoClustering

  def surprise_df(data):
    
    scale = (data.estimator.min(), data.estimator.max())
    reader = Reader(rating_scale=scale)

    df = Dataset.load_from_df(data[['User_Id',
                                    'Item_Id',
                                    'estimator']], reader)
    
    return df

  #Converting to form suitable for Scikit Surprise
  train_df=surprise_df(trains)
  test_df=surprise_df(test)
  train_set=train_df.build_full_trainset()
  test_set=test_df.build_full_trainset()


  #Co-clustering algorithm trained on Train_set
  co = CoClustering(n_cltr_u=3,n_cltr_i=3,n_epochs=20)         
  co.fit(train_set)

  
  trains=trains.drop_duplicates()
  tests=tests.drop_duplicates()
  #Transforming to reqd. form for Nueral Embedding model
  X_train = trains[['User_Id', 'Item_Id']].values
  y_train = trains['estimator'].values

  X_test = tests[['User_Id', 'Item_Id']].values
  y_test = tests['estimator'].values

  X_train_array = [X_train[:, 0], X_train[:, 1]]
  X_test_array = [X_test[:, 0], X_test[:, 1]]

  #Importing Required libraries
  from keras.layers import LSTM,Dense,Embedding,Conv1D,Reshape,Dot,Add,Activation
  from keras.layers import LeakyReLU,ReLU
  from keras.optimizers import Adam 
  from keras.models import Model,Input
  from keras.callbacks import ModelCheckpoint
  from keras.regularizers import l2
  from keras.layers import Add, Activation, Lambda


  from keras.layers import Concatenate, Dense, Dropout,LSTM
  def RecommenderRNN(n_users, n_items, n_factors, min_rating, max_rating):
    user = Input(shape=(1,))
    u = Embedding(n_users, n_factors)(user)
    
    movie = Input(shape=(1,))
    m = Embedding(n_items, n_factors)(movie)
    
    x = Concatenate()([u, m])
    x = Dropout(0.05)(x)
    
    x=LSTM(50,recurrent_initializer='glorot_uniform')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1, kernel_initializer='he_normal')(x)
    x = Activation('sigmoid')(x)
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)
    model = Model(inputs=[user, movie], outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


  #Now we make and train the Model
  model_RNN = RecommenderRNN(n_users, n_items, n_factors=256, min_rating=0, max_rating=1)
  model_RNN.summary()

  #Using  Modelcjeckpoint
  calls=ModelCheckpoint('/content/temp',monitor='val_loss',mode='min',save_best_only=True)
  model_RNN.fit(x=X_train_array, y=y_train, batch_size=64, epochs=5,callbacks=[calls],
                    verbose=1, validation_split=0.2)
  
  model_RNN.load_weights('/content/temp')
    
  return(tests,test,model_RNN,co,lab_user,lab_item)


tests1,test1,model_RNN,co1,lab_user,lab_item=Neural_embed_Reco_train('/content/drive/My Drive/Colab Notebooks/Recommendation_system/large_beauty_original.csv')


def predict_testset(test,tests,model_RNN,co,Top_k=5):
  #This is used to make predictions on the Test set
  user_id=tests['User_Id'].values
  item_id=tests['Item_Id'].values
  rat=tests['estimator'].values

  #We store predictions from the co-clustering model
  #In pred_co and which is the append to tests
  preds_co=[]
  for i in range(len(user_id)):
    preds_co.append(co.predict(user_id[i],item_id[i])[3])
  
  tests['Predicted_co']=preds_co

  #We form test array from tests
  X_test = tests[['User_Id', 'Item_Id']].values
  y_test = tests['estimator'].values
  X_test_array = [X_test[:, 0], X_test[:, 1]]

  #Predictions from RNN based model
  preds_RNN=model_RNN.predict(X_test_array)
  tests['Predicted_RNN']=preds_RNN

  #Now considering "test" dataset
  #That has only the required items
  #From last month
  usr=test['User_Id'].values
  itm=test['Item_Id'].values

  #This dictionary stores the Item_list
  #of each user
  usr_dict={}
  for i in range(len(usr)):
    usr_dict[usr[i]]=[]
  for i in range(len(usr)):
    usr_dict[usr[i]].append(itm[i])

  #We develop a new ratings matrix
  #We have all the entries to be -1
  #Except for those in 'tests'
  new_rat=-1*(np.ones((n_users,n_items)))
  new_rat1=-1*(np.ones((n_users,n_items)))
  res_usr=tests['User_Id'].values
  res_itm=tests['Item_Id'].values
  res_pred=tests['Predicted_co'].values
  res_pred1=tests['Predicted_RNN'].values
  
  for i in range(len(res_usr)):
    new_rat[res_usr[i]][res_itm[i]]=res_pred[i] 


  for i in range(len(res_usr)):
    new_rat1[res_usr[i]][res_itm[i]]=res_pred1[i]   

  #This stores the predicted Item_ids
  #As recommendations for users
  pred_user_dicts_co={}
  for i in range(n_users):
    pred_user_dicts_co[i]=[]
  
  pred_user_dicts_RNN={}
  for i in range(n_users):
    pred_user_dicts_RNN[i]=[]  

  for i in range(n_users):
    temp=new_rat[i]
    temp1=new_rat1[i]
    #Choosing the one with max rating as prediction
    pred_user_dicts_co[i].append(np.argmax(temp))
    pred_user_dicts_RNN[i].append(np.argmax(temp1))
    for j in range(Top_k-1):
      temp[np.argmax(temp)]*=-1
      pred_user_dicts_co[i].append(np.argmax(temp))
      temp1[np.argmax(temp1)]*=-1
      pred_user_dicts_RNN[i].append(np.argmax(temp1))

  #Finding the accuracy
  #Accuracy is considered as percentage of
  #Correctly predicted users for whom at least one 
  #item is recommended correctly
  f=0
  for i in usr_dict.keys():
    for j in pred_user_dicts_co[i]:
      if(j in usr_dict[i]):
        f=f+1
        break

  print('Accuracy of Recommendations using Co_Clustering Model: {}'.format(np.round((f/len(usr_dict))*100,2)))   


  f=0
  for i in usr_dict.keys():
    for j in pred_user_dicts_RNN[i]:
      if(j in usr_dict[i]):
        f=f+1
        break

  print('Accuracy of Recommendations using RNN+Embedding Based Model: {}'.format(np.round((f/len(usr_dict))*100,2))) 

  return(pred_user_dicts_RNN,pred_user_dicts_co,tests)

#This function is used to predict the ratings for a user product pair
#orig is a boolean which is True if we pass original User and Item id
#Instaed of the label encoded ones
def predict_rating(user_id,item_id,model_RNN,co,lab_user,lab_item,orig=False):
  
  if(orig==True):
    #Converting the given user id and item id from original Datasets
    try:
      user_id=lab_user.transform([user_id])
      item_id=lab_item.transform([item_id])
    except:
      print('We have new user or item')
      
  
  test=np.array([[user_id,item_id]])
  test_arr=[test[:, 0], test[:, 1]]
  try:
    RNN_rating=model_RNN.predict(test_arr)[0][0]
    CO_rating=co.predict(user_id,item_id)[3]
  except:

    print('New user or item .Please use Popularity recommendor.')

 
  print('Ratings predicted for Item  by RNN+Embed Based Model:{}'.format(np.round(RNN_rating,2)))
  print('Ratings predicted for Item  by Co-clustering: {}'.format(np.round(CO_rating,2)))
  

  return(RNN_rating,CO_rating)





    