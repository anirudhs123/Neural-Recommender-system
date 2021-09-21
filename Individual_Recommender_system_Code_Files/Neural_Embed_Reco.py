def Neural_embed_Reco_train(X_train_array,y_train,model_learning_rate=0.001,latent_dim=256,model_batch_size=64,model_EPOCHS=5,model_val_split=0.2):
  #Read the data to train the model
  #We now encode the Users and Items using LabelEnocder
  from sklearn.preprocessing import LabelEncoder
  #Importing Required libraries
  from keras.layers import LSTM,Dense,Embedding,Conv1D,Reshape,Dot,Add,Activation
  from keras.layers import LeakyReLU,ReLU
  from keras.optimizers import Adam 
  from keras.models import Model,Input
  from keras.callbacks import ModelCheckpoint
  from keras.regularizers import l2
  from keras.layers import Add, Activation, Lambda
  from keras.layers import Concatenate, Dense, Dropout,LSTM
  from sklearn.utils import shuffle

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
    opt = Adam(lr=model_learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


  #Now we make and train the Model
  model_RNN = RecommenderRNN(n_users, n_items, n_factors=latent_dim, min_rating=min_rating, max_rating=max_rating)
  model_RNN.summary()

  #Using  Modelcjeckpoint
  calls=ModelCheckpoint('/content/temp',monitor='val_loss',mode='min',save_best_only=True)
  model_RNN.fit(x=X_train_array, y=y_train, batch_size=model_batch_size, epochs=model_EPOCHS,callbacks=[calls],
                    verbose=1, validation_split=model_val_split)
  
  model_RNN.load_weights('/content/temp')
    
  return(tests,test,model_RNN,lab_user,lab_item,n_users,n_items)



def predict_testset(train,test,tests,model_RNN,n_users,n_items,Top_k=5):
  #This is used to make predictions on the Test set
  user_id=tests['User_Id'].values
  item_id=tests['Item_Id'].values
  rat=tests['estimator'].values

  

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
  
  new_rat1=(np.zeros((n_users,n_items)))
  res_usr=tests['User_Id'].values
  res_itm=tests['Item_Id'].values
  res_pred1=tests['Predicted_RNN'].values
  for i in range(len(res_usr)):
    new_rat1[res_usr[i]][res_itm[i]]=res_pred1[i]  


  res_usr=train['User_Id'].values
  res_itm=train['Item_Id'].values
  for i in range(len(res_usr)):
    new_rat1[res_usr[i]][res_itm[i]]=-1 

  #This stores the predicted Item_ids
  #As recommendations for users
  pred_user_dicts_RNN={}
  for i in range(n_users):
    pred_user_dicts_RNN[i]=[]  
  
  pred_user_dicts_RNN_values={}
  for i in range(n_users):
    pred_user_dicts_RNN_values[i]=[]   

  for i in range(n_users):
    temp=new_rat[i]
    temp1=new_rat1[i]
    #Choosing the one with max rating as prediction
    
    pred_user_dicts_RNN[i].append(np.argmax(temp1))
    pred_user_dicts_RNN_values[i].append(max(temp1))
    for j in range(Top_k-1):
      temp1[np.argmax(temp1)]*=-1
      pred_user_dicts_RNN[i].append(np.argmax(temp1))
      pred_user_dicts_RNN_values[i].append(max(temp1))

  #Finding the accuracy
  #Accuracy is considered as percentage of
  #Correctly predicted users for whom at least one 
  #item is recommended correctly
  f=0
  for i in usr_dict.keys():
    for j in pred_user_dicts_RNN[i]:
      if(j in usr_dict[i]):
        f=f+1
        break

  print('Accuracy of Recommendations using RNN+Embedding Based Model: {}'.format(np.round((f/len(usr_dict))*100,2))) 

  return(pred_user_dicts_RNN,pred_user_dicts_RNN_values,usr_dict,pred_user_dicts_co,tests)



def predict_rating(user_id,item_id,model_RNN):
  try:
  	test=np.array([[user_id,item_id]])
    test_arr=[test[:, 0], test[:, 1]]
    RNN_rating=model_RNN.predict(test_arr)[0][0]
    print('Ratings predicted for Item  by RNN+Embed Based Model:{}'.format(np.round([RNN_rating],2)))
    return(RNN_rating)
  except:
  	print('Please enter valid User_Id and Item_Id')


def preprocess_data(data,Test_k=5,israted=False):
  #Divide the dataset based on the Time at which 
  #Product is bought. We tend to use all except 
  #Last  month to train the model
  
  last_month=data['Month'].value_counts().sort_index(ascending=False).index[0]
  

  lab_item=LabelEncoder()
  data['Item_Id']=lab_item.fit_transform(data['Product Id'])

  lab_user=LabelEncoder()
  data['User_Id']=lab_user.fit_transform(data['User Id'])
 
  #We delete the unwanted cols 
  del data['User Id'],data['Product Id']
  ratings=data[['Item_Id','User_Id','Qty']]

  #We find the number of Unique users and Items 
  n_users=max(ratings['User_Id'])+1
  n_items=max(ratings['Item_Id'])+1

  #We now create a ratings matrix with corresponding User and Item values
  if(israted==False):
    ratings_matrix=np.zeros((n_users,n_items))
    for i in range(len(ratings)):
      ratings_matrix[ratings['User_Id'][i]][ratings['Item_Id'][i]]=1
    min_rating=0
    max_rating=1  

    estimator=[]
    for i in range(len(data)):
      estimator.append(ratings_matrix[data['User_Id'][i]][data['Item_Id'][i]])

    data['estimator']=estimator    

  #If the dataset is rated we use that as our measure
  if(israted==True):
    ratings_matrix=np.zeros((n_users,n_items))
    for i in range(len(data)):
      ratings_matrix[data['User_Id'][i]][data['Item_Id'][i]]=data['rating'][i]
    min_rating=min(data['rating'].values)
    max_rating=max(data['rating'].values)
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
    if(usr_dict1[data1['User_Id'][i]] < Test_k):
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
      
        train_rat.append(0)

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

  trains=trains.drop_duplicates()
  tests=tests.drop_duplicates()

  
  trains = shuffle(trains)
  tests=shuffle(tests)
  #Transforming to reqd. form for Nueral Embedding model
  X_train = trains[['User_Id', 'Item_Id']].values
  y_train = trains['estimator'].values

  X_test = tests[['User_Id', 'Item_Id']].values
  y_test = tests['estimator'].values

  X_train_array = [X_train[:, 0], X_train[:, 1]]
  X_test_array = [X_test[:, 0], X_test[:, 1]]

  return(train,test,tests,X_train_array,X_test_array,y_train,y_test,n_users,n_items)


def main():
	import pandas as pd
	print('Please enter the path for the dataset to be read :')
	path=Input()
	data=pd.read_csv(path)

	train,test,tests,X_train_array,X_test_array,y_train,y_test,n_users,n_items=preprocess_data(data,Test_k)



	print('Please eneter Maximum nubmer of products in test set for each user {Default:5} :')
	Test_k=int(Input())

	print('Please enter learning rate to be used for training {Default:0.001} :')
	model_learning_rate=float(Input())

	print('Please enter Embedding Dimension {Default:256} :')
	latent_dim=int(Input())

	print('Please enter Batch_size for training {Default:64} :')
	model_batch_size=int(input())

	print('Please enter Nmber of Epochs for training {Default:5} :')
	model_EPOCHS=int(Input())

	print('Please enter validation_split needed for training {Default:0.2} :')
	model_val_split=float(Input())

    tests1,test1,model_RNN,lab_user,lab_item,n_users,n_items=Neural_embed_Reco_train(model_learning_rate,latent_dim,model_batch_size,model_EPOCHS,model_val_split)
    
    print('Please enter How many recommendations you want:')
    Top_k=int(Input())
    pred_user_dicts_RNN,pred_user_dicts_RNN_values,usr_dict,pred_user_dicts_co,tests=predict_testset(train,test,tests,model_RNN,n_users,n_items)
    
    print('Please enter User_Id and Item_Id to obtain ratings prediction :')
    user_id,item_id=map(int,input().split())
    Predicted_rating=predict_rating(train,user_id,item_id,model_RNN)


if __name__=='__main__':
	main()


      