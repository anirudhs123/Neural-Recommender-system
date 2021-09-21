def Turicreate_collab_train(path,israted=False):
  import time
  !pip install turicreate
  import turicreate as tc
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
  ratings=data[['Item_Id','User_Id','Qty']]

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
  train=train.drop_duplicates()


  test=data1[['User_Id','Item_Id','estimator']]

  ins=pd.merge(train,test,how='inner')
  u_i_pairs=[]
  for i in range(len(train)):
  	u_i_pairs.append((train['User_Id'].iloc[i],train['Item_Id'].iloc[i]))

  u_i_pairs_ins=[]
  for i in range(len(ins)):
  	u_i_pairs_ins.append((ins['User_Id'].iloc[i],ins['Item_Id'].iloc[i]))

  rems=[]
  for i in range(len(train)):
  	if(u_i_pairs[i] in u_i_pairs_ins):
  		rems.append(list(train.index)[i])

  train=train.drop(rems,axis=0) 


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




  trains=pd.DataFrame(train_usr,columns=['User_Id'])
  trains['Item_Id']=train_itm
  trains['binary']=train_rat
  
  usr=trains['User_Id'].values
  itm=trains['Item_Id'].values
  est=trains['binary'].values

  trains=trains.drop_duplicates()
  from sklearn.utils import shuffle
  trains=shuffle(trains)

  usr=trains['User_Id'].values
  itm=trains['Item_Id'].values
  est=trains['binary'].values


  for i in range(len(usr)):
  	if(ratings_matrix[usr[i]][itm[i]]==1):
  		ratings_matrix[usr[i]][itm[i]]=-1
    else:
    	ratings_matrix[usr[i]][itm[i]]=-0.001

  usr_test=list(test['User_Id'].values)
  itm_test=list(test['Item_Id'].values)
  est_test=list(test['binary'].values)

  for i in range(n_users):
  	for j in range(n_items):
  		if(ratings_matrix[i][j]>=0):
  			usr_test.append(i)
            itm_test.append(j)
            est_test.append(ratings_matrix[i][j])

  tests=pd.DataFrame(usr_test,columns=['User_Id'])
  tests['Item_Id']=itm_test
  tests['binary']=est_test      

  tests=tests.drop_duplicates()
  tests=shuffle(tests)    

  ins=pd.merge(trains,tests,how='inner')
  u_i_pairs=[]
  for i in range(len(trains)):
  	u_i_pairs.append((trains['User_Id'].iloc[i],trains['Item_Id'].iloc[i]))

  u_i_pairs_ins=[]
  for i in range(len(ins)):
  	u_i_pairs_ins.append((ins['User_Id'].iloc[i],ins['Item_Id'].iloc[i]))

  rems=[]
  for i in range(len(trains)):
  	if(u_i_pairs[i] in u_i_pairs_ins):
  		rems.append(list(trains.index)[i])

  trains=trains.drop(rems,axis=0) 




  trains=trains.drop_duplicates()
  tests=tests.drop_duplicates()

  train_data = tc.SFrame(trains)
  test_data = tc.SFrame(tests)

  #Quantity based Recommendations
  df_matrix = pd.pivot_table(data, values='Qty', index='User_Id', columns='Item_Id')
  df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())

  user_id = 'User_Id'
  item_id = 'Item_Id'
  users_to_recommend = list(tests['User_Id'].unique())
  n_rec = 10 # number of items to recommend
  n_display = 20

  def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
    if (name == 'pearson'):
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target, 
                                                    similarity_type='pearson')
        
    recom = model.recommend(users=users_to_recommend, k=n_rec)
    recom.print_rows(n_display)
    return model



  name = 'pearson'
  target = 'binary'
  pear = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

  
  models_w_dummy = [cos, pear]
  names_w_dummy = ['Cosine Similarity on Purchase ', 'Pearson Similarity on Purchase ']
  eval_dummy = tc.recommender.util.compare_models(test_data, models_w_dummy, model_names=names_w_dummy)

  

  pearson_model = tc.item_similarity_recommender.create(tc.SFrame(trains), 
                                            user_id=user_id, 
                                            item_id=item_id, 
                                            target='binary', similarity_type='pearson')
  recom_pearson = perason_model.recommend(users=users_to_recommend, k=5)
  df_rec_pearson = recom_pearson.to_dataframe()
  recom_pearson.print_rows(n_display)

  return(test,recom_pearson,df_rec_pearson)


def predictions_model_pearson(test,df_rec_pearson):
	usr=test['User_Id'].values
	itm=test['Item_Id'].values
	usr_dict={}
	for i in range(len(usr)):
		usr_dict[usr[i]]=[]
    for i in range(len(usr)):
    	usr_dict[usr[i]].append(itm[i])

    pred_user_dict_pear={}
    for i in list(df_rec_pearson['User_Id'].unique()):
    	pred_user_dict_pear[i]=[]

    for i in range(len(df_rec_pearson)):
    	if(ratings_matrix[df_rec_pearson['User_Id'][i]][df_rec_pearson['Item_Id'][i]]!=-1):
    		pred_user_dict_pear[df_rec_pearson['User_Id'][i]].append(df_rec_pearson['Item_Id'][i])

    f=0
    for i in usr_dict.keys():
    	for j in pred_user_dict_pear[i]:
    		if(j in usr_dict[i]):
    			f=f+1
    			break		

    print('Accuracy of Pearson_Similarity Recommender : {}'.format(f/len(usr_dict)))		

    return(predict_user_dict_pear)		

  



  	






