def popularity_reco(path,israted=False):
  data=pd.read_csv(path)

  #Divide the dataset based on the Time at which 
  #Product is bought. We tend to use all except 
  #Last  month to train the model
  
  #last_month=data['Month'].value_counts().sort_index(ascending=False).index[0]
  
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
  #Item_feq_dict
  Item_freq_dict={}
  for i in range(n_items):
    Item_freq_dict[i]=0
  for i in range(len(ratings)):
    Item_freq_dict[ratings['Item_Id'].iloc[i]]+=1

  it_id=list(Item_freq_dict.keys())
  it_vals=list(Item_freq_dict.values())
  it_id.sort(reverse=True)
  it_vals.sort(reverse=True)  

  import operator
  it_id=sorted(Item_freq_dict.items(),key=operator.itemgetter(1),reverse=True)
  pop_df=pd.DataFrame(it_id,columns=['Item_Id','Frequency_Count'])
  return(pop_df,it_id)

