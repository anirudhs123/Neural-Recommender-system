def Apriori_asso_Reco(path):   
    data=pd.read_csv(path)
    #Label Encoding Item and users
    from sklearn.preprocessing import LabelEncoder
    lab_item=LabelEncoder()
    data['Item_Id']=lab_item.fit_transform(data['Product Id'])

    lab_user=LabelEncoder()
    data['User_Id']=lab_user.fit_transform(data['User Id'])

    #Removing the unwanted columns , after it was renamed appropriately
    del data['User Id'],data['Product Id']

    ratings=data[['Item_Id','User_Id']]

    #Finding out the unique users and items
    n_users=max(ratings['User_Id'])+1
    n_items=max(ratings['Item_Id'])+1

    #To account for the interactions we use the rating matrix
    ratings_matrix=np.zeros((n_users,n_items))
    for i in ratings.index:
    	ratings_matrix[ratings['User_Id'][i]][ratings['Item_Id'][i]]=1
    
    #The computed Ratings matrix is converted to dataframe
    ratings_df=pd.DataFrame(ratings_matrix)

    #Using the Apriori algorithm
    from mlxtend.frequent_patterns import apriori, association_rules
    frequent_itemsets = apriori(ratings_df, min_support=0.1,use_colnames=True, max_len=2)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=2)
    #Taking only the associated items
    arm=rules[['antecedents','consequents']]
 
    #The items from rules are Fixed sets
    #Hence cannot be manipulated.So converting them to lists.
    def list_conv(item):
    	return(list(item)[0])

    arm['antecedents']=arm['antecedents'].apply(lambda x: list_conv(x))
    arm['consequents']=arm['consequents'].apply(lambda x: list_conv(x))

   
    #This makes a dictionary for each item
    #and a list of all its associated item
    dict_item={}
    for i in range(len(arm)):
      dict_item[arm['antecedents'][i]]=[]

    for i in range(len(arm)):
      dict_item[arm['antecedents'][i]].append(arm['consequents'][i])

    return(arm,rules,dict_item)

  

