def Bipartite_graph_Reco(path):
  import pandas as pd
  import numpy as np
  import networkx as nx
  import matplotlib.pyplot as plt
  from community import community_louvain
  %matplotlib inline
	#Importing the dataset
  data=pd.read_csv(path)
  
  #Label Encoding Item and user
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

  ratings_matrix=np.zeros((n_users,n_items))
  for i in ratings.index:
    ratings_matrix[ratings['User_Id'][i]][ratings['Item_Id'][i]]=1

  ratings_df=pd.DataFrame(ratings_matrix)

  #Initialize zeros dataframe for product interactions
  products_integer = np.zeros((n_items,n_items))

  #Counting how many times each product pair has been purchased
  print('Counting how many times each pair of products has been purchased...')
  for i in range(n_items):
    for j in range(n_items):
      if (i != j):
        df_ij = ratings_df.iloc[:,[i,j]] #create a temporary df with only i and j products as columns
        sum_ij = df_ij.sum(axis=1)
        pairings_ij = len(sum_ij[sum_ij == 2]) #if s1_ij == 2 it means that both products were purchased by the same customer
        products_integer[i,j] = pairings_ij
        products_integer[j,i] = pairings_ij  

  #Counting how many customers have purchased each item
  print('Counting how many times each individual product has been purchased...')
  times_purchased = products_integer.sum(axis = 1)

  #Construct final weighted matrix of item interactions
  print('Building weighted product matrix...')
  products_weighted = np.zeros((n_items,n_items))
  for i in range(n_items):
    for j in range(n_items):
      if (times_purchased[i]+times_purchased[j]) !=0: 
        #make sure you do not divide with zero
        products_weighted[i,j] = (products_integer[i,j])/(times_purchased[i]+times_purchased[j])

  ratings_binary=ratings_df

  #Getting list of item labels (instead of Codes)
  nodes_codes = np.array(ratings_binary.columns).astype('str')
  nodes_labels = [[code] for code in nodes_codes]


  #Creating Graph object using the weighted product matrix as adjacency matrix
  G = nx.from_numpy_matrix(products_weighted)
  pos=nx.random_layout(G)
  labels = {}
  for idx, node in enumerate(G.nodes()):
    labels[node] = nodes_labels[idx]

  nx.draw_networkx_nodes(G, pos , node_color="skyblue", node_size=30)
  nx.draw_networkx_edges(G, pos,  edge_color='k', width= 0.3, alpha= 0.5)
  nx.draw_networkx_labels(G, pos, labels, font_size=4)
  plt.axis('off')
  plt.figure(figsize=(20,20))
  plt.show() # display


  #Finding communities of nodes (products)
  partition = community_louvain.best_partition(G, resolution = 0.75)
  values = list(partition.values())

  #Checking how many communities were created
  print('Number of communities:', len(np.unique(values)))

  #Creating dataframe with Item_id and community id
  products_communities = pd.DataFrame(nodes_labels, columns = ['Item_Id'])
  products_communities['community_id'] = values


  #A dictionary consisting of Items in each community
  comm_dicts={}
  for i in range(len(products_communities)):
    comm_dicts[products_communities['community_id'][i]]=[]
  for i in range(len(products_communities)):  
   	comm_dicts[products_communities['community_id'][i]].append(products_communities['Item_Id'][i])

  #A reverse dict consisting of Item_Id: Community_Id to which it belongs
  rev_comm_dicts={}
  for i in range(n_items):
    rev_comm_dicts[i]=products_communities['community_id'][i]

  return(products_communities,comm_dicts,rev_comm_dicts)