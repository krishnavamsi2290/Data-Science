#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[5]:


Abc = 100,000,000


# In[6]:


A,b,c = 100,200,300


# In[7]:


A b c = 100 200 300


# In[8]:


A_b_c = 100,000,000


# In[9]:


c =(-8,-6)
d = (-4,5)


# In[10]:


import numpy as np


# In[11]:


arr = np.array(c)
arr1 = np.array(d)


# In[12]:


arr@arr1


# In[13]:


a = (5,2,-2)
b = (2,-7,-2)
arr = np.array(a)
arr1 = np.array(b)


# In[14]:


arr2 = np.dot(arr,arr1)


# In[15]:


arr2


# In[16]:


data = {'Review': ['Negative', 'Negative', 'Negative', 'Positive','Positive','Negative','Positive','Positive','Positive','Negative'],
        'Taste': ['Sweet', 'Salty','Salty','Sour','Sour', 'Sweet','Sour','Salty','Salty', 'Sweet'],
        'Smell': ['Woody','Fruity','Fruity','Fruity', 'Woody', 'Woody', 'Woody','Fruity','Fruity','Woody'],
        'Portion_Size': ['Small','Large','Large', 'Small','Small', 'Large','Large','Small','Small', 'Large']}


# In[17]:


df = pd.DataFrame(data)


# In[ ]:


import pandas as pd


# In[ ]:


df


# In[18]:


x=df["Review"]
y = df["Taste"]


# In[ ]:





# In[19]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Assuming you have a dataset in a pandas DataFrame
# Replace this with your actual dataset
data = {
    'Smell': ['Pleasant', 'Unpleasant', 'Pleasant', 'Unpleasant', 'Pleasant', 'Unpleasant'],
    'Taste': ['Sweet', 'Salty', 'Spicy', 'Sweet', 'Salty', 'Spicy'],
    'Portion_Size': ['Large', 'Small', 'Medium', 'Large', 'Small', 'Medium'],
    'Review': ['Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Negative']
}

df = pd.DataFrame(data)

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Smell', 'Taste', 'Portion_Size'])

# Separate features (X) and target variable (y)
X = df.drop('Review', axis=1)
y = df['Review']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier(max_depth = 2)

# Train the Decision Tree model
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_rep)


# In[ ]:


fight_club_ratings = df_movie_rating[df_movie_rating['title'] == 'Fight Club (1999)']

# Plot a histogram of user ratings
plt.figure(figsize=(10, 6))
sns.histplot(fight_club_ratings['rating'], bins=10, kde=True)
plt.title('Distribution of User Ratings for Fight Club (1999)')
plt.xlabel('User Ratings')
plt.ylabel('Frequency')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Filter the DataFrame to include only rows for "Fight Club (1999)"
fight_club_ratings = df_movie_rating[df_movie_rating['title'] == 'Fight Club (1999)']

# Plot a histogram of user ratings
plt.figure(figsize=(10, 6))
sns.histplot(fight_club_ratings['rating'], bins=10, kde=True)
plt.title('Distribution of User Ratings for Fight Club (1999)')
plt.xlabel('User Ratings')
plt.ylabel('Frequency')
plt.show()



# In[20]:


import matplotlib.pyplot as plt


# In[21]:


import pandas as pd
import numpy as np


# In[22]:


df_links = pd.read_csv("links.csv")


# In[23]:


df_movies = pd.read_csv("movies.csv")


# In[24]:


df_rating = pd.read_csv("ratings.csv")


# In[25]:


df_tags = pd.read_csv("tags.csv")


# In[26]:


df_links


# In[27]:


df_movies


# In[28]:


df_rating


# In[29]:


df_tags


# In[30]:


df_rating["userId"].nunique()


# In[31]:


df_movies


# In[32]:


df_rating


# In[33]:


df_movie_rating = pd.merge(df_movies,df_rating,on="movieId")


# In[34]:


df_movie_rating


# In[35]:


df_movie_rating.isnull().sum()


# In[36]:


df_movie_rating["rating"] = df_movie_rating["rating"].astype("int")


# In[37]:


df_movie_rating


# In[38]:


group_by = df_movie_rating.groupby('title')['userId'].count()


# In[39]:


group_by.idxmax()


# In[40]:


group_by


# In[41]:


df_movie_rating["title"].value_counts()


# In[42]:


df_tags 


# In[43]:


df_movie_tags = pd.merge(df_movies,df_tags,on="movieId")


# In[44]:


df_movie_tags


# In[45]:


df_movie_tags.isnull().sum()


# In[46]:


df_movie_tags.head()


# In[47]:


df_movie_tags[df_movie_tags["title"]=="Matrix, The (1999)"]["tag"]


# In[48]:


df_movie_rating


# In[49]:


df_movie_rating[df_movie_rating["title"]=="Terminator 2: Judgment Day (1991)"]["rating"].mean()


# In[ ]:





# In[ ]:





# In[50]:


c =df_movie_rating[df_movie_rating["title"]=="Fight Club (1999)"]["rating"]


# In[51]:


import seaborn as sns


# In[52]:


sns.kdeplot(data=df_movie_rating, x=c)


# In[53]:


df_movie_rating[df_movie_rating["title"]=="Godfather, The (1972)"]["rating"].mean()


# In[ ]:





# In[54]:


df_rating


# In[55]:


df = df_rating.groupby(["movieId"])


# In[56]:


df_rating1_mean=df["rating"].agg(["mean","count"])


# In[57]:


df_rating1_mean


# In[58]:


df_movies


# In[59]:


df_rat = pd.merge(df_movies,df_rating1_mean,on="movieId")


# In[60]:


df_mov_rat = df_rat.loc[df_rat["count"]>50]


# In[61]:


df_mov_rat


# In[62]:


df_mov_rat.loc[df_mov_rat["title"]=="Godfather, The (1972)"]


# In[63]:


df_mov_rat.loc[df_mov_rat["title"]=="Shawshank Redemption, The (1994)"]


# In[64]:


df_mov_rat.loc[df_mov_rat["title"]=="Jumanji (1995)"]


# In[65]:


df_mov_rat.loc[df_mov_rat["title"]=="Wolf of Wall Street, The (2013)"]


# In[66]:


df_mov_rat.sort_values(by="count",ascending=False)


# In[67]:


df_mov_rat.loc[df_mov_rat["title"]=="Deadpool (2016)"]


# In[ ]:





# In[68]:


df_mov_rat.loc[df_mov_rat["title"]=="X-Men: The Last Stand (2006)"]


# In[69]:


df_mov_rat.loc[df_mov_rat["title"]=="Jurassic Park (1993)"]


# In[70]:


df_mov_rat.loc[df_mov_rat["genres"].str.contains("Sci-Fi")].sort_values(by ="count",ascending=False)


# In[71]:


df_links


# In[72]:


import requests as rs
from bs4 import BeautifulSoup


# In[73]:


df_links


# In[74]:


df_rt = pd.merge(df_mov_rat,df_links,on="movieId")


# In[75]:


df_rt


# In[76]:


df_rt.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[84]:


rating =[]
import requests
import numpy as np
from bs4 import BeautifulSoup
for i in df_rat["imdbid"]:
    id = str(int(i))
    n_zeroes = 7 - len(id)
    new_id = "0"*n_zeroes + id
    URL = f"https://www.imdb.com/title/tt{new_id}/"
    request_header = {'Content-Type': 'text/html; charset=UTF-8', 
                      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0', 
                      'Accept-Encoding': 'gzip, deflate, br'}
    response = requests.BeautifulSoup(URL, headers=request_header)
    soup =respone(response.text)
    imdb_rating = soup.find('div',attrs={'class':'sc-f056af46-3 dzYxjh'})
    if imdb_rating:
        rating.append(imdb_rating.text)
    else:
         rating.append(np.nan)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




