import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Initialize rating matrix
rating_mat = np.empty((6040, 3952), dtype=np.int64)

# Read the data from 'ratings.dat'
with open('ratings.dat', 'r') as file:  # Place ratings.dat in the same folder as this python file
    for line in file:
        user_id, movie_id, rating, timestamp = line.strip().split('::')
        user_id = int(user_id)
        movie_id = int(movie_id)
        rating = int(rating)
        rating_mat[user_id - 1][movie_id - 1] = rating  # Assign rating to matrix

# Perform KMeans clustering
km = KMeans(n_clusters=3, random_state=21)
km.fit(rating_mat)  # Fit to data

groups = km.predict(rating_mat)

# Assign groups to each group 1, 2, 3
group_1 = pd.DataFrame(rating_mat[groups == 0])
group_2 = pd.DataFrame(rating_mat[groups == 1])
group_3 = pd.DataFrame(rating_mat[groups == 2])

def AU(group):
  col_sum = group.sum(axis=0)  # Get sum by columns
  sorted_sum = col_sum.sort_values(axis=0, ascending=False)  # Sort in descending order
  sorted_sum.index.name = 'Movie ID'  # Set index name
  return sorted_sum

def Avg(group):
  col_num = (group > 0).sum(axis=0)  # Get number of users(number of rows)
  AU_sum = AU(group)  # Call AU for sum by columns
  average = AU_sum / col_num
  sorted_average = average.sort_values(ascending=False)
  sorted_average.index.name = 'Movie ID'
  return sorted_average

def SC(group):
  num_ratings = (group > 0).sum(axis=0)  # Get number of ratings excluding '0' cells(no rating)
  sorted_num_ratings = num_ratings.sort_values(ascending=False)
  sorted_num_ratings.index.name = 'Movie ID'
  return sorted_num_ratings

def AV(group):
  threshold = 4
  pos_ratings = (group >= threshold).sum(axis=0)  # Get number of ratings > threshold
  sorted_pos_ratings = pos_ratings.sort_values(ascending=False)
  sorted_pos_ratings.index.name = 'Movie ID'
  return sorted_pos_ratings

def BC(group):
  # Get ranks by row, breaking ties by assigning the mean rank to each group
  ranks = group.rank(axis=1, method='average') - 1

  BC = ranks.sum(axis=0)  # Get sum of ranks by column(by movie)
  sorted_BC = BC.sort_values(ascending=False)
  sorted_BC.index.name = 'Movie ID'
  return sorted_BC

def CR(group):
  row_num, col_num = group.shape

  # Empty matrices to store number of wins / losses
  positive = np.zeros((col_num, col_num), dtype=int)
  negative = np.zeros((col_num, col_num), dtype=int)

  # Compare movie by movie pairs using broadcasting for each row(user)
  for i in range(row_num):
    row_items = group.values[i]  # Row in Numpy array form

    # Calculate the difference between all movies by broadcasting
    diff_mat = row_items[None, :] - row_items[:, None]  # Stored in a 3D Numpy array

    positive += (diff_mat > 0).astype(int) # Update positive matrix
    negative += (diff_mat < 0).astype(int) # Update negative matrix

  T = pd.DataFrame(np.zeros((col_num, col_num)))  # Create CR matrix which stores the relative importance
  T[positive > negative] = 1  # More wins
  T[positive < negative] = -1  # More losses, ties are already 0

  CR = T.sum(axis=0)  # Get sum of CR by column(by movie)

  sorted_CR = CR.sort_values(ascending=False)
  sorted_CR.index.name = 'Movie ID'

  return sorted_CR

def print_results(group):
  print("by AU:")
  print(AU(group)[:10])
  print("\nby Avg:")
  print(Avg(group)[:10])
  print("\nby SC:")
  print(SC(group)[:10])
  print("\nby AV:")
  print(AV(group)[:10])
  print("\nby BC:")
  print(BC(group)[:10])
  print("\nby CR:")
  print(CR(group)[:10])


print("Top 10 Group Recommendation for Group 1")
print_results(group_1)

print("\nTop 10 Group Recommendation for Group 2")
print_results(group_2)

print("\nTop 10 Group Recommendation for Group 3")
print_results(group_3)

