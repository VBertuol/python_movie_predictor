# Created by Vinicius Bertuol.
# The code below analyzes a CSV file that represents my watched movies spreadsheet, categorized by: genre, director, release year, IMDB rating, writer, actors and my personal rating from 0 to 10.
# The program's objective is to use Supervised Decision Tree Learning to determine, based on six parameters: genre, director, release year, IMDB rating, writer and actors, whether a movie is good or not.
# For this purpose, the average rating of directors, genre, writers and actors are made, while the other parameters are used whith it's absolute value
# The database was taken off IMDB's website for free with only studying porpouses
# The full table with my watched movies can be seen in the "my_movies_df.tsv" file
# It's also important to note that movies are rated from 0 to 10, and we consider a movie "good" if its rating is equal to or greater than 7.5.
# The generated result can be viewed in the terminal, and the decision tree in the "movies.png" file.


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn import tree
import pydotplus
import os

# Read dataset into data frame
movies_df = pd.read_csv("my_movies_df.tsv", sep='\t')

# Use the groupby method to group the movies by director and then calculate the average ratings.
directors_avg_ratings = movies_df.groupby("Diretor")["Nota"].mean().reset_index()

# Rename the column "nota" to "media_nota" (average rating).
directors_avg_ratings = directors_avg_ratings.rename(columns={"nota": "media_nota"})

# Use the groupby method to group the movies by writer and then calculate the average ratings.
writers_avg_ratings = movies_df.groupby("Escritor")["Nota"].mean().reset_index()

# Rename the column "nota" to "media_nota" (average rating).
writers_avg_ratings = writers_avg_ratings.rename(columns={"nota": "media_nota"})

# split names
actors_df = movies_df["Ator/Atriz"].str.split(", ").explode()

# new DataFrame with columns "Ator/Atriz" and "Nota"
actors_avg_ratings = pd.DataFrame({
    "Ator/Atriz": actors_df,
    "Nota": movies_df["Nota"]
})

# Use the groupby method to calculate the average ratings.
actors_avg_ratings = actors_avg_ratings.groupby("Ator/Atriz")["Nota"].mean().reset_index()

# Rename the column "nota" to "media_nota" (average rating).
actors_avg_ratings = actors_avg_ratings.rename(columns={"Nota": "media_nota"})

# Use the groupby method to group the movies by genre and then calculate the average ratings.
genders_avg_ratings = movies_df.groupby("Gênero")["Nota"].mean().reset_index()

# Rename the column "nota" to "media_nota" (average rating).
genders_avg_ratings = genders_avg_ratings.rename(columns={"nota": "media_nota"})

# Function to get the director's average from the DataFrame directors_avg_ratings.
def get_director_average(row):
    director = row['Diretor']
    avg_rating = directors_avg_ratings.loc[directors_avg_ratings["Diretor"] == director, "Nota"].values
    if len(avg_rating) > 0:
        return avg_rating[0]
    else:
        return 0.0
    
def get_writer_average(row):
    writer = row['Escritor']
    if pd.notna(writer):
        avg_rating = writers_avg_ratings.loc[writers_avg_ratings["Escritor"] == writer, "Nota"].values
        if len(avg_rating) > 0:
            return avg_rating[0]
    return writers_avg_ratings["Nota"].mean()

    
# Function to get the genre's average from the DataFrame genders_avg_ratings.
def get_genre_average(row):
    genre = row['Gênero']
    avg_rating = genders_avg_ratings.loc[genders_avg_ratings["Gênero"] == genre, "Nota"].values
    if len(avg_rating) > 0:
        return avg_rating[0]
    else:
        return 0.0
    
def get_actor_average(row):
    actors = row['Ator/Atriz']
    
    # is string
    if isinstance(actors, str):
        actors = actors.split(', ')  # split names
    else:
        actors = []

    avg_ratings = [] 

    for actor in actors:
        avg_rating = actors_avg_ratings.loc[actors_avg_ratings["Ator/Atriz"] == actor, "media_nota"].values
        if len(avg_rating) > 0:
            avg_ratings.append(avg_rating[0])

    if avg_ratings:
        return sum(avg_ratings) / len(avg_ratings)
    else:
        return actors_avg_ratings["media_nota"].mean()

# Apply the functions
movies_df['Média Atores'] = movies_df.apply(get_actor_average, axis=1)
movies_df['Média Diretor'] = movies_df.apply(get_director_average, axis=1)
movies_df['Média Gênero'] = movies_df.apply(get_genre_average, axis=1)
movies_df['Média Escritor'] = movies_df.apply(get_writer_average, axis=1)


#This value can be changed to modify how good the movie must be, scores are between 0 and 10, >= 7.5 is at least a "good" movie, >= 6 would be at least an "OK" movie, >= 8 would be at least a "very good" movie
movies_df["Good Movie"] = (movies_df["Nota"] >= 7.5).astype(int)

# Define labels (output) as Good Movie
# Good Movie is the target, that we want to predict from the values of the other columns
classes = ["No", "Yes"] # 0 or 1
labels = "Good Movie"
y = movies_df["Good Movie"].values

# Not all of the other columns are helpful for classification
# So we choose a feature set by hand and convert the features into a numpy array for scikit learn
columns = ["Ano Lançamento", "Média Diretor", "Média Gênero", "Média Escritor", "rating_IMDB", "Média Atores"]
features = movies_df[list(columns)].values

# Replace 'nans' by mean to avoid issues
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(features)

# Learn the decision tree
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=7)
clf = clf.fit(X, y)

os.environ["PATH"] += os.pathsep + 'C:\ProgramData\anaconda3\Library\bin\dot'

# Export as png
#dot_data = tree.export_graphviz(clf, out_file=None, feature_names=columns) 
dot_data = tree.export_graphviz(clf, out_file=None,  feature_names=columns, 
                                class_names=classes,  filled=True,
                                rounded=True, special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('movies.png')


def get_director_rating(Diretor):
    director_avg = directors_avg_ratings[directors_avg_ratings["Diretor"] == Diretor]
    
    if not director_avg.empty:
        director_rating = director_avg["Nota"].values[0]
        
        return director_rating
    
    return None

def get_gender_rating(Genero):
    gender_avg = genders_avg_ratings[genders_avg_ratings["Gênero"] == Genero]
    
    if not gender_avg.empty:
        gender_rating = gender_avg["Nota"].values[0]
        
        return gender_rating
    
    return None

def get_writer_rating(Escritor):
    writer_avg = writers_avg_ratings[writers_avg_ratings["Escritor"] == Escritor]
    
    if not writer_avg.empty:
        writer_rating =  writer_avg["Nota"].values[0]
        
        return writer_rating
    
    return None

def get_actors_rating(actors):
    avg_ratings = [actors_avg_ratings.loc[actors_avg_ratings["Ator/Atriz"] == actor, "media_nota"].values[0] for actor in actors if actor in actors_avg_ratings["Ator/Atriz"].values]
    if avg_ratings:
        return sum(avg_ratings) / len(avg_ratings)
    return actors_avg_ratings["media_nota"].mean()


# Predict using our model
Filme = 'A Freira 2'
Ano_Lançamento = 2023
Diretor = 'Michael Chaves'
Gênero = 'Horror'
Escritor = 'Ian Goldberg'
rating_IMDB = 5.7
Atores = ('Taissa Farmiga', 'Jonas Bloquet', 'Storm Reid')

Atores1 = get_actors_rating(Atores)
Gênero1 = get_gender_rating(Gênero)
Diretor1 = get_director_rating(Diretor)
Escritor1 = get_writer_rating(Escritor)
print("\nGênero: " + Gênero + "\n")
print("Nota média do Gênero: " + str(Gênero1) + "\n")
print("Atores: " + str(Atores) + "\n")
print("Nota média dos Atores: " + str(Atores1) + "\n")
print("Diretor: " + Diretor + "\n")
print("Nota média do Diretor: " + str(Diretor1) + "\n")
print("Escritor: " + Escritor + "\n")
print("Nota média do Escritor: " + str(Escritor1) + "\n")
print("Rating no IMDB: " + str(rating_IMDB) + "\n")


# Predict a single decision
if(clf.predict([[Ano_Lançamento, Diretor1, Gênero1, Escritor1, rating_IMDB, Atores1]])):
    print("O Filme " + Filme + " é bom!\n")
else:
    print("O Filme " + Filme + " é ruim!\n")

# Predict probability of decision per class
print(clf.predict_proba([[Ano_Lançamento, Diretor1, Gênero1, Escritor1, rating_IMDB, Atores1]]))

