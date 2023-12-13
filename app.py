# -*- coding: utf-8 -*-
from flask import Flask, jsonify
import mysql.connector
import pandas as pd
import random
import numpy as np
from surprise import Reader, Dataset, SVD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input, Concatenate
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

NUM_RECOMMENDATIONS = 10
app = Flask(__name__)

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        port="3306",
        user="root",
        passwd="root",
        database="db_example"
    )

def get_fallback_recommendations(df, num_recommendations=10, fallback_strategy='popular'):
    if fallback_strategy == 'popular':
        popular_movies = df.groupby('movie_id')['user_id'].count().sort_values(ascending=False).index
        return popular_movies[:num_recommendations].tolist()
    elif fallback_strategy == 'random':
        all_movies = df['movie_id'].unique()
        return random.sample(all_movies.tolist(), min(num_recommendations, len(all_movies)))
    else:
        return []

# Plotting results
def plot1(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    ## Accuracy plot
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    ## Loss plot
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def plot2(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    #plt.gca().set_ylim(0,1)
    #plt.show()
    plt.savefig('Fig2.png')



def train_neural_collaborative_filtering(df):
    max_user_id = df['user_id'].max()
    max_movie_id = df['movie_id'].max()

    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')

    user_embedding = Embedding(input_dim=max_user_id + 1, output_dim=64)(user_input)
    movie_embedding = Embedding(input_dim=max_movie_id + 1, output_dim=64)(movie_input)

    concatenated = Concatenate()([Flatten()(user_embedding), Flatten()(movie_embedding)])
    dense = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='linear')(dense)

    model = Model(inputs=[user_input, movie_input], outputs=output)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

 
    model.compile(optimizer='adam', loss='mean_squared_error', metrics = metric) # -*-метод мінімізації похибки

    user_ids = df['user_id'].values.reshape((-1, 1))
    movie_ids = df['movie_id'].values.reshape((-1, 1))
    ratings = df['rating'].values

    #model.fit([user_ids, movie_ids], ratings, epochs=50, batch_size=32, validation_split=0.2)
    history = model.fit([user_ids, movie_ids], ratings, epochs=200, batch_size=32, validation_split=0.2)


   
    matplotlib.rcParams['figure.dpi'] = 150
    #plot1(history)
    #plot2(history)
    return model

def train_surprise_model(data):
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    return algo

def get_tensorflow_recommendations(ncf_model, df, userId):
    user_unrated_movies = df[~df['movie_id'].isin(df[df['user_id'] == userId]['movie_id'])]
    user_ids = np.array([userId] * len(user_unrated_movies)).reshape((-1, 1))
    movie_ids = user_unrated_movies['movie_id'].values.reshape((-1, 1))
    predicted_ratings = ncf_model.predict([user_ids, movie_ids]).flatten()
    top_n_recommendations = user_unrated_movies.iloc[np.argsort(-predicted_ratings)[:NUM_RECOMMENDATIONS]]
    return top_n_recommendations['movie_id'].tolist()

def get_surprise_recommendations(algo, df, userId, num_recommendations=10):
    user_unrated_movies = df[~df['movie_id'].isin(df[df['user_id'] == userId]['movie_id'])]
    predictions = [algo.predict(userId, movie_id).est for movie_id in user_unrated_movies['movie_id'].unique()]
    top_n_recommendations = sorted(zip(user_unrated_movies['movie_id'].unique(), predictions), key=lambda x: x[1], reverse=True)[:num_recommendations]
    return [int(movie_id) for movie_id, _ in top_n_recommendations]

@app.route('/api/<userId>', methods=['GET'])
def api(userId):
    mydb = get_connection()

    try:
        mycursor = mydb.cursor()
        sql = "SELECT user_id, movie_id, rating FROM evaluation"
        mycursor.execute(sql)

        raw_data = mycursor.fetchall()
        df = pd.DataFrame(raw_data, columns=['user_id', 'movie_id', 'rating'])

        user_ratings = df[df['user_id'] == int(userId)]
        if len(user_ratings) == 0:
            fallback_recommendations = get_fallback_recommendations(df)
            return jsonify({'message': 'Cold start for new user', 'fallback_recommendations': fallback_recommendations})

        reader = Reader(rating_scale=(1, 5))
        surprise_data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)

        tensorflow_model = train_neural_collaborative_filtering(df)
        surprise_model = train_surprise_model(surprise_data)

        tf_recommendations = get_tensorflow_recommendations(tensorflow_model, df, int(userId))
        surprise_recommendations = get_surprise_recommendations(surprise_model, df, int(userId))

        return jsonify({'message': 'Recommendations', 'tensorflow_recommendations': tf_recommendations, 'surprise_recommendations': surprise_recommendations})

    except mysql.connector.Error as e:
        return jsonify({'error': str(e)})
    finally:
        if mycursor:
            mycursor.close()
        if mydb.is_connected():
            mydb.close()

if __name__ == '__main__':
    app.run(debug=True)



