import pickle
import pandas as pd

# Load your models locally
movies = pd.DataFrame(pickle.load(open('movie_dict.pkl', 'rb')))
algo = pickle.load(open('collaborative_brain.pkl', 'rb'))

# Pre-calculate ratings for User 1 (or whichever user you are targeting)
user_id = 1
predictions = {}

for m_id in movies['movieId'].values:
    predictions[m_id] = algo.predict(user_id, m_id).est

# Save this simple dictionary
pickle.dump(predictions, open('user_predictions.pkl', 'wb'))
print("Successfully created user_predictions.pkl!")