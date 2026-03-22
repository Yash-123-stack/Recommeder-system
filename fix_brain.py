import pandas as pd
from surprise import SVD, Reader, Dataset
import pickle

# 1. Load your local ratings
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# 2. Setup the Surprise format
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# 3. Train the model locally (Ensures 100% compatibility)
print("🧠 Training your Collaborative Brain locally...")
algo = SVD()
algo.fit(trainset)

# 4. Overwrite the broken .pkl file
with open('collaborative_brain.pkl', 'wb') as f:
    pickle.dump(algo, f)

print("✅ Success! Your local 'collaborative_brain.pkl' is ready.")