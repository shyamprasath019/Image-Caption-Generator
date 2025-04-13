import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Step 1: Load the CSV
df = pd.read_csv('captions.txt')

# Step 2: Combine all captions into a list
captions = df['caption'].tolist()

# Step 3: Create and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

# Step 4: Save the tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("✅ Tokenizer regenerated and saved as tokenizer.pkl")
# import pickle

# with open("tokenizer.pkl", "wb") as f:
#     pickle.dump(tokenizer, f)
