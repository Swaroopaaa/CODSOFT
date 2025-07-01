import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
file_path = r"C:\Users\swaro\OneDrive\Desktop\codsoft intership\task 1\archive\Genre Classification Dataset\train_data.txt"

with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()


data = [line.strip().split(" ::: ") for line in lines if len(line.strip().split(" ::: ")) == 4]
df = pd.DataFrame(data, columns=["ID", "Title", "Genre", "Plot"])


df.dropna(subset=["Plot", "Genre"], inplace=True)


X = df["Plot"]
y = df["Genre"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)

print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

def predict_genre(plot_text):
    text_tfidf = vectorizer.transform([plot_text])
    return model.predict(text_tfidf)[0]


print("\nYou can test your own plot summaries!")
while True:
    user_input = input("\nEnter a movie plot (or type 'exit' to quit):\n> ")
    if user_input.lower() == 'exit':
        print(" Exiting. Thank you!")
        break
    prediction = predict_genre(user_input)
    print(" Predicted Genre:", prediction)