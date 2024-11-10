import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pickle


def train(df_train, y_train, C=1.0):
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train.to_dict(orient="records"))
    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model, dv


data = "healthcare-dataset-stroke-data.csv"

df = pd.read_csv(data)
df = df.sample(frac=1).reset_index(drop=True)
df.columns = df.columns.str.lower().str.replace(" ", "_")
del df["id"]
df.fillna(df.bmi.mean(), inplace=True)
index_to_delete = df[df.gender == "Other"].index
df = df.drop(index_to_delete)

df_full_train, _ = train_test_split(df, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.stroke.values
del df_full_train["stroke"]

model, dv = train(df_train=df_full_train, y_train=y_full_train, C=10)

with open("model.bin", "wb") as f_out:
    pickle.dump((dv, model), f_out)
