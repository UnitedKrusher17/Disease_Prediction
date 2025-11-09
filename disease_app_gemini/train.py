import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

FILE_PATH = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
TARGET_COLUMN = "diseases"
MODEL_PATH = "rf_model_local.joblib"
LE_PATH = "label_encoder_local.joblib"
RANDOM_STATE = 42

df = pd.read_csv(FILE_PATH)
df = df.fillna(0)

y = df[TARGET_COLUMN]
rare_classes = y.value_counts()[y.value_counts() < 2].index
df = df[~y.isin(rare_classes)]
y = df[TARGET_COLUMN]

X = df.drop(columns=[TARGET_COLUMN])

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded)

rf = RandomForestClassifier(n_estimators=250, max_depth=25, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE, verbose=1)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"Test set accuracy: {accuracy_score(y_test, y_pred):.4f}")
report = classification_report(y_test, y_pred, labels=range(len(le.classes_)), target_names=le.classes_)
print(report)

joblib.dump(rf, MODEL_PATH)
joblib.dump(le, LE_PATH)
print("Model and LabelEncoder saved locally.")
