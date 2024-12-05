import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Read the dataset
df = pd.read_csv("dataset.csv")

# Define the feature (X) and target (y)
X = df.drop("Weather Type", axis=1)
y = df["Weather Type"]

# Identify categorical columns automatically
categorical_columns = X.select_dtypes(include=['object', 'category']).columns

# Apply LabelEncoder to each categorical column
label_encoder = LabelEncoder()

for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Standardize the features
preprocessor = StandardScaler()
X_train_scaled = preprocessor.fit_transform(X_train)

# Save the model, preprocessor, and label encoder
joblib.dump(model, 'model/train.pkl')
joblib.dump(preprocessor, 'model/preprocess.pkl')
joblib.dump(label_encoder, 'model/label_encoder.pkl')  # Save the LabelEncoder for reverse encoding

print("Model and preprocessor have been saved successfully.")
