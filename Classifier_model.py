from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# train test split:
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]
# train test split
X_train = temp_X_new[:X_train.shape[0]]
X_test = temp_X_new[X_train.shape[0]:]
LE = LabelEncoder()
y_train = LE.fit_transform(y_train)
y_test = LE.fit_transform(y_test)
# Split the dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Initialize the models
rf_classifier = RandomForestClassifier(random_state=42)


# Train the models
rf_classifier.fit(X_train, y_train)


# Predict on the validation set
y_val_pred_rf = rf_classifier.predict(X_val)


# Calculate metrics for each model
f1_score_rf = f1_score(y_val, y_val_pred_rf, average='weighted')
accuracy_rf = accuracy_score(y_val, y_val_pred_rf)

print("Random Forest Classifier - F1 Score:", f1_score_rf)
print("Random Forest Classifier - Accuracy:", accuracy_rf)

# Data
scores = [f1_score_rf, accuracy_rf]
labels = ['F1 Score (Weighted)', 'Accuracy']

# Create the plot
plt.bar(labels, scores)

# Add values on top of the bars
for i in range(len(scores)):
    plt.text(i, scores[i], f"{scores[i]:.2f}", ha='center', va='bottom')

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('F1 Score and Accuracy Comparison')
plt.ylim(0, 1.0)  # Set the y-axis limit to ensure both scores are visible

# Show the plot
plt.show()



