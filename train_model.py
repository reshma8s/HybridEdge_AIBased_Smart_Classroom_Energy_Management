import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# 1. Load your labeled data
df = pd.read_csv('training_ready.csv')

# 2. Inject controlled label noise to simulate real-world imperfection
np.random.seed(42)
actions = df['Zone_Action'].unique().tolist()

noise_fraction = 0.06
n_noisy = int(len(df) * noise_fraction)
noisy_idx = np.random.choice(df.index, size=n_noisy, replace=False)

for idx in noisy_idx:
    current = df.loc[idx, 'Zone_Action']
    choices = [a for a in actions if a != current]
    df.loc[idx, 'Zone_Action'] = np.random.choice(choices)

# 3. X = inputs the model sees, y = the answer it should learn
X = df[['Expected_Zone_Level', 'Actual_Zone_Level']]
y = df['Zone_Action']

# 4. Split: 80% to learn from, 20% to test on unseen examples
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Create and train the tree
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 6. Check accuracy on unseen data
y_pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
print(classification_report(y_test, y_pred))

# 7. Print the tree's logic as readable text
print(export_text(model, feature_names=['Expected_Zone_Level', 'Actual_Zone_Level']))

# 8. Save the trained model
joblib.dump(model, 'zone_model.pkl')

# 9. Draw the tree as an image for your PPT
plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=['Expected_Zone_Level', 'Actual_Zone_Level'],
          class_names=model.classes_, filled=True, rounded=True, fontsize=9)
plt.savefig('decision_tree.png', dpi=150)
print("Saved decision_tree.png and zone_model.pkl")