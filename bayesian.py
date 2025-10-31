import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import os

# Load dataset
columns = ['age', 'workclass', 'fnlwgt', 'education', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'native-country', 'income']

train_data = pd.read_csv('dataset/Bayesian_Dataset_train.csv', header=None, names=columns)
test_data = pd.read_csv('dataset/Bayesian_Dataset_test.csv', header=None, names=columns)

# Encode categorical columns
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'native-country']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

# Encode labels
y_train = LabelEncoder().fit_transform(train_data['income'])
y_test = LabelEncoder().fit_transform(test_data['income'])

X_train = train_data.drop(columns=['income'])
X_test = test_data.drop(columns=['income'])

alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]
results = []

for a in alphas:
    model = CategoricalNB(alpha=a)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    results.append([a, report['1']['precision'], report['1']['recall'], report['1']['f1-score']])

# Results table
results_df = pd.DataFrame(results, columns=['alpha', 'precision', 'recall', 'F1-score'])
print(results_df)

results_base_path = "results/bayesian/"
os.makedirs(results_base_path, exist_ok=True)

# Evaluation curve
plt.figure()
plt.plot(results_df['alpha'], results_df['F1-score'], marker='o')
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('F1-score')
plt.title('Effect of Alpha on F1-score')
plt.grid(True)
plt.savefig(f"{results_base_path}/alpha_effect.png")

# Best model visualization
best_alpha = results_df.loc[results_df['F1-score'].idxmax(), 'alpha']
best_model = CategoricalNB(alpha=best_alpha).fit(X_train, y_train)
best_pred = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Best Alpha = {best_alpha})')
plt.savefig(f"{results_base_path}/confusion_matrix.png")
plt.close()

# Save predictions
test_data['predicted_income'] = ['>50K' if p == 1 else '<=50K' for p in best_pred]
test_data.to_csv(f"{results_base_path}/Bayesian_Dataset_test_with_predictions.csv", index=False)

print("\n✅ Done! Outputs saved in 'results/' folder.")
