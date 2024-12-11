
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

dataset = pd.read_csv('data/credit_history.csv')

if "Credit_Score" in dataset.columns:
    # 매핑 정의
    credit_mix_mapping = {"Standard": 1, "Good": 2, "Poor": 0}

    # Credit_Mix 열 추가
    dataset["Credit_Mix"] = dataset["Credit_Score"].map(credit_mix_mapping)

    # 결과 확인
    print("Updated dataset with new 'Credit_Mix' column:")
    print(dataset[["Credit_Score", "Credit_Mix"]].head(20))

    # 변환된 데이터셋을 새로운 CSV 파일로 저장
    dataset.to_csv('data/credit_history.csv', index=False)
    print("Updated dataset has been saved to 'data/credit_history.csv'.")
else:
    print("The column 'Credit_Score' does not exist in the dataset.")


important_features = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Outstanding_Debt",
    "Credit_History_Age",
    "Amount_invested_monthly",
    "Monthly_Balance"
]

# Feature matrix와 Label
if all(feature in dataset.columns for feature in important_features) and "Credit_Score" in dataset.columns:

    X = dataset[important_features]
    y = dataset["Credit_Score"]

    # 결과 확인
    print("Feature Matrix (X):")
    print(X.head(20))

    print("\nLabels (y):")
    print(y.head(20))


else:
    missing_features = [feature for feature in important_features if feature not in dataset.columns]
    if "Credit_Score" not in dataset.columns:
        print("Target column 'Credit_Score' is missing in the dataset.")
    if missing_features:
        print("The following important features are missing in the dataset:")
        print(missing_features)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("\nModel training completed.")


svm_model = SVC(kernel="linear", random_state=42)

svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)


print("\nSupport Vector Machine Results:")
print(classification_report(y_test, svm_preds))
print(f"Accuracy: {accuracy_score(y_test, svm_preds):.2f}")



param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear"],
    "gamma": ["scale", "auto"]
}


grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy", verbose=2)
grid_search.fit(X_train, y_train)


print("\nBest Parameters:", grid_search.best_params_)


best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

print("\nTuned Linear SVM Results:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
