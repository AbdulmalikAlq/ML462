from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

BASE = Path(r'c:\Users\Admin\Downloads\LM_462')
OUTPUT_ROOT = BASE / 'solutions' / 'outputs'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def write_text(path, content):
    path.write_text(content, encoding='utf-8')


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# Activity 1
iris_path = BASE / 'Activity 1 & 6 & 10 (Dataset)' / 'iris.csv'
activity1_dir = OUTPUT_ROOT / 'Activity1'
activity1_dir.mkdir(parents=True, exist_ok=True)

scipy = __import__('scipy').__version__
numpy_version = np.__version__
matplotlib_version = plt.matplotlib.__version__
pandas_version = pd.__version__

mylist = [[1, 2, 3], [3, 4, 5]]
myarray = np.array(mylist)
line_x = np.array([1, 2, 3])
line_y = np.array([2, 4, 6])

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(line_x, line_y, marker='o')
ax1.set_title('Line plot')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.scatter(line_x, line_y, color='tab:red')
ax2.set_title('Scatter plot')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
save_plot(fig1, activity1_dir / 'plots.png')

iris_df = pd.read_csv(iris_path)
activity1_text = f"""Activity 1 Summary
=================
Library versions:
- scipy: {scipy}
- numpy: {numpy_version}
- matplotlib: {matplotlib_version}
- pandas: {pandas_version}

Numpy array:
{myarray}
Shape: {myarray.shape}
First row: {myarray[0]}
Last row: {myarray[-1]}
Specific row and col: {myarray[0, 2]}
Whole column: {myarray[:, 2]}

Iris dataset preview:
{iris_df.head(3).to_string(index=False)}

Dataset shape: {iris_df.shape}
Dataset dtypes:
{iris_df.dtypes.to_string()}
"""
write_text(activity1_dir / 'activity1_output.txt', activity1_text)

# Activity 2
foodtruck_path = BASE / '‏‏Activity 2 & 3 (Dataset)' / 'foodtruck.csv'
activity2_dir = OUTPUT_ROOT / 'Activity2'
activity2_dir.mkdir(parents=True, exist_ok=True)

foodtruck_df = pd.read_csv(foodtruck_path)
X = foodtruck_df['Population'].to_numpy()
y = foodtruck_df['Profit'].to_numpy()

# Simple linear regression closed form
x_mean = X.mean()
y_mean = y.mean()
num = np.sum((X - x_mean) * (y - y_mean))
denom = np.sum((X - x_mean) ** 2)
w_opt = num / denom
b_opt = y_mean - w_opt * x_mean
preds = w_opt * X + b_opt
ss_res = np.sum((y - preds) ** 2)
ss_total = np.sum((y - y_mean) ** 2)
r2 = 1 - ss_res / ss_total

fig2, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X, y, alpha=0.7, label='Data')
ax.plot(np.sort(X), w_opt * np.sort(X) + b_opt, color='tab:red', label='Best-fit line')
ax.set_title('Food truck linear regression')
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.legend()
save_plot(fig2, activity2_dir / 'regression_fit.png')

activity2_text = f"""Activity 2 Summary
=================
Dataset shape: {foodtruck_df.shape}
Model parameters:
- w_opt: {w_opt:.6f}
- b_opt: {b_opt:.6f}
- SS_res: {ss_res:.6f}
- SS_total: {ss_total:.6f}
- R2: {r2:.6f}

Sample predictions:
{pd.DataFrame({'Population': X[:5], 'Actual': y[:5], 'Predicted': preds[:5]}).to_string(index=False)}
"""
write_text(activity2_dir / 'activity2_output.txt', activity2_text)

# Activity 3
activity3_dir = OUTPUT_ROOT / 'Activity3'
activity3_dir.mkdir(parents=True, exist_ok=True)


def gradient_descent(x, y, learning_rate, iterations):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = 0.0
    b = 0.0
    history = []
    for _ in range(iterations):
        preds = w * x + b
        error = preds - y
        w_grad = (2.0 / len(x)) * np.dot(error, x)
        b_grad = (2.0 / len(x)) * np.sum(error)
        w -= learning_rate * w_grad
        b -= learning_rate * b_grad
        sse = np.sum((y - (w * x + b)) ** 2)
        history.append((w, b, sse))
    return w, b, history

lr_results = []
for lr in [0.01, 0.001]:
    w, b, history = gradient_descent(X, y, lr, 1500)
    preds = w * X + b
    sse = np.sum((y - preds) ** 2)
    lr_results.append((lr, w, b, sse, history))

fig3, axes = plt.subplots(2, 2, figsize=(12, 8))
for idx, (lr, w, b, sse, history) in enumerate(lr_results):
    ws = [h[0] for h in history]
    bs = [h[1] for h in history]
    sses = [h[2] for h in history]
    row, col = divmod(idx, 2)
    axes[row, col].plot(range(len(history)), sses, label=f'lr={lr}')
    axes[row, col].set_title(f'SSE vs iteration (lr={lr})')
    axes[row, col].set_xlabel('Iteration')
    axes[row, col].set_ylabel('SSE')
    axes[row, col].legend()
    axes[row, col].grid(alpha=0.3)

# Additional plots for parameter traces
fig4, axes2 = plt.subplots(1, 2, figsize=(10, 4))
for lr, w, b, sse, history in lr_results:
    ws = [h[0] for h in history]
    bs = [h[1] for h in history]
    sses = [h[2] for h in history]
    axes2[0].plot(range(len(history)), ws, label=f'lr={lr}')
    axes2[1].plot(range(len(history)), bs, label=f'lr={lr}')
axes2[0].set_title('w vs iteration')
axes2[0].set_xlabel('Iteration')
axes2[0].set_ylabel('w')
axes2[1].set_title('b vs iteration')
axes2[1].set_xlabel('Iteration')
axes2[1].set_ylabel('b')
axes2[0].legend()
axes2[1].legend()
save_plot(fig4, activity3_dir / 'parameter_traces.png')
save_plot(fig3, activity3_dir / 'sse_iterations.png')

best_lr = sorted(lr_results, key=lambda x: x[3])[0]
activity3_text = f"""Activity 3 Summary
=================
Gradient descent results:
- Best learning rate: {best_lr[0]:.4f}
- Final w: {best_lr[1]:.6f}
- Final b: {best_lr[2]:.6f}
- Final SSE: {best_lr[3]:.6f}

Other learning rates:
{chr(10).join([f'lr={lr:.4f}: w={w:.6f}, b={b:.6f}, SSE={sse:.6f}' for lr, w, b, sse, _ in lr_results])}
"""
write_text(activity3_dir / 'activity3_output.txt', activity3_text)

# Activity 4
breast_path = BASE / '‏‏Activity 4 (Dataset)' / 'Breast_cancer.csv'
activity4_dir = OUTPUT_ROOT / 'Activity4'
activity4_dir.mkdir(parents=True, exist_ok=True)

breast_df = pd.read_csv(breast_path)
X_breast = pd.get_dummies(breast_df.drop(columns=['class']))
y_breast = breast_df['class'].map({'recurrence-events': 1, 'false-recurrence-events': 0})

X_train, X_test, y_train, y_test = train_test_split(X_breast, y_breast, test_size=0.25, random_state=42, stratify=y_breast)

best_score = -1
best_k = None
for k in range(1, 16):
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_breast, y_breast, cv=5)
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_k = k

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
pred = knn_model.predict(X_test)
acc = accuracy_score(y_test, pred)
misclassification_error = 1 - acc

activity4_text = f"""Activity 4 Summary
=================
Dataset shape: {breast_df.shape}
Training samples: {len(X_train)}
Test samples: {len(X_test)}
Best cross-validated k: {best_k}
Best CV accuracy: {best_score:.4f}
Test accuracy: {acc:.4f}
Misclassification error: {misclassification_error:.4f}

Confusion matrix:
{confusion_matrix(y_test, pred)}

Classification report:
{classification_report(y_test, pred)}
"""
write_text(activity4_dir / 'activity4_output.txt', activity4_text)

# Activity 5
pima_path = BASE / '‏‏Activity 5 & 8 (Dataset)' / 'pima_indian_diabeties.csv'
activity5_dir = OUTPUT_ROOT / 'Activity5'
activity5_dir.mkdir(parents=True, exist_ok=True)

pima_df = pd.read_csv(pima_path)
X_pima = pima_df.drop(columns=['Outcome'])
y_pima = pima_df['Outcome'].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pima)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_scaled, y_pima, test_size=0.2, random_state=42, stratify=y_pima)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sgd_logistic_regression(X, y, learning_rate=0.01, epochs=300):
    X_aug = np.column_stack([np.ones(len(X)), X])
    weights = np.zeros(X_aug.shape[1])
    intercepts = []
    costs = []
    for epoch in range(epochs):
        for i in range(len(X_aug)):
            xi = X_aug[i]
            yi = y[i]
            prob = sigmoid(np.dot(xi, weights))
            grad = (prob - yi) * xi
            weights -= learning_rate * grad
        probs = sigmoid(X_aug @ weights)
        cost = -np.mean(y * np.log(probs + 1e-12) + (1 - y) * np.log(1 - probs + 1e-12))
        intercepts.append(weights[0])
        costs.append(cost)
    return weights, intercepts, costs

weights, intercepts, costs = sgd_logistic_regression(X_train_p, y_train_p)

X_test_aug = np.column_stack([np.ones(len(X_test_p)), X_test_p])
preds_p = (sigmoid(X_test_aug @ weights) >= 0.5).astype(int)

fig5, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(range(len(intercepts)), intercepts)
axes[0].set_title('Intercept vs epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Intercept')
axes[1].plot(range(len(costs)), costs)
axes[1].set_title('Cost vs epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Cost')
save_plot(fig5, activity5_dir / 'logistic_training.png')

activity5_text = f"""Activity 5 Summary
=================
Dataset shape: {pima_df.shape}
Normalized features using StandardScaler.
Training samples: {len(X_train_p)}
Test samples: {len(X_test_p)}

Logistic regression coefficients:
{pd.Series(weights[1:], index=X_pima.columns).to_string()}

Accuracy on test data: {accuracy_score(y_test_p, preds_p):.4f}

Confusion matrix:
{confusion_matrix(y_test_p, preds_p)}

Classification report:
{classification_report(y_test_p, preds_p)}
"""
write_text(activity5_dir / 'activity5_output.txt', activity5_text)

# Activity 6
activity6_dir = OUTPUT_ROOT / 'Activity6'
activity6_dir.mkdir(parents=True, exist_ok=True)

iris = load_iris()
X_iris = iris.data
y_iris = iris.target
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_iris_scaled, y_iris, test_size=0.2, random_state=42, stratify=y_iris)

knn_model_iris = KNeighborsClassifier(n_neighbors=5)
logreg_model = LogisticRegression(max_iter=500, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn_scores = cross_val_score(knn_model_iris, X_iris_scaled, y_iris, cv=cv)
logreg_scores = cross_val_score(logreg_model, X_iris_scaled, y_iris, cv=cv)

knn_mean = knn_scores.mean()
logreg_mean = logreg_scores.mean()
selected_model = 'KNN' if knn_mean >= logreg_mean else 'Logistic Regression'

activity6_text = f"""Activity 6 Summary
=================
5-fold cross-validation results:
- KNN mean score: {knn_mean:.4f}
- Logistic regression mean score: {logreg_mean:.4f}
Selected model: {selected_model}
"""
write_text(activity6_dir / 'activity6_output.txt', activity6_text)

# Activity 7
golf_path = BASE / '‏‏Activity 7 (Dataset)' / 'golf-dataset.csv'
activity7_dir = OUTPUT_ROOT / 'Activity7'
activity7_dir.mkdir(parents=True, exist_ok=True)

Golf = pd.read_csv(golf_path)
X_golf = pd.get_dummies(Golf.drop(columns=['Play Golf']))
y_golf = Golf['Play Golf'].map({'Yes': 1, 'No': 0}).to_numpy()
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_golf, y_golf, test_size=0.3, random_state=42, stratify=y_golf)

nb = GaussianNB()
nb.fit(X_train_g, y_train_g)
pred_g = nb.predict(X_test_g)

# Conditional probability summary
class_probs = nb.class_prior_
conditional_summary = []
for cls_idx, cls_name in enumerate(nb.classes_):
    conditional_summary.append({
        'class': int(cls_name),
        'prior_probability': float(class_probs[cls_idx]),
        'feature_means': nb.theta_[cls_idx].tolist(),
        'feature_variances': nb.var_[cls_idx].tolist(),
    })

activity7_text = f"""Activity 7 Summary
=================
Dataset shape: {Golf.shape}
Training samples: {len(X_train_g)}
Test samples: {len(X_test_g)}

Accuracy: {accuracy_score(y_test_g, pred_g):.4f}

Conditional probability summary:
{json.dumps(conditional_summary, indent=2)}
"""
write_text(activity7_dir / 'activity7_output.txt', activity7_text)

# Activity 8
activity8_dir = OUTPUT_ROOT / 'Activity8'
activity8_dir.mkdir(parents=True, exist_ok=True)

pima_df = pd.read_csv(pima_path)
X_tree = pima_df.drop(columns=['Outcome'])
y_tree = pima_df['Outcome'].to_numpy()
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_tree, y_tree, test_size=0.25, random_state=42, stratify=y_tree)

results = []
for criterion in ['gini', 'entropy']:
    for depth in [3, 5, None]:
        model = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
        model.fit(X_train_t, y_train_t)
        pred_t = model.predict(X_test_t)
        acc_t = accuracy_score(y_test_t, pred_t)
        results.append((criterion, depth, acc_t))

best_tree = max(results, key=lambda item: item[2])
criterion, depth, acc_t = best_tree
best_model = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
best_model.fit(X_train_t, y_train_t)

fig6 = plt.figure(figsize=(8, 6))
from sklearn import tree
tree.plot_tree(best_model, filled=True, feature_names=X_tree.columns.tolist(), class_names=['0', '1'])
save_plot(fig6, activity8_dir / 'decision_tree.png')

activity8_text = f"""Activity 8 Summary
=================
Dataset shape: {pima_df.shape}
Test accuracy by tree configuration:
{chr(10).join([f'criterion={criterion}, max_depth={depth}, accuracy={acc:.4f}' for criterion, depth, acc in results])}

Best configuration: criterion={criterion}, max_depth={depth}, accuracy={acc_t:.4f}
"""
write_text(activity8_dir / 'activity8_output.txt', activity8_text)

# Activity 9
activity9_dir = OUTPUT_ROOT / 'Activity9'
activity9_dir.mkdir(parents=True, exist_ok=True)

X_blob, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=42)
scaler_9 = StandardScaler()
X_scaled_9 = scaler_9.fit_transform(X_blob)

kmeans_raw = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_labels = kmeans_raw.fit_predict(X_blob)
raw_score = silhouette_score(X_blob, raw_labels)

kmeans_scaled = KMeans(n_clusters=3, random_state=42, n_init=10)
scaled_labels = kmeans_scaled.fit_predict(X_scaled_9)
scaled_score = silhouette_score(X_scaled_9, scaled_labels)

fig7, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=raw_labels, cmap='viridis', s=40)
axes[0].set_title(f'K-means on raw data (silhouette={raw_score:.3f})')
axes[1].scatter(X_scaled_9[:, 0], X_scaled_9[:, 1], c=scaled_labels, cmap='viridis', s=40)
axes[1].set_title(f'K-means on normalized data (silhouette={scaled_score:.3f})')
save_plot(fig7, activity9_dir / 'kmeans_clusters.png')

comment = 'Normalization helps when features have different scales because it gives each feature equal influence during clustering.'
activity9_text = f"""Activity 9 Summary
=================
Generated dataset with 3 clusters.
Silhouette score on raw data: {raw_score:.4f}
Silhouette score on normalized data: {scaled_score:.4f}

Comment: {comment}
"""
write_text(activity9_dir / 'activity9_output.txt', activity9_text)

# Activity 10
activity10_dir = OUTPUT_ROOT / 'Activity10'
activity10_dir.mkdir(parents=True, exist_ok=True)

iris_data = load_iris()
X_svm = iris_data.data
y_svm = iris_data.target
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_svm, y_svm, test_size=0.2, random_state=42, stratify=y_svm)

linear_model = SVC(kernel='linear', C=1.0)
linear_model.fit(X_train_s, y_train_s)
linear_pred = linear_model.predict(X_test_s)
linear_acc = accuracy_score(y_test_s, linear_pred)

rbf_results = []
for C in [0.1, 1.0, 10.0]:
    for gamma in [0.01, 0.1, 1.0]:
        model = SVC(kernel='rbf', C=C, gamma=gamma)
        model.fit(X_train_s, y_train_s)
        pred = model.predict(X_test_s)
        acc = accuracy_score(y_test_s, pred)
        rbf_results.append((C, gamma, acc))

best_rbf = max(rbf_results, key=lambda item: item[2])
C_best, gamma_best, best_acc = best_rbf
rbf_model = SVC(kernel='rbf', C=C_best, gamma=gamma_best)
rbf_model.fit(X_train_s, y_train_s)
rbf_pred = rbf_model.predict(X_test_s)

activity10_text = f"""Activity 10 Summary
=================
Linear SVM accuracy: {linear_acc:.4f}
RBF SVM best configuration: C={C_best}, gamma={gamma_best}, accuracy={best_acc:.4f}

Other tested RBF configurations:
{chr(10).join([f'C={C}, gamma={gamma}, accuracy={acc:.4f}' for C, gamma, acc in rbf_results])}
"""
write_text(activity10_dir / 'activity10_output.txt', activity10_text)

# Final report
report_lines = []
for number, title in enumerate([
    'Activity 1: Scikit-learn basics',
    'Activity 2: Simple linear regression',
    'Activity 3: Gradient descent for linear regression',
    'Activity 4: KNN for breast cancer detection',
    'Activity 5: Logistic regression for diabetes prediction',
    'Activity 6: KNN vs logistic regression on iris',
    'Activity 7: Naive Bayes on golf dataset',
    'Activity 8: Decision tree on diabetes dataset',
    'Activity 9: K-means clustering',
    'Activity 10: SVM on iris',
], start=1):
    report_lines.append(f"{number}. {title} -> completed")

report_text = "\n".join(report_lines)
write_text(BASE / 'solutions' / 'completion_report.md', report_text)
print('All activities completed and outputs written to', OUTPUT_ROOT)
