import nbformat as nbf

nb = nbf.v4.new_notebook()

def add_md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def add_code(code):
    nb.cells.append(nbf.v4.new_code_cell(code))

add_md("# Machine Learning - Assignment 3\n\n**IDs**: 212958755, 212608368")

add_md("## Setup and Imports")
add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)""")

add_md("## Data Loading and Splitting")
add_code("""beans_df = pd.read_csv('Dry_Beans_Filtered_Assignment3.csv')
fashion_df = pd.read_csv('Fashion_MNIST_Filtered_Assignment3.csv')

# Split Dry Beans
X_beans = beans_df.drop('Class', axis=1)
y_beans = beans_df['Class']
X_b_train, X_b_temp, y_b_train, y_b_temp = train_test_split(X_beans, y_beans, test_size=0.2, random_state=42, stratify=y_beans)
X_b_val, X_b_test, y_b_val, y_b_test = train_test_split(X_b_temp, y_b_temp, test_size=0.5, random_state=42, stratify=y_b_temp)

# Split Fashion MNIST
X_fash = fashion_df.drop('label', axis=1)
y_fash = fashion_df['label']
X_f_train, X_f_temp, y_f_train, y_f_temp = train_test_split(X_fash, y_fash, test_size=0.2, random_state=42, stratify=y_fash)
X_f_val, X_f_test, y_f_val, y_f_test = train_test_split(X_f_temp, y_f_temp, test_size=0.5, random_state=42, stratify=y_f_temp)

print("Dry Beans Train/Val/Test sizes:", X_b_train.shape, X_b_val.shape, X_b_test.shape)
print("Fashion MNIST Train/Val/Test sizes:", X_f_train.shape, X_f_val.shape, X_f_test.shape)""")

add_md("## Part 1 - Dry Bean Dataset")
add_md("### Section A - Data Exploration & Visualization")
add_code("""# 1. Class Distribution
plt.figure(figsize=(8,5))
sns.countplot(data=beans_df, x='Class')
plt.title('Distribution of Bean Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# 2. Area vs Perimeter
plt.figure(figsize=(8,5))
sns.scatterplot(data=beans_df, x='Area', y='Perimeter', hue='Class', alpha=0.7)
plt.title('Area vs Perimeter by Class')
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(X_b_train.corr(), cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# 4. Boxplot of MajorAxisLength
plt.figure(figsize=(8,5))
sns.boxplot(data=beans_df, x='Class', y='MajorAxisLength')
plt.title('Major Axis Length by Class')
plt.show()

# 5. Pairplot of selected features
sns.pairplot(beans_df[['roundness', 'Compactness', 'ShapeFactor1', 'Class']], hue='Class')
plt.suptitle('Pairplot of Shape Features', y=1.02)
plt.show()""")

add_md("### Section B - Data Pre-processing")
add_code("""def add_features(X):
    X_new = X.copy()
    X_new['AspectR'] = X_new['MajorAxisLength'] / (X_new['MinorAxisLength'] + 1e-8)
    X_new['Rectangularity'] = X_new['Area'] / (X_new['MajorAxisLength'] * X_new['MinorAxisLength'] + 1e-8)
    X_new['Volume_Est'] = (4/3) * np.pi * (X_new['MajorAxisLength']/2) * ((X_new['MinorAxisLength']/2)**2)
    return X_new

feature_eng = FunctionTransformer(add_features)
preprocessor = Pipeline([
    ('feature_eng', feature_eng),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_b_train_prep = preprocessor.fit_transform(X_b_train)
X_b_val_prep = preprocessor.transform(X_b_val)
X_b_test_prep = preprocessor.transform(X_b_test)""")

add_md("### Section C - Classification and Clustering")
add_code("""split_index = [-1]*len(X_b_train_prep) + [0]*len(X_b_val_prep)
X_b_train_val = np.vstack((X_b_train_prep, X_b_val_prep))
y_b_train_val = np.concatenate((y_b_train, y_b_val))
pds = PredefinedSplit(test_fold=split_index)

models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), {'C': [0.1, 1, 10]}),
    'Random Forest': (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100], 'max_depth': [None, 10]}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]})
}

best_models_b = {}
scores_b = {}

for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=pds, scoring='accuracy', n_jobs=-1)
    grid.fit(X_b_train_val, y_b_train_val)
    best_models_b[name] = grid.best_estimator_
    preds = grid.predict(X_b_test_prep)
    scores_b[name] = accuracy_score(y_b_test, preds)
    print(f"{name} best params: {grid.best_params_}, Test Accuracy: {scores_b[name]:.4f}")

kmeans = KMeans(n_clusters=len(np.unique(y_b_train)), random_state=42)
agglo = AgglomerativeClustering(n_clusters=len(np.unique(y_b_train)))

k_labels = kmeans.fit_predict(X_b_train_prep)
a_labels = agglo.fit_predict(X_b_train_prep)

print("KMeans Silhouette:", silhouette_score(X_b_train_prep, k_labels))
print("Agglomerative Silhouette:", silhouette_score(X_b_train_prep, a_labels))""")

add_md("### Section D - PCA")
add_code("""pca = PCA(n_components=0.80, random_state=42)
X_b_train_pca = pca.fit_transform(X_b_train_prep)
X_b_val_pca = pca.transform(X_b_val_prep)
X_b_test_pca = pca.transform(X_b_test_prep)

X_b_train_val_pca = np.vstack((X_b_train_pca, X_b_val_pca))

scores_b_pca = {}
for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=pds, scoring='accuracy', n_jobs=-1)
    grid.fit(X_b_train_val_pca, y_b_train_val)
    preds = grid.predict(X_b_test_pca)
    scores_b_pca[name] = accuracy_score(y_b_test, preds)
    print(f"{name} (PCA) Test Accuracy: {scores_b_pca[name]:.4f}")""")

add_md("## Part 2 - Fashion-MNIST")
add_code("""plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_f_train.iloc[i].values.reshape(28,28), cmap='gray')
    plt.title(f"Label: {y_f_train.iloc[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

avg_pixels = X_f_train.mean(axis=0).values.reshape(28,28)
plt.figure(figsize=(6,5))
sns.heatmap(avg_pixels, cmap='viridis')
plt.title('Average Pixel Intensity Heatmap')
plt.show()""")

add_code("""scaler_f = StandardScaler()
X_f_train_s = scaler_f.fit_transform(X_f_train)
X_f_val_s = scaler_f.transform(X_f_val)
X_f_test_s = scaler_f.transform(X_f_test)

split_index_f = [-1]*len(X_f_train_s) + [0]*len(X_f_val_s)
X_f_train_val_s = np.vstack((X_f_train_s, X_f_val_s))
y_f_train_val = np.concatenate((y_f_train, y_f_val))
pds_f = PredefinedSplit(test_fold=split_index_f)

f_models = {
    'LogReg': (LogisticRegression(max_iter=1000, random_state=42), {'C': [0.1, 1]}),
    'RF': (RandomForestClassifier(random_state=42), {'n_estimators': [50], 'max_depth': [10]})
}

scores_f = {}
for name, (model, params) in f_models.items():
    grid = GridSearchCV(model, params, cv=pds_f, scoring='accuracy', n_jobs=-1)
    grid.fit(X_f_train_val_s, y_f_train_val)
    preds = grid.predict(X_f_test_s)
    scores_f[name] = accuracy_score(y_f_test, preds)
    print(f"{name} | Test Acc: {scores_f[name]:.4f}")""")

add_code("""pca_f = PCA(n_components=0.80, random_state=42)
X_f_train_pca = pca_f.fit_transform(X_f_train_s)
X_f_val_pca = pca_f.transform(X_f_val_s)
X_f_test_pca = pca_f.transform(X_f_test_s)

loadings_sq = pca_f.components_ ** 2
importance = np.sum(pca_f.explained_variance_ratio_[:, np.newaxis] * loadings_sq, axis=0)
norm_importance = importance / np.sum(importance)

top_5_idx = np.argsort(norm_importance)[-5:]
bottom_5_idx = np.argsort(norm_importance)[:5]

drop_percent = 0.10
sorted_idx = np.argsort(norm_importance)
cum_imp = np.cumsum(norm_importance[sorted_idx])
drop_indices = sorted_idx[cum_imp <= drop_percent]

print(f"Dropping {len(drop_indices)} pixels.")

X_f_train_drop = np.delete(X_f_train_s, drop_indices, axis=1)
X_f_val_drop = np.delete(X_f_val_s, drop_indices, axis=1)
X_f_test_drop = np.delete(X_f_test_s, drop_indices, axis=1)""")

add_code("""_, X_tsne, _, y_tsne = train_test_split(X_f_train_pca, y_f_train, test_size=0.1, random_state=42, stratify=y_f_train)
tsne = TSNE(n_components=2, random_state=42)
X_tsne_2d = tsne.fit_transform(X_tsne)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=y_tsne, cmap='tab10', alpha=0.6)
plt.legend(handles=scatter.legend_elements()[0], labels=list(np.unique(y_tsne)))
plt.title('t-SNE of Fashion MNIST (Sample, 2D)')
plt.show()""")

add_md("## Part 3 - LLM/Agent Use Appendix\n(Detailed in the implementation plan and markdown artifact)")

with open('ML_HW3_212958755_212608368.ipynb', 'w') as f:
    nbf.write(nb, f)
