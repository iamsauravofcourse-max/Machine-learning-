# 📊 Complete Data Science EDA Roadmap

A structured, phase-by-phase reference for exploratory data analysis — from business understanding to model-readiness.

---

## Phase 0 — Business Understanding
*Understand the problem before touching the data.*

| Item | Question to Answer |
|---|---|
| Business Problem | What problem are we solving? |
| Business Objective | What business value should the solution provide? |
| ML Problem Type | Classification, Regression, Clustering, Forecasting, etc. |
| Target Variable | What are we predicting? |
| Prediction Granularity | Per customer, transaction, product, etc. |
| Success Metrics | How will success be measured? (Business + ML) |
| Baseline | What is the current performance to beat? |
| Stakeholders | Who will use the model or its predictions? |
| End Users | Who benefits from the solution? |
| Business Constraints | Time, cost, interpretability, legal issues |
| Technical Constraints | Memory, latency, compute, storage |
| Dataset Source | Where did the data come from? |
| Data Collection Process | How was the data collected? |
| Time Period | What time range does the data cover? |
| Domain Knowledge | Understand the business context |
| Assumptions | Initial hypotheses to verify during EDA |
| Risks & Limitations | Known challenges or limitations |
| Expected Deliverables | Reports, model, dashboard, API, etc. |
| Project Success Criteria | What defines a successful project? |

---

## Phase 1 — Basic Data Audit
*Understand the structure of the dataset.*

| Check | Purpose | Code |
|---|---|---|
| Shape | Rows & columns count | `df.shape` |
| Head / Tail / Sample | Quick inspection | `df.head()` / `df.tail()` / `df.sample(5, random_state=42)` |
| Columns | View feature names | `df.columns` |
| Data Types | Verify dtype of each feature | `df.dtypes` |
| Dataset Info | Columns, missing values, memory | `df.info()` |
| Memory Usage | Check RAM consumption | `df.memory_usage(deep=True)` / `df.memory_usage(deep=True).sum()/1024**2` |
| Missing Values | Identify & measure % missing | `df.isnull().sum()` / `(df.isnull().mean()*100).sort_values(ascending=False)` |
| Missing Value Pattern | Visualize missingness | `msno.matrix(df)` / `msno.bar(df)` / `msno.heatmap(df)` |
| Duplicate Rows | Detect repeated observations | `df.duplicated().sum()` / `df[df.duplicated()]` |
| Duplicate Index | Ensure unique row indices | `df.index.duplicated().sum()` |
| Unique Values | Count distinct values per feature | `df.nunique()` |
| Constant Features | Columns with one unique value | `df.nunique()[df.nunique()==1]` |
| Near-Constant Features | Columns dominated by one value | `for col in df.columns: print(df[col].value_counts(normalize=True).head(1))` |
| Cardinality Check | Number of unique categories | `df.nunique().sort_values()` |
| ID Detection | Columns uniquely identifying rows | `df.nunique()[df.nunique()==len(df)]` |
| Basic Statistics | Numeric & categorical summary | `df.describe().T` / `df.describe(include='object').T` |
| Target Inspection | Class balance of target | `df['TARGET'].value_counts()` / `df['TARGET'].value_counts(normalize=True)` |
| Empty Rows | Rows with all values missing | `df[df.isnull().all(axis=1)]` |
| Empty Columns | Columns with only missing values | `df.columns[df.isnull().all()]` |
| Index Inspection | Check if index is meaningful | `df.index` |
| Feature Classification | Separate features by type | `numeric_cols = df.select_dtypes(include='number').columns` etc. |
| Data Type Validation | Infer better dtypes | `df.convert_dtypes()` |
| Initial Sanity Check | Look for impossible values | `df.describe(include='all').T` |
| Date Range Check | Inspect datetime columns | `df.select_dtypes(include='datetime')` |
| Category Standardization | Detect inconsistent labels | `df['column_name'].unique()` |
| Infinity Values | Detect infinite values | `np.isinf(df.select_dtypes(include='number')).sum()` |
| Dataset Documentation | Read data dictionary | *(manual)* |

---

## Phase 2 — Data Quality Assessment
*Determine whether the data is clean and trustworthy.*

| Check | Purpose | Code |
|---|---|---|
| Missing Value Pattern | Where/how missing values occur | `df.isnull().sum()` / `msno.matrix(df)` |
| Missingness Mechanism | MCAR / MAR / MNAR | Statistical test / domain analysis |
| Data Completeness | Verify features contain enough data | `df.notnull().mean()*100` |
| Null Correlation | Are missing values related? | `df.isnull().corr()` |
| Incorrect Data Types | Wrongly stored dtypes | `df.dtypes` / `df.convert_dtypes()` |
| Mixed Data Types | Columns with multiple types | `df.applymap(type).nunique()` |
| Encoding Issues | Detect corrupted text | `df.select_dtypes('object').head()` |
| Special Character Check | Unexpected symbols | `df['column'].str.contains(r'[^a-zA-Z0-9 ]', regex=True)` |
| Impossible Values | Logically impossible values | `df.describe()` |
| Out-of-Range Values | Values outside valid limits | `df.describe()` |
| Placeholder Values | Detect placeholders | `df.replace(['?','NA','None','Unknown','-999'], np.nan)` |
| Zero vs Missing | Distinguish zeros from missing | `(df==0).sum()` |
| Invalid Categories | Unexpected category values | `df['column'].unique()` |
| Inconsistent Formatting | e.g. casing/spacing | `df['column'].str.lower().str.strip()` |
| Duplicate Categories | Duplicate spellings | `df['column'].value_counts()` |
| Whitespace Check | Leading/trailing spaces | `df['column'].str.strip()` |
| Unit Consistency | Verify measurement units | `df.describe()` |
| Date Consistency | Logical dates | `df.select_dtypes(include='datetime')` |
| Timestamp Validation | Duplicate/invalid timestamps | `df['date_column'].duplicated().sum()` |
| Duplicate Record Check | Duplicate rows | `df.duplicated().sum()` |
| Noise Detection | Noisy observations | `df.describe()` |
| Label Quality Check | Inspect target labels | `df['TARGET'].value_counts()` |
| Class Imbalance Check | Target distribution | `df['TARGET'].value_counts(normalize=True)` |
| Schema Validation | Verify expected schema | `df.info()` |
| Primary Key Validation | Ensure PK uniqueness | `df['ID'].is_unique` |
| Foreign Key Validation | Verify FK relationships | `pd.merge(df1, df2, how='left', indicator=True)` |
| Referential Integrity | Validate table relationships | `pd.merge(df1, df2, indicator=True)` |
| Cross-Field Validation | Related-column logic | Custom logical conditions |
| Business Rule Validation | Domain constraints | Custom business rules |
| Domain-Specific Validation | Domain-knowledge checks | Domain-specific checks |
| Data Leakage Check | Detect future information | `df.corr(numeric_only=True)` |
| Target Leakage Check | Features revealing the target | `df.corr(numeric_only=True)['TARGET']` |
| Train/Test Distribution | Compare distributions | `train.describe()` / `test.describe()` |
| Sampling Bias Check | Compare subgroup distributions | `train['col'].value_counts(normalize=True)` vs `test[...]` |
| Data Drift Check | Compare feature distributions | `train.describe()` / `test.describe()` |
| Data Freshness Check | Dataset recency | `df['date_column'].min()` / `.max()` |

---

## Phase 3 — Univariate Analysis
*Understand each feature individually.*

### Numeric Features
| Technique | Gain | Code |
|---|---|---|
| Histogram | Distribution, skewness, spread, outliers | `sns.histplot(df['col'], bins=30, kde=True)` |
| KDE Plot | Density shape, multiple peaks | `sns.kdeplot(df['col'], fill=True)` |
| Boxplot | Spread, quartiles, outliers | `sns.boxplot(x=df['col'])` |
| Violin Plot | Boxplot + density combined | `sns.violinplot(x=df['col'])` |
| ECDF | Cumulative distribution / percentiles | `sns.ecdfplot(df['col'])` |
| Summary Statistics | Central tendency & spread | `df['col'].describe()` |
| Quantiles | Tails and spread | `df['col'].quantile([.01,.05,.25,.5,.75,.95,.99])` |
| Mode | Most frequent value | `df['col'].mode()` |
| Coefficient of Variation | Relative variability | `df['col'].std()/df['col'].mean()` |
| Skewness | Distribution asymmetry | `df['col'].skew()` |
| Kurtosis | Tail heaviness | `df['col'].kurt()` |
| Normality Tests | Test for normal distribution | `stats.shapiro()` / `stats.anderson()` / `stats.kstest()` |
| Distribution Fitting | Best-fit theoretical distribution | `stats.norm.fit(df['col'])` |
| Outlier Detection | Extreme observations | `stats.zscore(df['col'])` / `sns.boxplot()` |
| Zero Inflation | Excess zeros | `(df['col']==0).mean()*100` |
| Distribution Shape | Normal/skewed/bimodal/multimodal | `sns.histplot(df['col'], kde=True)` |

### Categorical Features
| Technique | Gain | Code |
|---|---|---|
| Count Plot | Category frequencies | `sns.countplot(x='col', data=df)` |
| Value Counts | Frequency overview | `df['col'].value_counts(dropna=False)` |
| Frequency Table | Counts & percentages | `df['col'].value_counts(normalize=True)*100` |
| Rare Categories | Low-frequency categories | `df['col'].value_counts()` |
| Cardinality | Unique category count | `df['col'].nunique()` |

### Other
| Technique | Gain | Code |
|---|---|---|
| Datetime Distribution | Trends/seasonality | `df['date'].dt.year.value_counts()` etc. |
| Target Distribution | Class imbalance | `df['TARGET'].value_counts()` / `sns.countplot(x='TARGET', data=df)` |
| Missing Value Distribution | Missing vs non-missing | `df['col'].isnull().value_counts()` |
| Boolean Distribution | True/False balance | `df['col'].value_counts()` |

---

## Phase 4 — Bivariate Analysis
*Understand relationships between two variables.*

### Numeric vs Numeric
| Technique | Gain | Code |
|---|---|---|
| Scatter Plot | Trends, clusters, outliers | `sns.scatterplot(data=df, x='col1', y='col2')` |
| Correlation | Strength & direction | `df[['col1','col2']].corr()` |
| Regression Plot | Linear trend / prediction potential | `sns.regplot(data=df, x='col1', y='col2')` |
| Covariance | Joint variability | `df[['col1','col2']].cov()` |
| Correlation by Target | Identify predictive features | `df.corr(numeric_only=True)['TARGET'].sort_values()` |

### Numeric vs Categorical
| Technique | Gain | Code |
|---|---|---|
| Boxplot | Group differences & outliers | `sns.boxplot(data=df, x='cat', y='num')` |
| Violin Plot | Distribution shape by group | `sns.violinplot(data=df, x='cat', y='num')` |
| Group Statistics | Compare mean/median/spread | `df.groupby('cat')['num'].describe()` |
| Effect Size | Practical significance | `pingouin.compute_effsize(group1, group2)` |

### Categorical vs Categorical
| Technique | Gain | Code |
|---|---|---|
| Crosstab | Category associations | `pd.crosstab(df['cat1'], df['cat2'])` |
| Heatmap | Visualize crosstab | `sns.heatmap(pd.crosstab(df['cat1'], df['cat2']), annot=True)` |
| Stacked Bar Chart | Relative frequencies | `pd.crosstab(...).plot(kind='bar', stacked=True)` |
| Chi-Square Test | Independence test | `stats.chi2_contingency(pd.crosstab(df['cat1'], df['cat2']))` |
| Cramer's V | Association strength | `association(pd.crosstab(df['cat1'], df['cat2']))` |
| Odds Ratio | Relative likelihood (binary) | `Table2x2(table).oddsratio` |

### Feature vs Target
| Technique | Gain | Code |
|---|---|---|
| Numeric vs Target | Identify predictive features | `sns.boxplot(data=df, x='TARGET', y='num')` |
| Categorical vs Target | Category-target relationships | `pd.crosstab(df['cat'], df['TARGET'], normalize='index')` |
| Mutual Dependence | Shared information | `mutual_info_classif(X, y)` |
| Confidence Intervals | Estimate reliability | `stats.t.interval(...)` |

### Statistical Tests
| Test | Gain | Code |
|---|---|---|
| Pearson | Linear correlation | `stats.pearsonr(df['col1'], df['col2'])` |
| Spearman | Monotonic correlation | `stats.spearmanr(df['col1'], df['col2'])` |
| ANOVA | Compare means across groups | `stats.f_oneway(group1, group2, group3)` |
| t-Test | Compare means (2 groups) | `stats.ttest_ind(group1, group2)` |
| Mann-Whitney U | Non-parametric 2-group comparison | `stats.mannwhitneyu(group1, group2)` |
| Chi-Square | Categorical independence | `stats.chi2_contingency(pd.crosstab(df['cat1'], df['cat2']))` |

### Additional Visuals
| Technique | Gain | Code |
|---|---|---|
| Pair Plot | Pairwise relationships | `sns.pairplot(df[numeric_cols])` |
| Joint Plot | Scatter + marginal distributions | `sns.jointplot(data=df, x='col1', y='col2', kind='scatter')` |
| Hexbin Plot | Dense scatter plots | `plt.hexbin(df['col1'], df['col2'], gridsize=30)` |
| Point Plot | Category means with CI | `sns.pointplot(data=df, x='cat', y='num')` |
| Bar Plot | Compare group averages | `sns.barplot(data=df, x='cat', y='num')` |
| Mutual Information | Feature-target dependency | `mutual_info_classif(X, y)` |

---

## Phase 5 — Multivariate Analysis
*Study interactions among multiple variables.*

| Technique | Gain | Code |
|---|---|---|
| Correlation Matrix | Relationships among all numeric features | `df.corr(numeric_only=True)` |
| Correlation Heatmap | Visualize correlation matrix | `sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')` |
| Pair Plot | Pairwise trends/clusters/outliers | `sns.pairplot(df[numeric_cols])` |
| Feature Interaction Analysis | How features influence each other | `sns.pairplot(df[['col1','col2','TARGET']], hue='TARGET')` |
| Partial Correlation | Correlation controlling for other vars | `pg.partial_corr(data=df, x='col1', y='col2', covar='col3')` |
| Feature Network Graph | Feature relationships as a network | `nx.from_pandas_adjacency(df.corr(numeric_only=True))` |
| Redundancy Analysis | Duplicated information | `df.corr(numeric_only=True)` |
| Multicollinearity | Highly correlated predictors | `df.corr(numeric_only=True)` |
| VIF | Quantify multicollinearity | `variance_inflation_factor(X.values, i)` |
| PCA | Reduce dimensionality | `PCA(n_components=2).fit_transform(X)` |
| t-SNE | Visualize high-dim data | `TSNE(n_components=2).fit_transform(X)` |
| UMAP | Non-linear dim reduction | `umap.UMAP().fit_transform(X)` |
| Cluster Visualization | Evaluate cluster separation | `sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=labels)` |
| Hierarchical Clustering | Build hierarchical clusters | `linkage(X, method='ward')` |
| Cluster Heatmap | Feature groups | `sns.clustermap(df.corr(numeric_only=True))` |
| Feature Importance | Feature contribution | `model.feature_importances_` |
| Mutual Information | Feature-target dependency | `mutual_info_classif(X, y)` |
| Interaction Effects | Combined feature effects | `PolynomialFeatures(interaction_only=True)` |
| SHAP Interaction *(optional)* | Explain interaction effects | `shap.TreeExplainer(model).shap_interaction_values(X)` |
| Latent Structure Analysis | Hidden patterns/factors | `FactorAnalysis().fit_transform(X)` |
| Dimensionality Analysis | Intrinsic dimensionality | `PCA().fit(X).explained_variance_ratio_` |
| Parallel Coordinates Plot | Class separation across features | `pd.plotting.parallel_coordinates(df, 'TARGET')` |
| Andrews Curves | Multivariate patterns | `pd.plotting.andrews_curves(df, 'TARGET')` |
| Dendrogram | Hierarchical clustering visual | `dendrogram(linkage(X, method='ward'))` |
| Explained Variance | Variance captured by PCA | `PCA().fit(X).explained_variance_ratio_` |
| Silhouette Analysis | Clustering quality | `silhouette_score(X, labels)` |
| Feature Interaction Heatmap | Interaction strengths | `sns.heatmap(df.corr(numeric_only=True))` |

---

## Phase 6 — Outlier & Anomaly Analysis
*Identify unusual observations.*

| Technique | Gain | Code |
|---|---|---|
| IQR Method | Robust univariate outliers | `Q1=df['col'].quantile(.25); Q3=df['col'].quantile(.75); IQR=Q3-Q1; df[(df['col']<Q1-1.5*IQR)\|(df['col']>Q3+1.5*IQR)]` |
| Z-Score | Extreme values (normal data) | `np.abs(stats.zscore(df['col']))` |
| Modified Z-Score | Robust outlier detection (skewed data) | `median = np.median(df['col']); mad = stats.median_abs_deviation(df['col'])` |
| Isolation Forest | Multivariate anomalies | `IsolationForest().fit_predict(X)` |
| Local Outlier Factor (LOF) | Local density anomalies | `LocalOutlierFactor().fit_predict(X)` |
| DBSCAN | Clusters + noise simultaneously | `DBSCAN().fit_predict(X)` |
| Global Outliers | Values far from overall dataset | `stats.zscore(df['col'])` |
| Local Outliers | Values unusual within neighborhood | `LocalOutlierFactor().fit_predict(X)` |
| Point Anomalies | Individual abnormal observations | `IsolationForest().fit_predict(X)` |
| Contextual Anomalies | Abnormal under specific conditions | `sns.boxplot(data=df, x='category', y='value')` |
| Collective Anomalies | Abnormal groups/sequences | `DBSCAN().fit_predict(X)` |
| Cook's Distance | Influential regression observations | `influence = model.get_influence(); influence.cooks_distance` |
| Leverage Analysis | Unusual predictor values | `influence.hat_matrix_diag` |
| Visual Verification | Confirm anomalies visually | `sns.boxplot(x=df['col'])` / `sns.scatterplot(...)` |
| Business Validation | Validate anomalies against domain knowledge | Domain-specific validation |
| Percentile Method | Threshold-based detection | `df['col'].quantile([0.01, 0.99])` |
| Mahalanobis Distance | Multivariate outliers | `mahalanobis(x, mean, cov)` |
| Elliptic Envelope | Robust covariance-based detection | `EllipticEnvelope().fit_predict(X)` |
| Outlier Percentage | Decide if treatment is necessary | `len(outliers)/len(df)` |
| Before vs After Comparison | Verify treatment improved quality | `sns.boxplot(...)` |
| **Treatment Strategy** | Improve model performance | `df.drop(...)`, `np.clip(...)`, `Winsorizer()`, `np.log1p(...)`, `RobustScaler()` |

---

## Phase 7 — Feature Engineering & Preprocessing
*Prepare data for machine learning.*

### Missing Values & Outliers
| Technique | Gain | Code |
|---|---|---|
| Missing Value Imputation | Complete missing data | `SimpleImputer(strategy='mean')` |
| Missing Indicators | Preserve info from missingness | `df['col_missing'] = df['col'].isnull().astype(int)` |
| Outlier Treatment | Reduce influence of extremes | `np.clip()`, `Winsorizer()`, `RobustScaler()` |
| Rare Category Grouping | Reduce noise/overfitting | `df['col'] = df['col'].where(df['col'].map(df['col'].value_counts())>10, 'Other')` |

### Encoding
| Method | Gain | Code |
|---|---|---|
| Label Encoding | Binary/ordinal variables | `LabelEncoder().fit_transform(df['col'])` |
| One-Hot Encoding | Nominal categories | `pd.get_dummies(df['col'])` |
| Ordinal Encoding | Preserve category order | `OrdinalEncoder().fit_transform(df[['col']])` |
| Frequency Encoding | High-cardinality features | `df['col'].map(df['col'].value_counts())` |
| Target Encoding | Capture target relationship | `TargetEncoder().fit_transform(X, y)` |
| Weight of Evidence (WOE) | Common in credit risk modeling | `WOEEncoder().fit_transform(X, y)` |
| Feature Hashing | Very high-cardinality data | `FeatureHasher(n_features=16)` |

### Scaling
| Method | Gain | Code |
|---|---|---|
| StandardScaler | Mean=0, Std=1 | `StandardScaler()` |
| MinMaxScaler | Scale to [0,1] | `MinMaxScaler()` |
| RobustScaler | Robust to outliers | `RobustScaler()` |
| Normalization | Unit-length vectors | `Normalizer()` |

### Feature Creation & Transformation
| Technique | Gain | Code |
|---|---|---|
| Datetime Features | Capture temporal information | `df['date'].dt.year`, `.dt.month`, `.dt.dayofweek` |
| Text Features | Prepare text for ML | `TfidfVectorizer()` |
| Binning | Reduce noise, capture trends | `pd.cut(df['col'], bins=5)` |
| Log Transformations | Reduce skewness | `np.log1p(df['col'])` |
| Power Transformations | Stabilize variance / normality | `PowerTransformer()` |
| Polynomial Features | Capture non-linear relationships | `PolynomialFeatures()` |
| Interaction Features | Improve model expressiveness | `PolynomialFeatures(interaction_only=True)` |
| Cyclical Encoding | Preserve cyclic relationships | `np.sin()`, `np.cos()` |
| Feature Creation | Increase predictive power | `df['new'] = df['A'] / df['B']` |
| Feature Dropping | Reduce noise/memory usage | `df.drop(columns=[...])` |
| Data Type Optimization | Faster processing | `df.astype(...)` |

### Feature Selection & Dimensionality Reduction
| Method | Gain | Code |
|---|---|---|
| Filter Methods | Fast feature selection | `SelectKBest()` |
| Wrapper Methods | Better predictive subset | `RFE(model)` |
| Embedded Methods | Automatic feature selection | `Lasso()`, `RandomForest.feature_importances_` |
| Feature Extraction | Better feature representation | `PCA()`, `FeatureAgglomeration()` |
| Dimensionality Reduction | Faster training, less redundancy | `PCA()`, `TruncatedSVD()` |

### Class Imbalance Handling
| Method | Gain | Code |
|---|---|---|
| SMOTE | Balance classes (synthetic) | `SMOTE()` |
| ADASYN | Adaptive synthetic oversampling | `ADASYN()` |
| Undersampling | Reduce majority class | `RandomUnderSampler()` |

### Final Pipeline Steps
| Step | Gain | Code |
|---|---|---|
| Train / Validation / Test Split | Fair model evaluation | `train_test_split()` |
| Cross Validation Strategy | Robustness estimation | `cross_val_score()`, `StratifiedKFold()` |
| Feature Scaling Decision | Avoid unnecessary preprocessing | Based on model type |
| ColumnTransformer | Cleaner preprocessing pipelines | `ColumnTransformer(...)` |
| Data Leakage Prevention | Fit preprocessors on training data only | `Pipeline(...)` |
| Pipeline Creation | Automate preprocessing + modeling | `Pipeline()`, `ColumnTransformer()` |

---

## Phase 8 — Final EDA Report
*Summarize everything learned.*

| Section | Purpose |
|---|---|
| Executive Summary | Summarize dataset and main findings |
| Key Insights | Highlight the most important discoveries |
| Major Data Issues | Summarize data quality problems |
| Data Quality Score | Rate overall dataset quality |
| Missing Data Summary | Prioritize imputation |
| Outlier Summary | Decide treatment strategy |
| Important Relationships | Significant correlations/associations |
| Key Visualizations | Present the most informative plots |
| Feature Ranking | Rank features by importance |
| Selected Features | Document final feature set |
| Assumptions | Document assumptions made during EDA |
| Risks & Limitations | Describe dataset limitations |
| Recommended Preprocessing | List preprocessing steps |
| Modeling Strategy | Recommend suitable ML algorithms |
| Modeling Readiness | Checklist of readiness for modeling |
| Reproducibility Notes | Record versions, seeds, methodology |
| Appendix | Extra tables, plots, supporting analysis |
| Dataset Overview | Rows, columns, target, data types |
| Feature Summary Table | One-page feature reference |
| Data Dictionary | Feature descriptions and meanings |
| EDA Checklist | Ensure nothing was missed |
| Business Recommendations | Connect EDA to real-world decisions |
| Next Steps | Outline modeling and deployment plan |

**Key code:** `df.info()`, `df.isnull().sum()`, `df.duplicated().sum()`, `df.corr(numeric_only=True)`, `model.feature_importances_`, `np.random.seed(42)`, `sklearn.__version__`

---

## Phase 9 — Model Readiness Checklist
*Verify the data is ready for modeling.*

- [ ] **Business Problem Clearly Defined** — Confirm the ML objective is well understood *(problem statement documented)*
- [ ] **No Data Leakage** — `X.columns`, `df.corr(numeric_only=True)['TARGET']`
- [ ] **Missing Values Handled** — `df.isnull().sum()`
- [ ] **Outliers Reviewed** — `sns.boxplot(x=df['col'])`
- [ ] **Correct Data Types** — `df.dtypes`
- [ ] **Features Properly Encoded** — `X.head()`
- [ ] **Features Properly Scaled** — `X.describe()`
- [ ] **Feature Selection Completed** — `selected_features`
- [ ] **Class Imbalance Addressed** — `y.value_counts(normalize=True)`
- [ ] **Train / Validation / Test Split Completed** — `X_train.shape`, `X_valid.shape`, `X_test.shape`
- [ ] **Cross Validation Ready** — `StratifiedKFold()`, `cross_val_score(model, X_train, y_train)`
- [ ] **Pipeline Tested** — `pipeline.fit(X_train, y_train)`
- [ ] **Baseline Model Ready** — `DummyClassifier()`, `LogisticRegression()`
- [ ] **Target Variable Verified** — `y.value_counts()`
- [ ] **Feature Names Verified** — `X.columns`
- [ ] **Reproducibility Ready** — `np.random.seed(42)`, `sklearn.__version__`
- [ ] **Evaluation Metric Selected** — accuracy, F1, ROC-AUC, RMSE, etc.
- [ ] **Model Assumptions Checked** — model-specific checks
- [ ] **Resource Check** — `df.memory_usage(deep=True).sum()`
- [ ] **Documentation Complete** — preprocessing, assumptions, decisions recorded

---

*End of roadmap — Phase 0 (Business Understanding) → Phase 9 (Model Readiness).*
