import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Загрузка датасета
df = pd.read_csv('Dataset.csv')

# Фильтрация по нужной категории
target_part = "RG3 MOLD'G W/SHLD, RH"
df_filtered = df[df['PART_NAME'] == target_part].copy()

# Проверка наличия колонки PassOrFail
if 'PassOrFail' not in df_filtered.columns:
    raise ValueError("Колонка 'PassOrFail' не найдена в датасете. Пожалуйста, проверьте название колонки.")

# Преобразование целевой переменной в числовой формат
print("\nУникальные значения в целевой переменной до преобразования:")
print(df_filtered['PassOrFail'].unique())

# Создаем словарь для преобразования
pass_fail_mapping = {'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, '1': 1, '0': 0}
df_filtered['PassOrFail'] = df_filtered['PassOrFail'].map(pass_fail_mapping)

# Проверяем результат преобразования
print("\nУникальные значения в целевой переменной после преобразования:")
print(df_filtered['PassOrFail'].unique())

# Анализ типов данных
print("\nТипы данных в датасете:")
print(df_filtered.dtypes)

# Удаление нечисловых колонок, кроме целевой переменной
non_numeric_cols = df_filtered.select_dtypes(exclude=['int64', 'float64']).columns
non_numeric_cols = [col for col in non_numeric_cols if col != 'PassOrFail']
print("\nУдаляемые нечисловые колонки:")
print(non_numeric_cols)
df_filtered = df_filtered.drop(columns=non_numeric_cols)

# Удаление колонок с нулевыми значениями
zero_cols = df_filtered.columns[df_filtered.sum() == 0]
print("\nУдаляемые колонки с нулевыми значениями:")
print(zero_cols)
df_filtered = df_filtered.drop(columns=zero_cols)

# Удаление колонок с одной уникальной переменной
single_unique_cols = [col for col in df_filtered.columns if df_filtered[col].nunique() == 1]
print("\nУдаляемые колонки с одной уникальной переменной:")
print(single_unique_cols)
df_filtered = df_filtered.drop(columns=single_unique_cols)

# Удаление колонок с низкой дисперсией
variance_threshold = 0.01
low_variance_cols = [col for col in df_filtered.columns if df_filtered[col].var() < variance_threshold]
print("\nУдаляемые колонки с низкой дисперсией:")
print(low_variance_cols)
df_filtered = df_filtered.drop(columns=low_variance_cols)

# Удаление колонок с высокой долей пропущенных значений
missing_threshold = 0.5
high_missing_cols = [col for col in df_filtered.columns if df_filtered[col].isnull().mean() > missing_threshold]
print("\nУдаляемые колонки с высокой долей пропущенных значений:")
print(high_missing_cols)
df_filtered = df_filtered.drop(columns=high_missing_cols)

# Анализ корреляций с целевой переменной
target_corr = df_filtered.corr()['PassOrFail'].abs().sort_values(ascending=False)
low_corr_cols = target_corr[target_corr < 0.01].index.tolist()
print("\nУдаляемые колонки с низкой корреляцией с целевой переменной:")
print(low_corr_cols)
df_filtered = df_filtered.drop(columns=low_corr_cols)

# Анализ распределения классов
print("\nРаспределение классов в датасете:")
class_dist = df_filtered['PassOrFail'].value_counts(normalize=True)
print(class_dist)

# Визуализация распределения классов
plt.figure(figsize=(10, 6))
sns.countplot(x='PassOrFail', data=df_filtered)
plt.title('Распределение классов в датасете')
plt.show()

# Анализ корреляций
correlation_matrix = df_filtered.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Матрица корреляций')
plt.tight_layout()
plt.show()

# Удаление коррелирующих признаков
threshold = 0.9
corr_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            corr_features.add(colname)

print("\nУдаленные коррелирующие признаки:")
print(corr_features)
df_filtered = df_filtered.drop(columns=corr_features)

# Разделение на признаки и целевую переменную
X = df_filtered.drop('PassOrFail', axis=1)
y = df_filtered['PassOrFail']

# Проверка на наличие пропущенных значений
print("\nПропущенные значения в признаках:")
print(X.isnull().sum())
print("\nПропущенные значения в целевой переменной:")
print(y.isnull().sum())

# Удаление строк с пропущенными значениями
X = X.dropna()
y = y[X.index]  # Сохраняем соответствующие значения y

# Проверка на бесконечные значения
print("\nБесконечные значения в признаках:")
print(np.isinf(X).sum())

# Замена бесконечных значений на NaN и их удаление
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()
y = y[X.index]

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Список моделей для сравнения
models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, class_weight='balanced', probability=True),
    'KNN': KNeighborsClassifier()
}

# Методы балансировки
sampling_methods = {
    'Original': None,
    'SMOTE': SMOTE(random_state=42),
    'UnderSampling': RandomUnderSampler(random_state=42),
    'SMOTEENN': SMOTEENN(random_state=42)
}

# Создание фигуры для ROC-кривых
plt.figure(figsize=(12, 8))

# Обучение и оценка моделей
results = {}
for model_name, model in models.items():
    print(f"\n=== {model_name} ===")
    model_results = {}
    
    for sampling_name, sampler in sampling_methods.items():
        if sampler is not None:
            X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
        else:
            X_resampled, y_resampled = X_train_scaled, y_train
            
        # Обучение модели
        model.fit(X_resampled, y_resampled)
        
        # Предсказания
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # ROC-кривая
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Построение ROC-кривой
        plt.plot(fpr, tpr, label=f'{model_name} - {sampling_name} (AUC = {roc_auc:.2f})')
        
        # Сохранение результатов
        model_results[sampling_name] = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc
        }
        
        # Вывод результатов
        print(f"\n{sampling_name}:")
        print(classification_report(y_test, y_pred))
        
        # Визуализация матрицы ошибок
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Матрица ошибок - {model_name} ({sampling_name})')
        plt.show()
    
    results[model_name] = model_results

# Завершение ROC-кривой
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые для всех моделей')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Анализ важности признаков для лучшей модели
best_model = RandomForestClassifier(random_state=42, class_weight='balanced')
best_model.fit(X_train_scaled, y_train)

feature_importance = pd.DataFrame({
    'Признак': X.columns,
    'Важность': best_model.feature_importances_
}).sort_values('Важность', ascending=False)

print("\nТоп-10 самых важных признаков:")
print(feature_importance.head(10))

# Визуализация важности признаков
plt.figure(figsize=(12, 8))
sns.barplot(x='Важность', y='Признак', data=feature_importance.head(10))
plt.title('Топ-10 самых важных признаков')
plt.tight_layout()
plt.show()

# Кросс-валидация для лучшей модели
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print("\nРезультаты кросс-валидации:")
print(f"Средний AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Сохранение лучшей модели
import joblib
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl') 