import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE

# Игнорирование предупреждений
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Загрузка датасета
print("\nЗагрузка датасета...")
df = pd.read_csv('Dataset.csv')

# Фильтрация по нужным категориям
print("\nФильтрация данных...")
target_parts = ["RG3 MOLD'G W/SHLD, RH", "RG3 MOLD'G W/SHLD, LH"]
df_filtered = df[df['PART_NAME'].isin(target_parts)].copy()

# Проверка наличия колонки PassOrFail
if 'PassOrFail' not in df_filtered.columns:
    print("Предупреждение: Колонка 'PassOrFail' не найдена в датасете. Пожалуйста, проверьте название колонки.")
    # Создаем колонку PassOrFail с нулевыми значениями
    df_filtered['PassOrFail'] = 0
    print("Создана колонка PassOrFail с нулевыми значениями")

# Преобразование целевой переменной
print("\nПреобразование целевой переменной...")
pass_fail_mapping = {'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, '1': 1, '0': 0}
df_filtered['PassOrFail'] = df_filtered['PassOrFail'].map(pass_fail_mapping)

# Анализ типов данных
print("\nАнализ типов данных:")
print(df_filtered.dtypes)

# Визуализация распределения классов
plt.figure(figsize=(12, 8))
ax = sns.countplot(x='PassOrFail', data=df_filtered, palette='viridis')
plt.title('Распределение классов в датасете', fontsize=16, pad=20)
plt.xlabel('Класс', fontsize=14)
plt.ylabel('Количество наблюдений', fontsize=14)

# Добавляем значения над столбцами
for p in ax.patches:
    ax.annotate(f'{p.get_height()}\n({p.get_height()/len(df_filtered):.1%})', 
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
                textcoords='offset points')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Удаление нечисловых колонок, кроме целевой переменной
print("\nУдаление нечисловых колонок...")
non_numeric_cols = df_filtered.select_dtypes(exclude=['int64', 'float64']).columns
non_numeric_cols = [col for col in non_numeric_cols if col != 'PassOrFail']
df_filtered = df_filtered.drop(columns=non_numeric_cols)

# Удаление колонок с нулевыми значениями
print("\nУдаление колонок с нулевыми значениями...")
zero_cols = df_filtered.columns[df_filtered.sum() == 0]
df_filtered = df_filtered.drop(columns=zero_cols)

# Удаление колонок с одной уникальной переменной
print("\nУдаление колонок с одной уникальной переменной...")
single_unique_cols = [col for col in df_filtered.columns if df_filtered[col].nunique() == 1]
df_filtered = df_filtered.drop(columns=single_unique_cols)

# Анализ пропущенных значений
print("\nАнализ пропущенных значений:")
missing_values = df_filtered.isnull().sum()
print(missing_values[missing_values > 0])

# Визуализация пропущенных значений
plt.figure(figsize=(15, 8))
missing_data = pd.DataFrame({
    'Количество': missing_values,
    'Процент': (missing_values / len(df_filtered)) * 100
})
missing_data = missing_data[missing_data['Количество'] > 0].sort_values('Количество', ascending=False)

ax = sns.barplot(x=missing_data.index, y='Количество', data=missing_data, color='salmon')
plt.title('Анализ пропущенных значений', fontsize=16, pad=20)
plt.xlabel('Признаки', fontsize=14)
plt.ylabel('Количество пропущенных значений', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Добавляем проценты над столбцами
for i, v in enumerate(missing_data['Количество']):
    ax.text(i, v + 0.5, '{:.1f}%'.format(missing_data['Процент'].iloc[i]),
            ha='center', va='bottom', fontsize=12, color='black')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Удаление колонок с высокой долей пропущенных значений
print("\nУдаление колонок с высокой долей пропущенных значений...")
missing_threshold = 0.5
high_missing_cols = [col for col in df_filtered.columns if df_filtered[col].isnull().mean() > missing_threshold]
df_filtered = df_filtered.drop(columns=high_missing_cols)

# Анализ корреляций
print("\nАнализ корреляций...")
correlation_matrix = df_filtered.corr()
plt.figure(figsize=(15, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
            fmt='.2f', vmin=-1, vmax=1, center=0, square=True,
            linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Матрица корреляций', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# Удаление коррелирующих признаков
print("\nУдаление коррелирующих признаков...")
threshold = 0.9
corr_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            corr_features.add(colname)
df_filtered = df_filtered.drop(columns=corr_features)

# Анализ выбросов
print("\nАнализ выбросов...")
Q1 = df_filtered.quantile(0.25)
Q3 = df_filtered.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df_filtered < (Q1 - 1.5 * IQR)) | (df_filtered > (Q3 + 1.5 * IQR))).sum()
print("\nКоличество выбросов по признакам:")
print(outliers)

# Визуализация выбросов
plt.figure(figsize=(20, 12))
sns.boxplot(data=df_filtered, palette='Set3', width=0.8)
plt.title('Анализ выбросов по признакам', fontsize=18, pad=20)
plt.xlabel('Признаки', fontsize=16)
plt.ylabel('Значения', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Добавляем сетку
plt.grid(True, alpha=0.3, linestyle='--')

# Добавляем аннотации с количеством выбросов
for i, col in enumerate(df_filtered.columns):
    outlier_count = outliers[col]
    if outlier_count > 0:
        plt.text(i, df_filtered[col].max() * 1.05, 
                f'Выбросов: {outlier_count}\n({outlier_count/len(df_filtered):.1%})',
                ha='center', va='bottom', fontsize=10, color='red')

plt.tight_layout()
plt.show()

# Разделение на признаки и целевую переменную
X = df_filtered.drop('PassOrFail', axis=1)
y = df_filtered['PassOrFail']

# Удаление строк с пропущенными значениями
X = X.dropna()
y = y[X.index]

# Разделение данных
print("\nРазделение данных на train и test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация признаков
print("\nНормализация признаков...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Распределение признаков после нормализации
plt.figure(figsize=(20, 15))
for i, col in enumerate(X_train_scaled.columns, 1):
    plt.subplot(4, 3, i)
    sns.histplot(data=X_train_scaled[col], kde=True, color='lightgreen')
    plt.title(f'Распределение {col} (нормализованное)', fontsize=12)
    plt.xlabel('Нормализованное значение', fontsize=10)
    plt.ylabel('Частота', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Добавляем статистику
    mean = X_train_scaled[col].mean()
    median = X_train_scaled[col].median()
    std = X_train_scaled[col].std()
    plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='g', linestyle='--', label=f'Median: {median:.2f}')
    plt.legend(fontsize=8)
    
    if i == 12:  # Ограничиваем количество графиков
        break

plt.suptitle('Распределение признаков после нормализации (обучающая выборка)', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# Визуализация данных до балансировки
print("\nВизуализация данных до балансировки...")
top_features = X_train_scaled.columns[:3]
plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_features, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(x=feature, y='PassOrFail', data=pd.concat([X_train_scaled, y_train], axis=1), alpha=0.5)
    plt.title(f'До балансировки: {feature}')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Визуализация данных после балансировки
print("\nВизуализация данных после балансировки...")
sampling_methods = {
    'SMOTE': SMOTE(random_state=42),
    'UnderSampling': RandomUnderSampler(random_state=42),
    'SMOTEENN': SMOTEENN(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42)
}

all_model_results = []

for sampling_name, sampler in sampling_methods.items():
    X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
    df_resampled = pd.DataFrame(X_resampled, columns=X_train_scaled.columns)
    df_resampled['PassOrFail'] = y_resampled
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(1, 3, i)
        sns.scatterplot(x=feature, y='PassOrFail', data=df_resampled, alpha=0.5)
        plt.title(f'После {sampling_name}: {feature}')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Базовая модель через Logistic Regression
print("\n=== Базовая модель через Logistic Regression ===")
baseline_lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
baseline_lr.fit(X_train_scaled, y_train)
y_pred_lr = baseline_lr.predict(X_test_scaled)
y_prob_lr = baseline_lr.predict_proba(X_test_scaled)[:, 1]

# Оценка базовой модели Logistic Regression
print("\nОтчет о классификации для базовой модели Logistic Regression:")
print(classification_report(y_test, y_pred_lr, zero_division=0))

# ROC-кривая для базовой модели Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Визуализация матрицы ошибок для базовой модели Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок - Базовая модель Logistic Regression')
plt.show()

# Создание фигуры для сравнения ROC-кривых
plt.figure(figsize=(15, 10))

# ROC-кривая для базовой модели Logistic Regression
plt.plot(fpr_lr, tpr_lr, 'b-', 
         label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')

# Методы балансировки
print("\n=== Балансировка данных ===")

# Создание фигуры для Precision-Recall кривых
plt.figure(figsize=(15, 10))
for result in all_model_results:
    model_name = result['model_name']
    sampling_name = result['sampling_name']
    precision = result['precision']
    recall = result['recall']
    pr_auc = result['pr_auc']
    
    plt.plot(recall, precision, 
             label=f'{model_name} - {sampling_name} (PR AUC = {pr_auc:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall кривые для всех моделей')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Создание DataFrame с результатами всех моделей
results_df = pd.DataFrame(all_model_results)
results_df = results_df[['model_name', 'sampling_name', 'roc_auc', 'pr_auc']]

# Визуализация метрик производительности
plt.figure(figsize=(15, 8))
sns.barplot(data=results_df, x='model_name', y='roc_auc', hue='sampling_name')
plt.title('AUC-ROC для всех моделей и методов балансировки')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))
sns.barplot(data=results_df, x='model_name', y='pr_auc', hue='sampling_name')
plt.title('PR AUC для всех моделей и методов балансировки')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Вывод результатов в табличном виде
print("\nСравнение результатов всех моделей:")
print(results_df.sort_values('roc_auc', ascending=False))

# Определение лучшей модели
best_result = results_df.loc[results_df['roc_auc'].idxmax()]
print(f"\nЛучшая модель: {best_result['model_name']} с методом балансировки {best_result['sampling_name']}")
print(f"AUC-ROC: {best_result['roc_auc']:.3f}")

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
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Визуализация распределения PassOrFail по важным признакам
plt.figure(figsize=(15, 10))
top_features = feature_importance.head(3)['Признак'].tolist()

for i, feature in enumerate(top_features, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(x=feature, y='PassOrFail', data=df_filtered, alpha=0.5)
    plt.title(f'Распределение PassOrFail по {feature}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Визуализация распределения PassOrFail по двум самым важным признакам
plt.figure(figsize=(10, 8))
sns.scatterplot(x=top_features[0], y=top_features[1], 
                hue='PassOrFail', data=df_filtered, 
                palette='coolwarm', alpha=0.6)
plt.title(f'Распределение PassOrFail по {top_features[0]} и {top_features[1]}')
plt.grid(True, alpha=0.3)
plt.legend(title='PassOrFail')
plt.show()

# Определение числовых и категориальных признаков
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Создание трансформеров для числовых и категориальных признаков
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Создание ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Создание итогового пайплайна с лучшей моделью
best_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])

# Обучение итогового пайплайна
print("\nОбучение итогового пайплайна...")
best_model_pipeline.fit(X_train_scaled, y_train)

# Оценка качества итогового пайплайна
print("\nОценка качества итогового пайплайна:")
y_pred_pipeline = best_model_pipeline.predict(X_test_scaled)
y_prob_pipeline = best_model_pipeline.predict_proba(X_test_scaled)[:, 1]

# ROC-кривая для итогового пайплайна
fpr_pipeline, tpr_pipeline, _ = roc_curve(y_test, y_prob_pipeline)
roc_auc_pipeline = auc(fpr_pipeline, tpr_pipeline)

plt.figure(figsize=(10, 8))
plt.plot(fpr_pipeline, tpr_pipeline, color='darkorange',
         lw=2, label=f'ROC curve (AUC = {roc_auc_pipeline:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Final Pipeline')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# Вывод метрик для итогового пайплайна
print("\nClassification Report для итогового пайплайна:")
print(classification_report(y_test, y_pred_pipeline, zero_division=0))

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_pipeline), 
            annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Final Pipeline')
plt.show()

# Сохранение итогового пайплайна
print("\nСохранение итогового пайплайна...")
joblib.dump(best_model_pipeline, 'final_pipeline.pkl') 