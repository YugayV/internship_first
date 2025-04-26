# ===================================================
# Импорт необходимых библиотек / Import required libraries
# ===================================================
import pandas as pd  # Для работы с данными / For data manipulation
import numpy as np  # Для математических операций / For mathematical operations
import matplotlib.pyplot as plt  # Для визуализации / For visualization
import seaborn as sns  # Для улучшенной визуализации / For enhanced visualization
import warnings  # Для управления предупреждениями / For warning management
import joblib  # Для сохранения моделей / For saving models

# Импорт методов машинного обучения / Import machine learning methods
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # Для разделения данных / For data splitting
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Для масштабирования данных / For data scaling
from sklearn.decomposition import PCA  # Для анализа главных компонент / For principal component analysis
from sklearn.linear_model import LogisticRegression  # Логистическая регрессия
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, HistGradientBoostingClassifier  # Случайный лес, Градиентный бустинг, Экстра-деревья, Бэггинг, Градиентный бустинг с гистограммами
from sklearn.svm import SVC  # Метод опорных векторов
from sklearn.neighbors import KNeighborsClassifier  # Метод k-ближайших соседей
from sklearn.tree import DecisionTreeClassifier  # Дерево решений
from xgboost import XGBClassifier  # Экстремальный градиентный бустинг
from lightgbm import LGBMClassifier  # Легкий градиентный бустинг
from catboost import CatBoostClassifier  # Категориальный бустинг
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score  # Отчет о классификации, Матрица ошибок, ROC-кривая, Площадь под ROC-кривой, Precision-Recall кривая, Средняя точность
from sklearn.dummy import DummyClassifier  # Случайная модель
from sklearn.neighbors import KNeighborsClassifier  # Метод k-ближайших соседей
from sklearn.naive_bayes import GaussianNB  # Наивный байесовский классификатор
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  # Квадратичный дискриминантный анализ
from sklearn.pipeline import Pipeline  # Для создания конвейера / For creating pipeline

# Импорт методов балансировки / Import balancing methods
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler  # Случайное уменьшение выборки
from imblearn.combine import SMOTEENN  # SMOTE с редактированием соседей
from imblearn.over_sampling import ADASYN  # Адаптивная синтетическая выборка
from imblearn.over_sampling import BorderlineSMOTE  # SMOTE для пограничных случаев

# Игнорирование предупреждений / Ignoring warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ===================================================
# Загрузка и предобработка данных / Loading and preprocessing data
# ===================================================
print("\nЗагрузка датасета...")
print("\nLoading dataset...")
df = pd.read_csv('Dataset.csv')

# Фильтрация данных по нужным категориям / Filtering data by required categories
print("\nФильтрация данных...")
print("\nFiltering data...")
target_parts = ["RG3 MOLD'G W/SHLD, RH", "RG3 MOLD'G W/SHLD, LH"]
df_filtered = df[df['PART_NAME'].isin(target_parts)].copy()

# Проверка наличия колонки PassOrFail / Checking for PassOrFail column
if 'PassOrFail' not in df_filtered.columns:
    print("Предупреждение: Колонка 'PassOrFail' не найдена в датасете. Пожалуйста, проверьте название колонки.")
    print("Warning: Column 'PassOrFail' not found in dataset. Please check column name.")
    # Создаем колонку PassOrFail с нулевыми значениями / Creating PassOrFail column with zero values
    df_filtered['PassOrFail'] = 0
    print("Создана колонка PassOrFail с нулевыми значениями")
    print("Created PassOrFail column with zero values")

# Преобразование целевой переменной / Transforming target variable
print("\nПреобразование целевой переменной...")
print("\nTransforming target variable...")
pass_fail_mapping = {'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, '1': 1, '0': 0}
df_filtered['PassOrFail'] = df_filtered['PassOrFail'].map(pass_fail_mapping)

# Анализ типов данных / Data types analysis
print("\nАнализ типов данных:")
print("\nData types analysis:")
print(df_filtered.dtypes)

# Визуализация распределения классов / Class distribution visualization
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='PassOrFail', data=df_filtered, palette=['#ff9999', '#66b3ff'])
plt.title('Распределение классов / Class Distribution', fontsize=14, pad=20)
plt.xlabel('Класс (0 - Неудача, 1 - Успех) / Class (0 - Fail, 1 - Pass)', fontsize=12)
plt.ylabel('Количество / Count', fontsize=12)

# Добавляем значения над столбцами / Adding values above bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}\n({p.get_height()/len(df_filtered):.1%})', 
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Удаление нечисловых колонок / Removing non-numeric columns
print("\nУдаление нечисловых колонок...")
print("\nRemoving non-numeric columns...")
non_numeric_cols = df_filtered.select_dtypes(exclude=['int64', 'float64']).columns
non_numeric_cols = [col for col in non_numeric_cols if col != 'PassOrFail']
df_filtered = df_filtered.drop(columns=non_numeric_cols)

# Удаление колонок с нулевыми значениями / Removing columns with zero values
print("\nУдаление колонок с нулевыми значениями...")
print("\nRemoving columns with zero values...")
zero_cols = df_filtered.columns[df_filtered.sum() == 0]
df_filtered = df_filtered.drop(columns=zero_cols)

# Удаление колонок с одной уникальной переменной / Removing columns with single unique value
print("\nУдаление колонок с одной уникальной переменной...")
print("\nRemoving columns with single unique value...")
single_unique_cols = [col for col in df_filtered.columns if df_filtered[col].nunique() == 1]
df_filtered = df_filtered.drop(columns=single_unique_cols)

# Анализ пропущенных значений / Missing values analysis
print("\nАнализ пропущенных значений:")
print("\nMissing values analysis:")
missing_values = df_filtered.isnull().sum()
print(missing_values[missing_values > 0])

# Удаление колонок с высокой долей пропущенных значений / Removing columns with high percentage of missing values
print("\nУдаление колонок с высокой долей пропущенных значений...")
print("\nRemoving columns with high percentage of missing values...")
missing_threshold = 0.5
high_missing_cols = [col for col in df_filtered.columns if df_filtered[col].isnull().mean() > missing_threshold]
df_filtered = df_filtered.drop(columns=high_missing_cols)

# Анализ выбросов / Outliers analysis
print("\nАнализ выбросов...")
print("\nOutliers analysis...")
Q1 = df_filtered.quantile(0.25)
Q3 = df_filtered.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df_filtered < (Q1 - 1.5 * IQR)) | (df_filtered > (Q3 + 1.5 * IQR))).sum()
print("\nКоличество выбросов по признакам:")
print("\nNumber of outliers by features:")
print(outliers)

# Создаем копии датасета для разных методов обработки выбросов / Creating dataset copies for different outlier handling methods
df_mean = df_filtered.copy()
df_median = df_filtered.copy()
df_zscore = df_filtered.copy()

# Метод 1: Замена на среднее значение / Method 1: Replacement with mean value
print("\nМетод 1: Замена выбросов на среднее значение...")
print("\nMethod 1: Replacing outliers with mean value...")
for column in df_mean.columns:
    if column != 'PassOrFail':
        Q1 = df_mean[column].quantile(0.25)
        Q3 = df_mean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mean_value = df_mean[(df_mean[column] >= lower_bound) & 
                            (df_mean[column] <= upper_bound)][column].mean()
        
        df_mean.loc[df_mean[column] < lower_bound, column] = mean_value
        df_mean.loc[df_mean[column] > upper_bound, column] = mean_value

# Метод 2: Замена на медиану / Method 2: Replacement with median
print("\nМетод 2: Замена выбросов на медиану...")
print("\nMethod 2: Replacing outliers with median...")
for column in df_median.columns:
    if column != 'PassOrFail':
        Q1 = df_median[column].quantile(0.25)
        Q3 = df_median[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        median_value = df_median[(df_median[column] >= lower_bound) & 
                                (df_median[column] <= upper_bound)][column].median()
        
        df_median.loc[df_median[column] < lower_bound, column] = median_value
        df_median.loc[df_median[column] > upper_bound, column] = median_value

# Метод 3: Z-score / Method 3: Z-score
print("\nМетод 3: Удаление выбросов по Z-score...")
print("\nMethod 3: Removing outliers by Z-score...")
for column in df_zscore.columns:
    if column != 'PassOrFail':
        z_scores = np.abs((df_zscore[column] - df_zscore[column].mean()) / df_zscore[column].std())
        df_zscore = df_zscore[z_scores < 3]

# Визуализация результатов разных методов / Visualization of different methods results
plt.figure(figsize=(10, 6))

# Создаем DataFrame для визуализации разницы / Create DataFrame for difference visualization
df_diff = pd.DataFrame({
    'Разница с Mean / Diff with Mean': df_filtered.iloc[:, 0] - df_mean.iloc[:, 0],
    'Разница с Median / Diff with Median': df_filtered.iloc[:, 0] - df_median.iloc[:, 0]
})

# Boxplot для сравнения разницы / Boxplot for difference comparison
sns.boxplot(data=df_diff, palette='Set2', showfliers=False)
plt.title('Разница между оригинальными данными и методами обработки', fontsize=14)
plt.xlabel('Метод обработки', fontsize=12)
plt.ylabel('Разница значений', fontsize=12)

# Добавляем горизонтальную линию на уровне 0 / Add horizontal line at zero
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Используем метод медианы как основной / Using median method as main method
df_filtered = df_median.copy()

# Разделение на признаки и целевую переменную / Splitting into features and target variable
X = df_filtered.drop('PassOrFail', axis=1)
y = df_filtered['PassOrFail']

# Удаление строк с пропущенными значениями / Removing rows with missing values
X = X.dropna()
y = y[X.index]

# Разделение данных / Splitting data
print("\nРазделение данных на train и test...")
print("\nSplitting data into train and test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Нормализация данных / Data normalization
print("\nНормализация данных...")
print("\nNormalizing data...")

# Сохраняем оригинальные данные / Save original data
X_train_original = X_train.copy()
X_test_original = X_test.copy()

# Визуализация до нормализации / Visualization before normalization
plt.figure(figsize=(12, 6))
means_before = X_train_original.mean()
plt.bar(range(len(means_before)), means_before.values, color='#66b3ff')
plt.title('Средние значения до нормализации\nMeans before normalization', fontsize=14)
plt.xticks(range(len(means_before)), means_before.index, rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Нормализуем данные / Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_original)
X_test_scaled = scaler.transform(X_test_original)

# Преобразование обратно в DataFrame для удобства / Converting back to DataFrame for convenience
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_original.columns, index=X_train_original.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_original.columns, index=X_test_original.index)

# Визуализация после нормализации / Visualization after normalization
plt.figure(figsize=(12, 6))
means_after = X_train_scaled.mean()
plt.bar(range(len(means_after)), means_after.values, color='#ff9999')
plt.title('Средние значения после нормализации\nMeans after normalization', fontsize=14)
plt.xticks(range(len(means_after)), means_after.index, rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Методы балансировки / Balancing methods
sampling_methods = {
    'SMOTE': SMOTE(random_state=0),
    'UnderSampling': RandomUnderSampler(random_state=0),
    'SMOTEENN': SMOTEENN(random_state=0),
    'ADASYN': ADASYN(random_state=0),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=0),
}

# Визуализация распределения данных после каждого метода балансировки / Visualization of data distribution after each balancing method
print("\nВизуализация методов балансировки...")
print("\nVisualizing balancing methods...")

# Создаем фигуру для всех методов балансировки / Creating figure for all balancing methods
n_methods = len(sampling_methods)
n_cols = 3
n_rows = (n_methods + 1) // n_cols + 1

# Исходные данные / Original data
plt.figure(figsize=(15, 5))
sns.countplot(x=y_train, palette=['#ff9999', '#66b3ff'])
plt.title('Исходные данные / Original Data', fontsize=14)
plt.xlabel('Класс / Class', fontsize=12)
plt.ylabel('Количество / Count', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Применяем каждый метод балансировки и визуализируем результат / Applying each balancing method and visualizing result
for method_name, sampler in sampling_methods.items():
    try:
        X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
        plt.figure(figsize=(15, 5))
        sns.countplot(x=y_resampled, palette=['#ff9999', '#66b3ff'])
        plt.title(f'{method_name}', fontsize=14)
        plt.xlabel('Класс / Class', fontsize=12)
        plt.ylabel('Количество / Count', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Ошибка при применении метода {method_name}: {str(e)}")
        print(f"Error applying method {method_name}: {str(e)}")
        continue

# Параметры для GridSearchCV / Parameters for GridSearchCV
model_params = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=0, max_iter=1000),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=0),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=0),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=0),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=0),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(random_state=0, verbose=False),
        'params': {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [3, 5, 7],
            'l2_leaf_reg': [1, 3, 5]
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=0),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=0),
        'params': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Extra Trees': {
        'model': ExtraTreesClassifier(random_state=0),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Bagging': {
        'model': BaggingClassifier(random_state=0),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_samples': [0.5, 0.7, 1.0],
            'max_features': [0.5, 0.7, 1.0]
        }
    }
}

# Функция для поиска лучших параметров / Function to find best parameters
def find_best_params(X_train, y_train, model_name, params):
    print(f"\nПоиск лучших параметров для {model_name}...")
    print(f"\nFinding best parameters for {model_name}...")
    
    grid_search = GridSearchCV(
        estimator=params['model'],
        param_grid=params['params'],
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nЛучшие параметры для {model_name}:")
    print(f"\nBest parameters for {model_name}:")
    print(grid_search.best_params_)
    print(f"Лучший ROC AUC: {grid_search.best_score_:.4f}")
    print(f"Best ROC AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# Поиск лучших параметров для каждой модели / Finding best parameters for each model
best_models = {}
for model_name, params in model_params.items():
    best_models[model_name] = find_best_params(X_train_scaled, y_train, model_name, params)

# Обновление списка моделей с лучшими параметрами / Updating models list with best parameters
models = best_models

# Функция для обучения и оценки модели / Function for model training and evaluation
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, sampling_name=None):
    try:
        # Обучение модели / Training model
        model.fit(X_train, y_train)
        
        # Предсказания / Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Оценка модели / Model evaluation
        print(f"\nОтчет о классификации для {model_name}" + (f" с {sampling_name}" if sampling_name else ""))
        print(f"\nClassification report for {model_name}" + (f" with {sampling_name}" if sampling_name else ""))
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # ROC-кривая / ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        return {
            'model': model_name,
            'sampling': sampling_name if sampling_name else 'None',
            'roc_auc': roc_auc,
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
    except Exception as e:
        print(f"Ошибка при обучении модели {model_name}" + (f" с {sampling_name}" if sampling_name else ""))
        print(f"Error training model {model_name}" + (f" with {sampling_name}" if sampling_name else ""))
        print(str(e))
        return None

# Создание базовых моделей / Creating baseline models
print("\n=== Базовая модель на несбалансированных данных ===")
print("\n=== Baseline model on unbalanced data ===")
baseline_model = LogisticRegression(random_state=0, max_iter=1000)
baseline_result = train_and_evaluate_model(baseline_model, X_train_scaled, y_train, X_test_scaled, y_test, 'Baseline')
y_prob_baseline = baseline_model.predict_proba(X_test_scaled)[:, 1]

print("\n=== Базовая модель на сбалансированных данных ===")
print("\n=== Baseline model on balanced data ===")
balanced_baseline_model = LogisticRegression(random_state=0, max_iter=1000, class_weight='balanced')
balanced_result = train_and_evaluate_model(balanced_baseline_model, X_train_scaled, y_train, X_test_scaled, y_test, 'Baseline', 'Class Weight')
y_prob_balanced = balanced_baseline_model.predict_proba(X_test_scaled)[:, 1]

# Создание списка результатов / Creating results list
results = [baseline_result, balanced_result]

# Обучение и оценка моделей с разными методами балансировки / Training and evaluating models with different balancing methods
for model_name, model in models.items():
    print(f"\n=== Модель: {model_name} ===")
    print(f"\n=== Model: {model_name} ===")
    
    for sampling_name, sampler in sampling_methods.items():
        try:
            # Применение метода балансировки / Applying balancing method
            X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
            
            # Обучение и оценка модели / Training and evaluating model
            result = train_and_evaluate_model(model, X_resampled, y_resampled, X_test_scaled, y_test, model_name, sampling_name)
            if result:
                results.append(result)
                
        except Exception as e:
            print(f"Ошибка при применении метода {sampling_name}: {str(e)}")
            print(f"Error applying method {sampling_name}: {str(e)}")
            continue

# Создание DataFrame с результатами / Creating DataFrame with results
results_df = pd.DataFrame(results)

# Сортировка результатов по ROC AUC / Sorting results by ROC AUC
results_df = results_df.sort_values('roc_auc', ascending=False)

# Вывод результатов в табличном виде / Output results in tabular form
print("\nСравнение результатов всех моделей:")
print("\nComparison of all models results:")
print(results_df.sort_values('roc_auc', ascending=False))

# Определение лучшей модели / Determining the best model
best_result = results_df.loc[results_df['roc_auc'].idxmax()]
print(f"\nЛучшая модель: {best_result['model']} с методом балансировки {best_result['sampling']}")
print(f"\nBest model: {best_result['model']} with balancing method {best_result['sampling']}")
print(f"ROC AUC: {best_result['roc_auc']:.3f}")
print(f"Precision: {best_result['precision']:.3f}")
print(f"Recall: {best_result['recall']:.3f}")
print(f"F1-score: {best_result['f1']:.3f}")

# Использование лучшей модели / Using the best model
if best_result['sampling'] == 'Class Weight':
    # Если лучшая модель - с балансировкой через class_weight / If best model uses class_weight balancing
    best_model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=0, max_iter=1000, class_weight='balanced'))
    ])
    best_model_pipeline.fit(X_train, y_train)
else:
    # Если лучшая модель - с другим методом балансировки / If best model uses other balancing method
    best_sampler = sampling_methods[best_result['sampling']]
    X_resampled, y_resampled = best_sampler.fit_resample(X_train_scaled, y_train)
    best_model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', models[best_result['model']])
    ])
    best_model_pipeline.fit(X_resampled, y_resampled)

# Оценка финальной модели / Evaluating final model
y_pred_final = best_model_pipeline.predict(X_test_scaled)
y_prob_final = best_model_pipeline.predict_proba(X_test_scaled)[:, 1]

print("\nОтчет о классификации для финальной модели:")
print("\nClassification report for final model:")
print(classification_report(y_test, y_pred_final, zero_division=0))

# Матрица ошибок / Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_final), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Неудача / Fail', 'Успех / Pass'],
            yticklabels=['Неудача / Fail', 'Успех / Pass'])
plt.title('Матрица ошибок / Confusion Matrix', fontsize=14, pad=20)
plt.xlabel('Предсказанный класс / Predicted class', fontsize=12)
plt.ylabel('Истинный класс / True class', fontsize=12)
plt.tight_layout()
plt.show()

# Сохранение финальной модели / Saving final model
print("\nСохранение финальной модели...")
print("\nSaving final model...")
joblib.dump(best_model_pipeline, 'final_pipeline.pkl')

# Визуализация ROC-кривых для каждой модели / Visualization of ROC curves for each model
print("\nВизуализация ROC-кривых для каждой модели...")
print("\nVisualizing ROC curves for each model...")

# Создаем DataFrame для хранения AUC значений / Creating DataFrame to store AUC values
auc_results = []

# Добавляем результаты базовых моделей / Adding baseline models results
auc_results.append({
    'model': 'Baseline',
    'sampling': 'None',
    'auc': baseline_result['roc_auc']
})

auc_results.append({
    'model': 'Baseline',
    'sampling': 'Class Weight',
    'auc': balanced_result['roc_auc']
})

# Добавляем результаты для каждой модели и метода балансировки / Adding results for each model and balancing method
for model_name, model in models.items():
    plt.figure(figsize=(10, 6))
    
    # ROC-кривая для базовой модели / ROC curve for baseline model
    fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_prob_baseline)
    roc_auc_baseline = auc(fpr_baseline, tpr_baseline)
    plt.plot(fpr_baseline, tpr_baseline, 'b-', 
             label=f'Baseline (AUC = {roc_auc_baseline:.3f})')
    
    # ROC-кривая для сбалансированной базовой модели / ROC curve for balanced baseline model
    fpr_balanced, tpr_balanced, _ = roc_curve(y_test, y_prob_balanced)
    roc_auc_balanced = auc(fpr_balanced, tpr_balanced)
    plt.plot(fpr_balanced, tpr_balanced, 'g-', 
             label=f'Balanced Baseline (AUC = {roc_auc_balanced:.3f})')
    
    # ROC-кривые для каждого метода балансировки / ROC curves for each balancing method
    for sampling_name, sampler in sampling_methods.items():
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
            model.fit(X_resampled, y_resampled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{sampling_name} (AUC = {roc_auc:.3f})')
            
            auc_results.append({
                'model': model_name,
                'sampling': sampling_name,
                'auc': roc_auc
            })
        except Exception as e:
            print(f"Ошибка при построении ROC-кривой для {model_name} с {sampling_name}: {str(e)}")
            print(f"Error plotting ROC curve for {model_name} with {sampling_name}: {str(e)}")
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC-кривые для модели {model_name}\nROC Curves for {model_name} model', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Преобразуем в DataFrame / Convert to DataFrame
auc_df = pd.DataFrame(auc_results)

# Сортируем по значению AUC / Sort by AUC value
auc_df = auc_df.sort_values('auc', ascending=False)

# Создаем визуализацию AUC в виде столбчатой диаграммы / Create AUC visualization as bar plot
plt.figure(figsize=(15, 8))
bars = plt.bar(auc_df['model'] + ' - ' + auc_df['sampling'], auc_df['auc'], color='#66b3ff')

# Добавляем значения в DataFrame / Add values to DataFrame
auc_df['AUC Value'] = auc_df['auc'].apply(lambda x: f'{x:.3f}')
auc_df['AUC Percentage'] = auc_df['auc'].apply(lambda x: f'{x*100:.1f}%')

plt.title('Сравнение ROC AUC для всех моделей и методов балансировки\nComparison of ROC AUC for all models and balancing methods', fontsize=14)
plt.xlabel('Модель и метод балансировки / Model and balancing method', fontsize=12)
plt.ylabel('ROC AUC', fontsize=12)
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Выводим DataFrame с дополнительными столбцами / Display DataFrame with additional columns
print("\nРезультаты AUC для всех моделей:")
print("\nAUC results for all models:")
print(auc_df[['model', 'sampling', 'AUC Value', 'AUC Percentage']])

# Установка стиля для графиков / Setting style for plots
plt.style.use('ggplot')  # Используем стиль ggplot вместо seaborn 

# Экспорт необходимых переменных для использования в других файлах
__all__ = ['df_filtered', 'X', 'y', 'results_df', 'best_model_pipeline'] 