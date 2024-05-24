import sklearn
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import DBSCAN
from .PreprocessingAnalysis import PreprocessingAnalysis


class DataAnalysis:
   
    @staticmethod
    def train_by_model(dataframe_model: pd.DataFrame, model_training_method: str, test_size: float, n_estimators: int,
                       max_depth: int, min_samples_split: int, min_samples_leaf: int, max_features: int,
                       do_data_synthesis: bool = False, do_emission_removal: bool = True, do_noise_remove: bool = False):
        """Обучение по модели

        Args:
            dataframe_model (dataframe): dataframe с обработанной моделью
            model_training_method (str): тип модели для обучения
            test_size (float): размер выборки, который будет предназначен для тестирования (0.1 - 0.9)
            n_estimators (int): сколько деревьев решений необходимо создать в процессе обучения модели
            max_depth (int): максимальная глубина дерева решений
            min_samples_split (int): минимальное количество образцов (наблюдений), которые необходимо иметь в узле, чтобы он мог быть разделен на две ветви. 
            min_samples_leaf (int): минимальное количество объектов, которое должно быть в листовой вершине дерева решений. 
            max_features (int): максимальное количество признаков, которые рассматриваются при поиске наилучшего разделения на каждом узле дерева решений. 
            do_data_synthesis (bool): делать синтез данных с помощью ADASYN
            do_emission_removal (bool): удалять выбросы
            do_noise_remove (bool): удалять шумы

        Returns:
            clf: обученная модель
            classification_quality_assessment: оценка качества обученной аналитической модели
            X: все столбцы кроме column_for_learning для входных данных
            y_test: данные прогнозируемого столбца для тестирования
            y_pred: предсказания по обученной модели
            results_df: dataframe для вывода результатов
        """
        
        training_models = {'RandomForestClassifier': RandomForestClassifier, 'GradientBoostingClassifier': GradientBoostingClassifier}
        
        # Колонка для обучения
        column_for_learning = dataframe_model.columns[-1]
            
        if do_noise_remove:
            dataframe_model = PreprocessingAnalysis.remove_noise_with_dbscan(dataframe_model)
            
        if do_data_synthesis: 
            dataframe_model = PreprocessingAnalysis.data_synthesis(dataframe_model)
            
        if do_emission_removal:
            dataframe_model = PreprocessingAnalysis.emission_removal(dataframe_model, column_for_learning)
        
        # сохранить все столбцы кроме column_for_learning для входных данных
        X = dataframe_model.drop(column_for_learning, axis=1)
        # задать column_for_learning как столбец выходных данных (прогнозируемый)
        y = dataframe_model[column_for_learning]
        y = y.astype('int')
        # разбиение на данные для обучения и тестирования
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=test_size, random_state=42)

        # Вычисление весов примеров
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        # обучение модели
        clf = training_models[model_training_method](n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)
        clf.fit(X_train, y_train, sample_weight=sample_weights)

        # сделать предсказания по обученной модели
        y_pred = clf.predict(X_test)
        
        # Создание DataFrame для вывода результатов
        results_df = pd.DataFrame(X_test, columns=X.columns)
        results_df[column_for_learning] = y_test
        results_df['Predicted'] = y_pred

        # оценки классификации precision, recall, fscore
        acc = sklearn.metrics.accuracy_score(y_test, y_pred)
        precision_score = sklearn.metrics.precision_score(
            y_test, y_pred, average='macro', zero_division=0)
        recall_score = sklearn.metrics.recall_score(
            y_test, y_pred, average='macro')
        f1_score = sklearn.metrics.f1_score(y_test, y_pred, average='macro')

        classification_quality_assessment = (
            f"Accuracy оценка: {acc:.2f}\n" +
            f"Precision оценка: {precision_score:.2f}\n" +
            f"Recall оценка: {recall_score:.2f}\n" +
            f"F1 оценка: {f1_score:.2f}\n"
        )

        return clf, classification_quality_assessment, X, y_test, y_pred, results_df