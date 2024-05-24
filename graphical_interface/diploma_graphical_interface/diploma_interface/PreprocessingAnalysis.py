import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.cluster import DBSCAN

class PreprocessingAnalysis:
    
    @staticmethod
    def data_synthesis(model):
        # подготовка входных данных
        X = model.drop(model.columns[-1], axis=1)
        y = model[model.columns[-1]]

        # разделение данных на обучающую и тестовую выборки
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # создание объекта ADASYN
        oversampler = ADASYN(random_state=42, sampling_strategy='minority')

        # применение ADASYN к обучающим данным
        X_res, y_res = oversampler.fit_resample(X_train, y_train)

        dataframe_res = pd.concat([X_res, y_res], axis=1)
        
        return dataframe_res
    
    
    @staticmethod
    def emission_removal(dataframe_model, column_for_learning):
        # Поиск межквартильного размаха
        Q1 = dataframe_model[column_for_learning].quantile(0.25)
        Q3 = dataframe_model[column_for_learning].quantile(0.75)
        IQR = Q3 - Q1

        # Определение границ выбросов
        lower_bound = Q1 - 0.5 * IQR
        upper_bound = Q3 + 0.5 * IQR

        # Удаление выбросов
        dataframe_model = dataframe_model[(dataframe_model[column_for_learning] >= lower_bound) & (dataframe_model[column_for_learning] <= upper_bound)]
        
        return dataframe_model
    
    
    @staticmethod
    def remove_noise_with_dbscan(dataframe_model, eps=0.5, min_samples=5):
        """Удаление шума с помощью DBSCAN

        Args:
            dataframe_model (dataframe): dataframe с обработанной моделью
            eps (float): максимальное расстояние между двумя образцами, чтобы они считались в одном кластере
            min_samples (int): минимальное количество образцов в кластере

        Returns:
            dataframe_model: dataframe без шумовых точек
        """
        # Применение DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(dataframe_model)
        labels = db.labels_

        # Булев массив, где True означает, что точка является шумом
        is_noise = labels == -1

        # Удаление шумовых точек
        dataframe_model = dataframe_model[~is_noise]
        
        return dataframe_model