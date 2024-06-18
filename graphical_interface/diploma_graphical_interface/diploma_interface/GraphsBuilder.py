import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

class GraphsBuilder:
    def __init__(self, clf, X, y_test, y_pred):
        self.clf = clf
        self.X = X
        self.y_test = y_test
        self.y_pred = y_pred

    def feature_importance_graph(self):
        """График важности признаков

        Returns:
            feature_importance_graph: график важности признаков
        """
        
        figsize=(12,6)
        
        y_axis_labels = {
            'CheckType': 'Тип контроля',
            'CourseEduId' : 'Id в EduSusu',
            'CourseNumber' : 'Номер курса',
            'DirectionCode' : 'Код специальности',
            'DirectionName': 'Название специальности',
            'Id': 'Id в "Универис"',
            'IsPractice': 'Является ли практикой',
            'Speciality': 'Квалификация',
            'StudyForm': 'Форма обучения',
            'SubjectName': 'Название предмета',
            'Term': 'Семестр',
            'Year': 'Год',
            'EnrollScore': 'Вступительные баллы',
            'FinancialForm': 'Форма финансирования\nобучения',
            'HasOlympParticipation': 'Участие в олипиаде',
            'LiveCity': 'Город проживания',
            'RegisterCity': 'Город регистрации',
            'Sex': 'Пол',
            'Status': 'Статус обучения',
            'Mark': 'Полученная оценка',
            'Rating': 'Рейтинг по БРС',
            'StudentId': 'Id студента\nв “Универис”',
            'journal_id': 'Id записи',
            'student_id': 'Id студента',
            'Mark1': 'Баллы 1 предмет',
            'Mark2': 'Баллы 2 предмет',
            'Mark3': 'Баллы 3 предмет',
            'Subject1': '1 предмет',
            'Subject2': '2 предмет',
            'Subject3': '3 предмет', 
            'FinancialForm_budget_share_decile': 'Квантиль бюджетников\nв группе',
            'EgeMark1': '1 дециль\nоценки за ЕГЭ',
            'EgeMark2': '2 дециль\nоценки за ЕГЭ',
            'EgeMark3': '3 дециль\nоценки за ЕГЭ',
            'Predicted': 'Предсказанный результат',
            'AvgScoreFinalRating': 'Средний рейтинг\nза период обучения',
        }
        
        for i in range(1, 9):
            y_axis_labels[f'Mark_Term_{i}'] = f'Оценка за {i} семестр'
            y_axis_labels[f'DecileSumRatingStudent{i}Sem'] = f'Дециль суммарного рейтинга\nза {i} семестр'
            y_axis_labels[f'DecileMedianRatingCredits{i}Sem'] = f'Дециль медианного рейтинга\nпо зачетам за {i} семестр'
            y_axis_labels[f'DecileMedianRatingExam{i}Sem'] = f'Дециль медианного рейтинга\nпо экзаменам за {i} семестр'
            
        importance = self.clf.feature_importances_
        feat_importances = pd.Series(importance, index=self.X.columns)
        
        fig = plt.figure(figsize=figsize) 
        ax = fig.add_subplot(111)
        feat_importances.nlargest(10).plot(kind='barh', ax=ax) 
        
        # Переименовывание столбцов
        ax.set_yticklabels([y_axis_labels.get(label, label) for label in feat_importances.nlargest(10).index])
        
        # Увеличение размера текста подписей
        plt.tick_params(axis='both', which='major', labelsize=14)
        
        # Увеличение размера текста подписей осей
        plt.yticks(fontsize=14)
        
        # Сохранение графика в файл
        plt.savefig('feature_importance_graph.png')
        
        # Преобразование графика в кодировку base64
        fig = plt.gcf()
        plt.subplots_adjust(left=0.3)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        feature_importance_graph = base64.b64encode(image_png)
        feature_importance_graph = feature_importance_graph.decode('utf-8')
        buffer.close()
    
        return feature_importance_graph
    
    def plot_predictions(self):
        """График предсказаний и реальных значений

        Args:
            y_test: данные прогнозируемого столбца для тестирования из DataAnalysis
            y_pred: предсказания по обученной модели из DataAnalysis
            figsize (int, int): размер графика. По умолчанию (12,6).

        Returns:
            plot_predictions_graph: график предсказаний и реальных значений
        """
        figsize=(18,6)
        fig = plt.figure(figsize=figsize)

        # Построение гистограммы
        plt.hist(self.y_test.values, alpha=0.5, label='Истинные значения', color='blue')
        plt.hist(self.y_pred, alpha=0.5, label='Предсказанные значения', color='orange')

        # Установка целочисленных значений для оси x
        plt.xticks(np.arange(min(self.y_test.values), max(self.y_test.values)+1, 1.0))

        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.subplots_adjust(left=0.2, right=0.8)
        
        # Увеличение размера текста подписей
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        # Увеличение размера текста легенды
        plt.legend(fontsize=16)

        # Сохранение графика в файл
        plt.savefig('plot_predictions_hist.png')

        # Преобразование графика в кодировку base64
        fig = plt.gcf()
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        plot_predictions_hist = base64.b64encode(image_png)
        plot_predictions_hist = plot_predictions_hist.decode('utf-8')
        buffer.close()

        return plot_predictions_hist

    

    def plot_confusion_matrix(self):
        # Вычисление матрицы ошибок
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        figsize=(10,7)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Построение графика матрицы ошибок
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, xticklabels=self.clf.classes_, yticklabels=self.clf.classes_)
        plt.xlabel('Предсказанные классы', fontsize=16)
        plt.ylabel('Истинные классы', fontsize=16)
        
        # Увеличение размера текста подписей
        plt.tick_params(axis='both', which='major', labelsize=14)
        
        # Сохранение графика в файл
        plt.savefig('confusion_matrix.png')
        
        # Преобразование графика в кодировку base64
        fig = plt.gcf()
        plt.subplots_adjust(left=0.2)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        confusion_matrix_graph = base64.b64encode(image_png)
        confusion_matrix_graph = confusion_matrix_graph.decode('utf-8')
        buffer.close()
        
        return confusion_matrix_graph