from .DataClear import DataClear
from .ModelsSimplification import ModelsSimplification
import pandas as pd
import numpy as np
import math


class PredictiveModels:
    def __init__(self, students_data, ege_marks_data, rating_data, by_year_data):
        self.students_data = students_data
        self.ege_marks_data = ege_marks_data
        self.rating_data = rating_data
        self.by_year_data = by_year_data

    def _prepare_student_data(self):
        # Объединение данных студентов и оценок ЕГЭ
        merged_data = pd.merge(self.students_data, self.ege_marks_data, left_on='Id', right_on='student_id')
        merged_data['Sex'] = merged_data['Sex'].map({'Мужской': 1, 'Женский': 0})
        merged_data['RegisterCity'] = merged_data['RegisterCity'].apply(lambda x: 1 if isinstance(x, str) and x.replace(" ", "") == "г.Челябинск" or x == 1 else 0)
        merged_data['FinancialForm'] = merged_data['FinancialForm'].map({'бюджет': 1, 'контракт': 0})
        merged_data[['EgeMark1', 'EgeMark2', 'EgeMark3']] = merged_data[['Mark1', 'Mark2', 'Mark3']].apply(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1)
        # Удаление дубликатов студентов
        merged_data = merged_data.drop_duplicates(subset=['student_id'])
        # Выбор нужных столбцов
        merged_data = merged_data[['student_id', 'Sex', 'EgeMark1', 'EgeMark2', 'EgeMark3', 'RegisterCity', 'FinancialForm', 'Status']]
        
        # Объединение данных студентов с рейтингом и журналами по годам
        merged_data = pd.merge(merged_data, self.rating_data, left_on='student_id', right_on='StudentId')
        merged_data = pd.merge(merged_data, self.by_year_data, left_on='journal_id', right_on='Id')
        
        # Приведение типов значений в необходимых столбцах к числам
        check_type_mapping = {'зачет': 0, 'дифференцированный зачет': 1, 'экзамен': 1,
                              'курсовые работы': 3, 'курсовые проекты': 4, 'практика': 5}
        merged_data.loc[:, 'CheckType'] = merged_data['CheckType'].map(check_type_mapping)
        
        status_mapping = {'учится': 1, 'закончил': 1, 'отчислен': 0}
        merged_data.loc[:, 'Status'] = merged_data['Status'].map(status_mapping)
        merged_data = merged_data[merged_data['Status'] != 'в академе']

        return merged_data
    
    
    def _decil_median_data(self, merged_data_df: pd.DataFrame, columns_for_keep: list, term_from: int, term_to: int,
                            check_type: int = 123, decile_median_rating_disciplin_check_type: bool = False):
            
        for n in range(term_from, term_to + 1):
            merged_data_df = ModelsSimplification.decile_sum_rating_students_by_semestr(merged_data_df, n)
            columns_for_keep.append(f'DecileSumRatingStudent{n}Sem')

            if decile_median_rating_disciplin_check_type:
                merged_data_df = DataClear.drop_rows_in_journal(merged_data_df, CheckType=[1 - check_type])

                merged_data_df = ModelsSimplification.decile_median_rating_disciplin_check_type_by_semestr(merged_data_df, check_type, n)
                columns_for_keep.append(f'DecileMedianRating{"Credits" if check_type == 0 else "Exam"}{n}Sem')
                    
        return merged_data_df, columns_for_keep
    
    
    def _get_columns_for_keep(self, include_student_id: bool) -> list:
        """Список названий сохраняемых столбцов

        Args:
            include_student_id (bool): Добавлять в начало списка атрибут 'student_id'

        Returns:
            list: список названий столбцов
        """
        columns_for_keep = ['Sex', 'EgeMark1', 'EgeMark2', 'EgeMark3', 'RegisterCity', 'FinancialForm']
        if include_student_id:
            columns_for_keep.insert(0, 'student_id')
        return columns_for_keep
    
    
    def _convert_to_int(self, df: pd.DataFrame, include_student_id: bool) -> pd.DataFrame:
        """Конвертирование столбцов в int

        Args:
            df (pd.DataFrame): датафрейм merged_df
            include_student_id (bool): Присутствует ли атрибут 'student_id'

        Returns:
            pd.DataFrame: датафрейм с приведенными к int столбцами
        """
        if include_student_id:
            cols = df.columns.drop('student_id')
        else:
            cols = df.columns
        df[cols] = df[cols].astype(int)
        return df

    
    def model_subjectName(self, subject_name: str, term_from: str, term_to: str, add_decile_sum_rating_student: bool,
                          include_student_id: bool = False) -> pd.DataFrame:
        """models 1, 2, 3

        Args:
            subject_name (str): название предмета
            term_from (str): семестр от
            term_to (str): семестр до
            add_decile_sum_rating_student (bool): добавлять дециль суммарного рейтинга

        Returns:
            pd.DataFrame: dataframe с сформированной моделью
        """
        # Подготовка данных студентов
        merged_data = self._prepare_student_data()
        
        # Создание списка всех семестров в указанном диапазоне
        terms_list = list(range(term_from, term_to + 1))
        
        # Выбор нужных столбцов
        columns_for_keep = self._get_columns_for_keep(include_student_id)

        # Добавление дециля рейтинга студента по семестру 
        if add_decile_sum_rating_student:
            deciels_data = self._decil_median_data(merged_data, columns_for_keep, 1, term_to)
            merged_data = deciels_data[0]
            columns_for_keep = deciels_data[1]

        # Фильтрация данных по предмету
        merged_data = merged_data[merged_data['SubjectName'] == subject_name]
        # Фильтрация данных по диапазону семестров
        merged_data = merged_data[merged_data['Term'].isin(terms_list)]

        # Создание новых столбцов с оценками для каждого семестра
        for term in terms_list:
            term_columns = merged_data[merged_data['Term'] == term].pivot_table(index='student_id', columns='Term', values='Mark', aggfunc='first', fill_value=0)
            term_columns.columns = [f'Mark_Term_{term}' for _ in term_columns.columns]
            merged_data = pd.merge(merged_data, term_columns, how='left', on='student_id')
            columns_for_keep.append(f'Mark_Term_{term}')
        
        merged_data.drop_duplicates(subset=["student_id"], keep="first", inplace=True)  
            
        merged_data = merged_data[columns_for_keep]
        merged_data = merged_data.dropna(axis=0)
        merged_data = self._convert_to_int(merged_data, include_student_id)
        
        return merged_data
    
    
    def model_students_rating_status(self, term_from: int, term_to: int,
                                        check_type: int, add_status_column: bool,
                                        include_student_id: bool = False) -> pd.DataFrame:
        """models 4, 5

        Args:
            term_from (int): семестр от
            term_to (int): семестр до
            check_type (int): тип контроля предмета
            add_status_column (bool): добавлять колонку статуса

        Returns:
            pd.DataFrame: dataframe с сформированной моделью
        """
        # Подготовка данных студентов
        merged_data = self._prepare_student_data()

        # Столцбы для сохранения
        columns_for_keep = self._get_columns_for_keep(include_student_id)

        deciels_data = self._decil_median_data(merged_data, columns_for_keep, term_from, term_to, check_type, True)
        merged_data = deciels_data[0]
        columns_for_keep = deciels_data[1]

        # Добавление столбца Status если True
        if add_status_column:
            columns_for_keep.append('Status')
        
        merged_data.drop_duplicates(subset=["student_id"], keep="first", inplace=True)
        merged_data = merged_data[columns_for_keep]
        merged_data = merged_data.dropna(axis=0)
        merged_data = self._convert_to_int(merged_data, include_student_id)

        return merged_data
    
    
    def model_avg_final_rating(self, term_from: int, term_to: int, include_student_id: bool = False, is_reuse: bool = False) -> pd.DataFrame:
        """model 6

        Args:
            term_from (int): семестр от
            term_to (int): семестр до

        Returns:
            pd.DataFrame: dataframe с сформированной моделью
        """
        merged_data = self._prepare_student_data()
        
        # Столцбы для сохранения
        columns_for_keep = self._get_columns_for_keep(include_student_id)
        
        if is_reuse:
            deciels_data = self._decil_median_data(merged_data, columns_for_keep, term_from, term_to)
        else:
            deciels_data = self._decil_median_data(merged_data, columns_for_keep, 1, 8)
        merged_data = deciels_data[0]
        columns_for_keep = deciels_data[1]

        
        if is_reuse:
            # Если is_reuse True, создаем столбец AvgScoreFinalRating, полностью заполненный единицами
            merged_data['AvgScoreFinalRating'] = 1
        else:
            if include_student_id:
                student_id_column = merged_data['student_id'] # Сохраненине значений student_id 
                merged_data = merged_data.drop(columns=['student_id'])
            
            # Вычисление среднего значения столбцов DecileSumRatingStudent{n}Sem
            avg_scores = merged_data[[f'DecileSumRatingStudent{n}Sem' for n in range(1, 8 + 1)]].astype('float').mean(axis=1)
            avg_scores_rounded = avg_scores.apply(lambda x: math.ceil(x) if x - math.floor(x) >= 0.5 else math.floor(x))
            merged_data['AvgScoreFinalRating'] = avg_scores_rounded
            
            if include_student_id:
                merged_data.insert(0, 'student_id', student_id_column) # Возврат сохраненного столбца значений student_id
        
        columns_for_keep.append('AvgScoreFinalRating')
        
        merged_data.drop_duplicates(subset=["student_id"], keep="first", inplace=True)
        merged_data = merged_data[columns_for_keep]
        merged_data = merged_data.dropna(axis=0)
        merged_data = self._convert_to_int(merged_data, include_student_id)
        
        # Создание списка столбцов для удаления на основе введенных семестров
        if is_reuse == False:
            columns_to_drop = [f'DecileSumRatingStudent{n}Sem' for n in range(1, 9) if n < term_from or n > term_to]
            merged_data = merged_data.drop(columns=columns_to_drop)

        return merged_data
    
    
    # def model_checkType(self, check_type: int, term_from: str, term_to: str) -> pd.DataFrame:
    #     """model 3

    #     Args:
    #         check_type (int): тип контроля предмета
    #         term_from (str): семестр от
    #         term_to (str): семестр до

    #     Returns:
    #         pd.DataFrame: dataframe с сформированной моделью
    #     """
    #     # Подготовка данных студентов
    #     merged_data = self._prepare_student_data()
    
    #     # Создание списка всех семестров в указанном диапазоне
    #     terms_list = list(range(term_from, term_to + 1))
        
    #     # Выбор нужных столбцов
    #     columns_for_keep = ['Sex', 'EgeMark1', 'EgeMark2', 'EgeMark3', 'RegisterCity', 'FinancialForm']

    #     # Добавление дециля рейтинга студента по семестру 
    #     deciels_data = self._decil_median_data(merged_data, columns_for_keep, 1, term_to)
    #     merged_data = deciels_data[0]
    #     columns_for_keep = deciels_data[1]

    #     # Фильтрация данных по предмету
    #     merged_data = merged_data[merged_data['CheckType'] == check_type]
    #     # Фильтрация данных по диапазону семестров
    #     merged_data = merged_data[merged_data['Term'].isin(terms_list)]

    #     # Создание новых столбцов с оценками для каждого семестра
    #     for term in terms_list:
    #         term_columns = merged_data[merged_data['Term'] == term].pivot_table(index='student_id', columns='Term', values='Mark', aggfunc='first', fill_value=0)
    #         term_columns.columns = [f'Mark_Term_{term}' for _ in term_columns.columns]
    #         merged_data = pd.merge(merged_data, term_columns, how='left', on='student_id')
    #         columns_for_keep.append(f'Mark_Term_{term}')
        
    #     merged_data.drop_duplicates(subset=["student_id"], keep="first", inplace=True)
            
    #     merged_data = merged_data[columns_for_keep]
    #     merged_data = merged_data.dropna(axis=0)
    #     merged_data = merged_data.astype(int)
        
    #     return merged_data