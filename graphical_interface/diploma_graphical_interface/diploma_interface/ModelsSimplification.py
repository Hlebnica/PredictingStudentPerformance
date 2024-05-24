import pandas as pd
import numpy as np

class ModelsSimplification:
    @staticmethod
    def decile_sum_rating_students_by_semestr(merged_df: pd.DataFrame, term_number: int):
        """Дециль суммы рейтингов студента по всем дисциплинам номера(term_number) семестра

        Args:
            merged_df (dataframe): объединенный dataframe 
            term_number (int): номер курса

        Returns:
            merged_df: dataframe с новым столбцом 
        """
        # отбор только данных для term_number семестра
        df_first_semester = merged_df[merged_df['Term'] == term_number]
        # группировка по студентам и суммирование их рейтингов
        student_grouped_by_sem = df_first_semester.groupby('student_id')['Rating'].sum().reset_index()
        # сортировка по сумме рейтингов и вычисление децилей
        student_grouped_by_sem = student_grouped_by_sem.sort_values(by='Rating', ascending=False)
        labels = np.linspace(2, 5, 5)
        student_grouped_by_sem[f'DecileSumRatingStudent{term_number}Sem'] = pd.qcut(student_grouped_by_sem['Rating'], 5, labels=labels, duplicates='drop')

        # объединяем результаты с исходным DataFrame
        merged_df = pd.merge(merged_df, student_grouped_by_sem[['student_id', f'DecileSumRatingStudent{term_number}Sem']], on='student_id', how='inner')
        return merged_df
     
    @staticmethod
    def decile_median_rating_disciplin_check_type_by_semestr(merged_df: pd.DataFrame, check_type: int, term_number: int):
        """Дециль медианного рейтинга дисциплин зачетов/экзаменов(check_type) номера(term_number) семестра

        Args:
            merged_df (dataframe): объединенный dataframe 
            check_type (int): тип дисциплины 1 - экзамен / 0 - зачет
            term_number (int): номер семестра

        Returns:
            merged_df: dataframe с новым столбцом 
        """

        # создаем отдельный dataframe для дисциплин check_type term_number семестра
        credit_df= merged_df[(merged_df['CheckType'] == check_type) & (merged_df['Term'] == term_number)]

        # группируем по id студента и находим медианный рейтинг для каждой группы
        credit_median_ratings = credit_df.groupby('student_id')['Rating'].median()

        # находим дециль медианного рейтинга для каждого студента
        labels = np.linspace(2, 5, 5)
        credit_decile_ratings = pd.qcut(credit_median_ratings, 5, labels=labels, duplicates='drop')

        # создаем новый DataFrame для столбцов "student_id" и "DecileMedianRatingCredits1Sem"
        if check_type == 0:
            credit_decile_df = pd.DataFrame({'student_id': credit_decile_ratings.index, f'DecileMedianRatingCredits{term_number}Sem': credit_decile_ratings.values})
        else:
            credit_decile_df = pd.DataFrame({'student_id': credit_decile_ratings.index, f'DecileMedianRatingExam{term_number}Sem': credit_decile_ratings.values})

        # объединяем новый DataFrame с исходным по столбцу "student_id"
        merged_df = merged_df.merge(credit_decile_df, on='student_id', how='inner')
        
        return merged_df
