import pandas as pd
import joblib
import os

class AssistantForInterface:
    
    @staticmethod
    def get_subjects_intersection(df: pd.DataFrame, start_semester: int, end_semester: int, make_only_partice: bool):
        """Получить пересечение предметов, которые проходили с start_semester по end_semester

        Args:
            df (pd.DataFrame): dataframe с журналами по годам
            start_semester (int): семестр с которого нужно выбрать предметы 
            end_semester (int): семестр до которого нужно выбрать предметы

        Returns:
            List: Список из предметов, которые проходили с start_semester по end_semester
        """
        
        common_subjects = None

        for semester in range(start_semester, end_semester + 1):
            semester_subjects = set(df.loc[df['Term'] == semester, 'SubjectName'].dropna())
            if common_subjects is None:
                common_subjects = semester_subjects
            else:
                common_subjects = common_subjects.intersection(semester_subjects)

        if common_subjects is None:
            return []
        
        if make_only_partice:
            common_subjects = [subj for subj in common_subjects if 'практика' in subj.lower()]
        else:
            common_subjects = [subj for subj in common_subjects if 'практика' not in subj.lower()]
        
        return sorted(common_subjects)
    
    
    @staticmethod
    def get_subjects_in_term(df: pd.DataFrame, term_number: int):
        """Получить список предметов, которые проходили в указанном семестре

        Args:
            df (pd.DataFrame): dataframe с журналами по годам
            term_number (int): номер семестра

        Returns:
            List: список предметов, которые проходили в указанном семестре
        """
        
        return df['SubjectName'].where(df['Term'] == term_number).dropna().unique().tolist()
    
    @staticmethod
    def save_model(model, filename):
        """_summary_

        Args:
            model (_type_): _description_
            filename (_type_): _description_
        """
        if not filename.endswith('.joblib'):
            filename += '.joblib'
        joblib.dump(model, filename)
    
    @staticmethod
    def load_model(filename):
        """_summary_

        Args:
            filename (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if not filename.endswith('.joblib'):
            raise ValueError("Неверное расширение файла. Ожидается расширение .joblib.")
        return joblib.load(filename)


def get_students_id_intersection(dfStudentsByJournalId: pd.DataFrame, dfSamplingYearsData: pd.DataFrame, start_semester: int, end_semester: int):
    """Получить пересечение student_id, которые проходили с start_semester по end_semester

    Args:
        dfStudentsByJournalId (pd.DataFrame): dataframe с журналами студентов
        dfSamplingYearsData (pd.DataFrame): dataframe с журналами по годам
        start_semester (int): семестр с которого нужно выбрать группы 
        end_semester (int): семестр до которого нужно выбрать группы 

    Returns:
        List: Список id студентов
    """
    
    merged_df = pd.merge(dfStudentsByJournalId, dfSamplingYearsData, left_on='journal_id', right_on='Id')
    student_ids = None

    for semester in range(start_semester, end_semester + 1):
        student_ids_in_term = set(merged_df.loc[merged_df['Term'] == semester, 'Id_x'].dropna())
        if student_ids is None:
            student_ids = student_ids_in_term
        else:
            student_ids = student_ids.intersection(student_ids_in_term)

    if student_ids is None:
        return []
    
    
    return sorted(student_ids)


def intersection_students_by_model_students_union(model_dataframe: pd.DataFrame, students_union: list) -> list:
    """Пересечение id студентов в диапазоне семестров и находящихся в передаваемой модели

    Args:
        model_dataframe (pd.DataFrame): _description_
        students_union (list): _description_

    Returns:
        list: Список id студентов
    """
    intersection = model_dataframe[model_dataframe['student_id'].isin(students_union)]
    intersection_list = intersection['student_id'].tolist()
    return intersection_list