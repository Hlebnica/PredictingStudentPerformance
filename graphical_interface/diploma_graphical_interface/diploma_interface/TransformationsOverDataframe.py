from .DataClear import DataClear
import pandas as pd

class TransformationsOverDataframe:

    @staticmethod
    def ege_marks_transpose(ege_marks_df):
        """Преобразование строк предметов в столбцы для GetEgeMarksByStudent
        
            Args:
                ege_marks_df (dataframe): dataframe c результатами ЕГЭ
                
            Returns:
                ege_marks_df: dataframe c результатами ЕГЭ разбитыми на столбцы
            
        """
        # Оставить только необходимые вступительные
        ege_marks_df = DataClear.keep_rows_in_journal(ege_marks_df,
                                                         Subject=['Математика', 'Информатика и ИКТ',
                                                                  'Русский язык', 'Физика', 'Биология',
                                                                  'Химия', 'Обществознание', 'История',
                                                                  'Основы экономики', 'Рисунок и натюрморт',
                                                                  'Основы здорового образа жизни', 'Сочинение',
                                                                  'Творческий конкурс', 'Общая физическая подготовка',
                                                                  'Иностранный язык'
                                                                  ])

        # Убрать дубликаты
        ege_marks_df = ege_marks_df.drop_duplicates(subset=['student_id', 'Subject'], keep='first')
        
        # Ограничиваем количество строк до трех для каждого студента
        ege_marks_df = ege_marks_df[ege_marks_df.groupby('student_id').cumcount() < 3]

        # Разбиение по столбцам вступительных баллов
        ege_marks_df = ege_marks_df.pivot_table(index='student_id',
                                                columns=ege_marks_df.groupby('student_id').cumcount() + 1,
                                                values=['Subject', 'Mark'], aggfunc='first')

        # переименовываем столбцы
        ege_marks_df.columns = [f'{col}{num}' for col, num in ege_marks_df.columns]

        # удаляем мультииндекс и сбрасываем индекс
        ege_marks_df = ege_marks_df.reset_index()
        ege_marks_df.columns = ege_marks_df.columns.map(''.join)

        # Убрать пустые строки
        ege_marks_df.dropna(inplace=True)

        return ege_marks_df