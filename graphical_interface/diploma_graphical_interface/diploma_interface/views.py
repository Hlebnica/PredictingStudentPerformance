from django.shortcuts import render

import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import re
import asyncio
import pandas as pd
import json
from django.http import JsonResponse, HttpResponse
import joblib
import os
from django.conf import settings
from urllib.parse import quote

from .AsyncRequests import AsyncRequests
from .DataClear import DataClear
from .TransformationsOverDataframe import TransformationsOverDataframe
from .AssistantForInterface import AssistantForInterface
from .PredictiveModels import PredictiveModels
from .DataAnalysis import DataAnalysis
from .GraphsBuilder import GraphsBuilder

from django.contrib import messages

dfByYear = None
dfStudentsByJournalId = None
dfRatingByJournalId = None
dfEgeMarksByStudentsId = None
analysis_model = None
institutes_list_item = None
year_from = None
year_to = None


def rename_dataframe_columns(df):
    column_names_dict = {
        'GroupId': 'Id группы',
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
        'FinancialForm': 'Форма финансирования обучения',
        'HasOlympParticipation': 'Участие в олипиаде',
        'LiveCity': 'Город проживания',
        'RegisterCity': 'Город регистрации',
        'Sex': 'Пол',
        'Status': 'Статус обучения',
        'Mark': 'Полученная оценка',
        'Rating': 'Рейтинг по БРС',
        'StudentId': 'Id студента в “Универис”',
        'journal_id': 'Id записи',
        'student_id': 'Id студента',
        'Mark1': 'Баллы 1 предмет',
        'Mark2': 'Баллы 2 предмет',
        'Mark3': 'Баллы 3 предмет',
        'Subject1': '1 предмет',
        'Subject2': '2 предмет',
        'Subject3': '3 предмет', 
        'FinancialForm_budget_share_decile': 'Квантиль бюджетников в группе',
        'EgeMark1': '1 дециль оценки за ЕГЭ',
        'EgeMark2': '2 дециль оценки за ЕГЭ',
        'EgeMark3': '3 дециль оценки за ЕГЭ',
        'MarkSecondSem ': 'Полученная оценка за 2 семестр',
        'MarkFirstSem': 'Дециль оценки за 1 семестр',
        'MarkSecondSem': 'Дециль оценки за 2 семестр',
        'Predicted': 'Предсказанный результат',
        'AvgScoreFinalRating': 'Средний рейтинг за период обучения',
    }
    for i in range(1, 9):
        column_names_dict[f'Mark_Term_{i}'] = f'Оценка за {i} семестр'
        column_names_dict[f'DecileSumRatingStudent{i}Sem'] = f'Дециль суммарного рейтинга за {i} семестр'
        column_names_dict[f'DecileMedianRatingCredits{i}Sem'] = f'Дециль медианного рейтинга по зачетам за {i} семестр'
        column_names_dict[f'DecileMedianRatingExam{i}Sem'] = f'Дециль медианного рейтинга по экзаменам за {i} семестр'
        
    return df.rename(columns=column_names_dict)
     


# Преобразование dataframe в вид для вывода на html страницу
def df_to_html(df):
    """Преобразование dataframe в html

    Args:
        df (dataframe): dataframe, который необходимо перевести в html

    Returns:
        html: dataframe преобразованный в html
    """
    
    df_renamed = rename_dataframe_columns(df)
    df_html = df_renamed.to_html(classes='table table-bordered scrollable', index=False)
    return df_html


# Получение данных из API
def index(request):
    
    global dfByYear 
    global dfStudentsByJournalId
    global dfRatingByJournalId
    global dfEgeMarksByStudentsId
    global institutes_list_item
    global year_from
    global year_to
    
    institutes_list = ["Высшая школа электроники и компьютерных наук", "Архитектурно-строительный институт", "Высшая медико-биологическая школа", "Высшая школа экономики и управления", 
                    "Институт естественных и точных наук", "Институт лингвистики и международных коммуникаций", "Институт медиа и социально-гуманитарных наук", 
                    "Институт спорта, туризма, сервиса", "Политехнический институт", "Юридический институт", "Филиалы ЮУрГУ"]
    
    if request.method == 'POST':
        try:
            year_from = request.POST.get('year_from')
            year_to = request.POST.get('year_to')
            range_from = request.POST.get('range_from')
            range_to = request.POST.get('range_to')
            institutes_list_item = request.POST.get('institutes_list_item')
                    
            institutes_copy = institutes_list.copy()
            institutes_copy.remove(institutes_list_item)
            flat_excluded_institutes = DataClear.institutional_exclusion(*institutes_copy)
            
            # Года "от" и "до"
            years_list = []
            for i in range(int(year_from), int(year_to) + 1):
                years_list.append(i)
            
            dfByYear = asyncio.run(AsyncRequests.journal_by_years(years_list))
            
            # Дипазон курсов для сохранения 
            course_range = [1, 2, 3, 4, 5, 6]
            filtered_course_range = [x for x in course_range if x < int(range_from) or x > int(range_to)]
            
            dfByYear[0] = DataClear.drop_rows_in_journal(
                dfByYear[0], 
                Speciality=['магистр', 'аспирант', 'специалист'], 
                StudyForm=['заочная', 'очно-заочная'],
                CourseNumber=filtered_course_range,
                DirectionCode=flat_excluded_institutes)

            # Все Учебные практики и Производственные практики к единому названию
            dfByYear[0]['SubjectName'] = dfByYear[0]['SubjectName'].apply(lambda x: re.sub(r'.*Производственная.*практика.*', 'Производственная практика', x))
            dfByYear[0]['SubjectName'] = dfByYear[0]['SubjectName'].apply(lambda x: re.sub(r'.*Учебная.*практика.*', 'Учебная практика', x))
            
            # Запрос студентов по id за N год
            dfStudentsByJournalId = asyncio.run(AsyncRequests.students_by_journal_id(dfByYear[0]))

            # Запрос оценок за предмет по id за N год
            dfRatingByJournalId = asyncio.run(AsyncRequests.rating_by_journal_id(dfByYear[0]))

            # Получение результатов ЕГЭ по StudentsByJournalId
            dfEgeMarksByStudentsId = asyncio.run(AsyncRequests.ege_marks_by_student_id(dfStudentsByJournalId))
            dfEgeMarksByStudentsId = TransformationsOverDataframe.ege_marks_transpose(dfEgeMarksByStudentsId)
            
            # Перевод файлов в csv для скачивания
            dfByYear_csv = dfByYear[0].to_csv(index=False)
            dfStudentsByJournalId_csv = dfStudentsByJournalId.to_csv(index=False)
            dfRatingByJournalId_csv = dfRatingByJournalId.to_csv(index=False)
            dfEgeMarksByStudentsId_csv = dfEgeMarksByStudentsId.to_csv(index=False)
            
            # Перевод в html
            df_year_now_html = df_to_html(dfByYear[0].head(100))
            df_students_by_journal_id_html = df_to_html(dfStudentsByJournalId.head(100))
            df_rating_by_journal_id_html = df_to_html(dfRatingByJournalId.head(100))
            df_ege_by_students_id_html = df_to_html(dfEgeMarksByStudentsId.head(100))

            context = {
                'year_from': year_from,
                'year_to': year_to,
                'range_from': range_from,
                'range_to': range_to,
                
                'institutes_list': institutes_list,
                'institutes_list_item': institutes_list_item,
                
                'df_year_now_html': df_year_now_html,
                'df_students_by_journal_id_html': df_students_by_journal_id_html,
                'df_rating_by_journal_id_html': df_rating_by_journal_id_html,
                'df_ege_by_students_id_html': df_ege_by_students_id_html,
                
                'df_year_size': dfByYear[0].shape[0], 
                'df_students_by_journal_size': dfStudentsByJournalId.shape[0],
                'df_rating_by_journal_size': dfRatingByJournalId.shape[0],
                'df_ege_by_students_size': dfEgeMarksByStudentsId.shape[0],
                
                'dfByYear_csv': dfByYear_csv,
                'dfStudentsByJournalId_csv': dfStudentsByJournalId_csv, 
                'dfRatingByJournalId_csv': dfRatingByJournalId_csv,
                'dfEgeMarksByStudentsId_csv': dfEgeMarksByStudentsId_csv,
            }

            return render(request, 'index.html', context)
        except:
            messages.warning(request, 'Некорректно введенные данные или проблема с интернет соединением')

    return render(request, 'index.html', {'institutes_list': institutes_list})


# Получение данных из CSV файлов
def get_info_from_csv(request):
    """Генерация интерфейса взаимодействия со страницей экстракции данных из CSV файлов
    """

    global dfByYear 
    global dfStudentsByJournalId
    global dfRatingByJournalId
    global dfEgeMarksByStudentsId
    global institutes_list_item
    global year_from
    global year_to
    
    institutes_list = ["Высшая школа электроники и компьютерных наук", "Архитектурно-строительный институт", "Высшая медико-биологическая школа", "Высшая школа экономики и управления", 
                    "Институт естественных и точных наук", "Институт лингвистики и международных коммуникаций", "Институт медиа и социально-гуманитарных наук", 
                    "Институт спорта, туризма, сервиса", "Политехнический институт", "Юридический институт", "Филиалы ЮУрГУ"]
    
    expected_headers_dfYear = {'CheckType', 'CourseEduId', 'CourseNumber', 'DirectionCode', 'DirectionName', 'GroupId', 'Id', 'IsPractice', 'Speciality', 'StudyForm', 'SubjectName', 'Term', 'Year'}
    expected_headers_dfStudents = {'EnrollScore', 'FinancialForm', 'HasOlympParticipation', 'Id', 'LiveCity', 'RegisterCity', 'Sex', 'Status', 'journal_id', 'GroupId'}  
    expected_headers_dfRating = {'Mark', 'Rating', 'StudentId', 'journal_id'}
    expected_headers_dfEge = {'student_id', 'Mark1', 'Mark2', 'Mark3', 'Subject1', 'Subject2', 'Subject3'}
    
    if request.method == 'POST':
        try:
            institutes_list_item = request.POST.get('institutes_list_item')
            file_dfYear = request.FILES['file_dfYear']
            file_dfStudents = request.FILES['file_dfStudents']
            file_dfRating = request.FILES['file_dfRating']
            file_dfEge = request.FILES['file_dfEge']
            
            dfByYear = [1] 
            dfByYear[0] = pd.read_csv(file_dfYear)
            dfStudentsByJournalId = pd.read_csv(file_dfStudents)
            dfRatingByJournalId = pd.read_csv(file_dfRating)
            dfEgeMarksByStudentsId = pd.read_csv(file_dfEge)
            
            # Проверка заголовков
            if set(dfByYear[0].columns) != expected_headers_dfYear:
                raise ValueError('Некорректные заголовки в файле "Журнал по годам"')
            if set(dfStudentsByJournalId.columns) != expected_headers_dfStudents:
                raise ValueError('Некорректные заголовки в файле "Информация о студентах"')
            if set(dfRatingByJournalId.columns) != expected_headers_dfRating:
                raise ValueError('Некорректные заголовки в файле "Рейтинг студентов по предметам"')
            if set(dfEgeMarksByStudentsId.columns) != expected_headers_dfEge:
                raise ValueError('Некорректные заголовки в файле "Результаты ЕГЭ студентов"')
            
            year = re.findall(r'\d{4}-\d{4}', file_dfYear.name)  # поиск всех цифр в названии файла по шаблону
            year_from, year_to = year[0].split('-')
            
            df_year_now_html = df_to_html(dfByYear[0].head(100))
            df_students_by_journal_id_html = df_to_html(dfStudentsByJournalId.head(100))
            df_rating_by_journal_id_html = df_to_html(dfRatingByJournalId.head(100))
            df_ege_by_students_id_html = df_to_html(dfEgeMarksByStudentsId.head(100))
            
            return render(request, 'import-csv.html', {
                'df_year_now_html': df_year_now_html,
                'df_students_by_journal_id_html': df_students_by_journal_id_html,
                'df_rating_by_journal_id_html': df_rating_by_journal_id_html,
                'df_ege_by_students_id_html': df_ege_by_students_id_html,
                'year': year[0],
                
                'df_year_size': dfByYear[0].shape[0], 
                'df_students_by_journal_size': dfStudentsByJournalId.shape[0],
                'df_rating_by_journal_size': dfRatingByJournalId.shape[0],
                'df_ege_by_students_size': dfEgeMarksByStudentsId.shape[0],
                
                'institutes_list': institutes_list,
                'institutes_list_item': institutes_list_item,
            })
        except ValueError as e:
            messages.warning(request, str(e))
        except Exception:
            messages.warning(request, 'Загружены некорректные файлы')
    
    return render(request, 'import-csv.html', {
        'institutes_list': institutes_list,
    })


# Анализ данных
def get_data_analysis(request):

    global dfByYear 
    global dfStudentsByJournalId
    global dfRatingByJournalId
    global dfEgeMarksByStudentsId
    global analysis_model
    global institutes_list_item
    global year_from
    global year_to
    
    df_predictive_model = None
    
    # Словарь моделей
    predictive_model_names = {
        'model1': 'Предсказание оценки дисциплины за указанный семестр',
        'model2': 'Предсказание оценки дисциплины, проходящей в указанном диапазоне семестров',
        'model3': 'Предсказание оценки по практикам',
        'model4': 'Предсказание отчисления студентов по экзаменам или зачетам',
        'model5': 'Предсказание успеваемости на основе рейтинга по экзаменам или зачетам',
        'model6': 'Предсказание успеваемости при завершении обучения',
    }
    
    if request.method == 'POST' and dfByYear: 
        try:
            predictive_model = request.POST.get('predictive_model') # Модель
            selected_subject = request.POST.get('subjects_in_term_intersection') # Предмет 
            model_training_method = request.POST.get('model_training_method') # Модель машинного обучения 
            test_size = float(request.POST.get('test_size')) ########   
            n_estimators = int(request.POST.get('n_estimators')) ######## 
            max_depth = int(request.POST.get('max_depth'))  ######## 
            min_samples_split = int(request.POST.get('min_samples_split')) ######## --------> Гиперпараметры
            min_samples_leaf = int(request.POST.get('min_samples_leaf')) ######## 
            max_features = int(request.POST.get('max_features')) ######## 
            term_from = int(request.POST.get('term_from')) # Семестр ОТ
            term_to = int(request.POST.get('term_to')) # Семестр ДО
            
            type_of_control = request.POST.get('type_of_control') # Тип контроля предмета зачет/экзамен
            
            do_data_synthesis = request.POST.get('do_data_synthesis') # Делать синтез данных
            if do_data_synthesis == 'on':
                do_data_synthesis = True
            else:
                do_data_synthesis = False
            
            do_emission_removal = request.POST.get('do_emission_removal') # Делать удаление выбросов
            if do_emission_removal == 'on':
                do_emission_removal = True
            else:
                do_emission_removal = False
                
            do_noise_remove = request.POST.get('do_noise_remove') # Делать удаление шумов
            if do_noise_remove == 'on':
                do_noise_remove = True
            else:
                do_noise_remove = False

            dfSamplingYearsData = dfByYear[0]
            
            # Формирование обучающей модели
            predictive_model_object = PredictiveModels(dfStudentsByJournalId, dfEgeMarksByStudentsId, dfRatingByJournalId, dfSamplingYearsData)
            df_predictive_model = predictive_model_result(predictive_model_object, predictive_model, selected_subject, type_of_control, term_from, term_to)
            
            # Анализ данных   
            analysis_model = DataAnalysis.train_by_model(dataframe_model=df_predictive_model, model_training_method=model_training_method, test_size=test_size,
                n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf, max_features=max_features, do_data_synthesis=do_data_synthesis,
                do_emission_removal=do_emission_removal, do_noise_remove=do_noise_remove)
            
            # Формирование сохраняемого файла с результатом обучения и доп параметрами
            analysis_model_save_list = analysis_model + (predictive_model, selected_subject, type_of_control, term_from, term_to, institutes_list_item, year_from, year_to)
            
            predictive_model_name = predictive_model_names.get(predictive_model, '')
            # Формирование названия файла для сохранения
            full_predictive_model_file_name = predictive_model_file_name(predictive_model_name, predictive_model, selected_subject,
                                                                        type_of_control, term_from, term_to, year_from, year_to)
            
            # Создание пути к файлу
            model_directory = os.path.join(settings.MEDIA_ROOT, 'analysis_models_files')
            os.makedirs(model_directory, exist_ok=True)  # Создание директории, если она не существует
            model_filename = os.path.join(model_directory, f'{full_predictive_model_file_name}.joblib')
            
            # Сохранение модели в файл joblib в папку analysis_models_files
            joblib.dump(analysis_model_save_list, model_filename)
                
            # Графики      
            graph_builder = GraphsBuilder(analysis_model[0], analysis_model[2], analysis_model[3], analysis_model[4])
            feature_importance_graph = graph_builder.feature_importance_graph()
            predictions_graph = graph_builder.plot_predictions()
            plot_confusion_matrix = graph_builder.plot_confusion_matrix()
            
            # Обучающая выборка в html
            df_predictive_model_html = df_to_html(df_predictive_model)
                
            return render(request, 'analysis.html', {
                'df_predictive_model_html': df_predictive_model_html,
                'df_predictive_model_size': df_predictive_model.shape[0],
                
                'prediction_result': analysis_model[1],
                'results_df': df_to_html(analysis_model[5]),
                'analysis_model_file': model_filename,
                
                'predictive_model': predictive_model,
                'selected_subject': selected_subject,
                'term_from': term_from,
                'term_to': term_to,
                
                'model_training_method': model_training_method,
                'type_of_control': type_of_control,
                'test_size': test_size,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': max_features,
                'do_data_synthesis': do_data_synthesis,
                'do_emission_removal': do_emission_removal,
                'do_noise_remove': do_noise_remove,
                
                'feature_importance_graph': feature_importance_graph,
                'predictions_graph': predictions_graph,
                'plot_confusion_matrix': plot_confusion_matrix,
            })
        except:
            messages.warning(request, 'Некорректные параметры для обучения или используемые данные')
            
    return render(request, 'analysis.html')


# Предметы в диапазоне семестров
def get_subjects_intersection(request):
    global dfByYear 
    
    if dfByYear:
        term_from = int(request.GET.get('term_from', None))
        term_to = int(request.GET.get('term_to', None))
        selected_model = request.GET.get('selected_model', None)
    
        if selected_model == 'model3':
            subjects_in_term_intersection = AssistantForInterface.get_subjects_intersection(dfByYear[0], term_from, term_to, True)
        else:
            subjects_in_term_intersection = AssistantForInterface.get_subjects_intersection(dfByYear[0], term_from, term_to, False)
    
        return JsonResponse({'subjects_in_term_intersection': subjects_in_term_intersection})
    else:
        return JsonResponse({'error': 'dfByYear is None'})
    
    
# Формирование обучающей выборки в зависимости от модели   
def predictive_model_result(predictive_model_object: PredictiveModels, predictive_model, selected_subject, type_of_control, 
                            term_from, term_to, include_student_id: bool = False, is_reuse: bool = False):
    
    type_of_control_inner = 0
    if type_of_control == 'credit':
        type_of_control_inner = 0
    else:
        type_of_control_inner = 1
          
    if predictive_model == 'model1' and selected_subject and term_from and term_to:
        return predictive_model_object.model_subjectName(selected_subject, term_from, term_to, False, include_student_id)
    
    if predictive_model == 'model2' and selected_subject and term_from and term_to:
        return predictive_model_object.model_subjectName(selected_subject, term_from, term_to, False, include_student_id)
    
    if predictive_model == 'model3' and selected_subject and term_from and term_to:
        return predictive_model_object.model_subjectName(selected_subject, term_from, term_to, True, include_student_id)
    
    if predictive_model == 'model4' and term_from and term_to:
        return predictive_model_object.model_students_rating_status(term_from, term_to, type_of_control_inner, True, include_student_id)
    
    if predictive_model == 'model5' and term_from and term_to:
        return predictive_model_object.model_students_rating_status(term_from, term_to, type_of_control_inner, False, include_student_id)
    
    if predictive_model == 'model6' and term_from and term_to:
        return predictive_model_object.model_avg_final_rating(term_from, term_to, include_student_id, is_reuse)

    return None


# Скачать обученную модель на странице анализа данных 
def download_analysis_model(request, filename):
    # Составление пути к файлу
    filepath = os.path.join(settings.MEDIA_ROOT, filename)
    # Открытие файла и передача его в HttpResponse
    with open(filepath, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/octet-stream')
        # Кодируем имя файла
        encoded_filename = quote(os.path.basename(filepath))
        response['Content-Disposition'] = f'attachment; filename*=UTF-8\'\'{encoded_filename}'
    return response


# Формирование названия для файла в зависимости от модели
def predictive_model_file_name(predictive_model_name, predictive_model, selected_subject, type_of_control, 
                            term_from, term_to, year_from, year_to):
    
    if type_of_control == 'credit':
        type_of_control_inner = 'Зачетам'
    else:
        type_of_control_inner = 'Экзаменам'
          
    if predictive_model == 'model1':
        return f'({year_from}-{year_to}) {predictive_model_name} по предмету {selected_subject} в {term_from} семестре'
    
    if predictive_model == 'model2':
        return f'({year_from}-{year_to}) {predictive_model_name} по предмету {selected_subject} c {term_from} по {term_to} семестр'

    if predictive_model == 'model3':
        return f'({year_from}-{year_to}) {predictive_model_name} по {selected_subject} в {term_from} семестре'
    
    if predictive_model == 'model4':
        return f'({year_from}-{year_to}) {predictive_model_name} по {type_of_control_inner} c {term_from} по {term_to} семестр'
    
    if predictive_model == 'model5':
        return f'({year_from}-{year_to}) {predictive_model_name} по {type_of_control_inner} c {term_from} по {term_to} семестр'
    
    if predictive_model == 'model6':
        return f'({year_from}-{year_to}) {predictive_model_name} c {term_from} по {term_to} семестр'

    return None


def predition_page(request):
    """Страница предсказания 
    """
    
    institutes_list = ["Высшая школа электроники и компьютерных наук", "Архитектурно-строительный институт", "Высшая медико-биологическая школа", "Высшая школа экономики и управления", 
                    "Институт естественных и точных наук", "Институт лингвистики и международных коммуникаций", "Институт медиа и социально-гуманитарных наук", 
                    "Институт спорта, туризма, сервиса", "Политехнический институт", "Юридический институт", "Филиалы ЮУрГУ"]
    
    if request.method == 'POST':
        try:
            predictive_model_file = request.FILES['predictive_model_file']
            predictive_model_loaded = joblib.load(predictive_model_file)
            
            clf_loaded = predictive_model_loaded[0]
            classification_quality_assessment_loaded = predictive_model_loaded[1] # Метрики обученной модели
            X_loaded = predictive_model_loaded[2]
            y_test_loaded = predictive_model_loaded[3] 
            y_pred_loaded = predictive_model_loaded[4]
            results_df_loaded = predictive_model_loaded[5] # Обученный dataframe
            model_name_from_file_loaded = predictive_model_loaded[6] # Модель (modelN) используемая при обучении
            selected_subject_loaded = predictive_model_loaded[7] # Выбранный предмет обученной модели
            type_of_control_loaded = predictive_model_loaded[8] # Тип контроля обученной модели
            term_from_loaded = predictive_model_loaded[9] # Семестр от
            term_to_loaded = predictive_model_loaded[10] # Семестр до
            institutes_list_item_loaded = predictive_model_loaded[11] # Высшая школа
            year_from_loaded = predictive_model_loaded[12] # Год от
            year_to_loaded = predictive_model_loaded[13] # Год до
            
            # ----------- Запрос к API -----------
            
            year_from_predict = request.POST.get('year_from_predict') # Год от
            year_to_predict = request.POST.get('year_to_predict') # Год до
            
            institutes_list_item_predict = institutes_list_item_loaded # Высшая школа
            institutes_copy_predict = institutes_list.copy()
            institutes_copy_predict.remove(institutes_list_item_predict)
            flat_excluded_institutes_predict = DataClear.institutional_exclusion(*institutes_copy_predict)
            
            # Года "от" и "до"
            years_list_predict = []
            for i in range(int(year_from_predict), int(year_to_predict) + 1):
                years_list_predict.append(i)
            
            dfByYear_predict = asyncio.run(AsyncRequests.journal_by_years(years_list_predict))
            # Дипазон курсов для сохранения 
            course_range_predict = [1, 2, 3, 4, 5, 6]
            filtered_course_range_predict = [x for x in course_range_predict if x < int(1) or x > int(4)]
            
            dfByYear_predict[0] = DataClear.drop_rows_in_journal(
                dfByYear_predict[0], 
                Speciality=['магистр', 'аспирант', 'специалист'], 
                StudyForm=['заочная', 'очно-заочная'],
                CourseNumber=filtered_course_range_predict,
                DirectionCode=flat_excluded_institutes_predict)

            # Все Учебные практики и Производственные практики к единому названию
            dfByYear_predict[0]['SubjectName'] = dfByYear_predict[0]['SubjectName'].apply(lambda x: re.sub(r'.*Производственная.*практика.*', 'Производственная практика', x))
            dfByYear_predict[0]['SubjectName'] = dfByYear_predict[0]['SubjectName'].apply(lambda x: re.sub(r'.*Учебная.*практика.*', 'Учебная практика', x))
            
            # Запрос студентов по id за N год
            dfStudentsByJournalId_predict = asyncio.run(AsyncRequests.students_by_journal_id(dfByYear_predict[0]))
            
            # Запрос оценок за предмет по id за N год
            dfRatingByJournalId_predict = asyncio.run(AsyncRequests.rating_by_journal_id(dfByYear_predict[0]))
            
            # Получение результатов ЕГЭ по StudentsByJournalId
            dfEgeMarksByStudentsId_predict = asyncio.run(AsyncRequests.ege_marks_by_student_id(dfStudentsByJournalId_predict))
            dfEgeMarksByStudentsId_predict = TransformationsOverDataframe.ege_marks_transpose(dfEgeMarksByStudentsId_predict)  
                    
            # ----------- Конец запроса к API -----------
            
            # Формирование обучающей модели
            predictive_model_object_predict = PredictiveModels(dfStudentsByJournalId_predict, dfEgeMarksByStudentsId_predict, dfRatingByJournalId_predict, dfByYear_predict[0])
                    
            df_predictive_model_predict = predictive_model_result(predictive_model_object_predict, model_name_from_file_loaded,
                                                                selected_subject_loaded, type_of_control_loaded,
                                                                term_from_loaded, term_to_loaded, True, True)
            
            # Предсказание
            student_id_column = df_predictive_model_predict['student_id'] # Сохраненине значений student_id 
            df_predictive_model_predict = df_predictive_model_predict.drop(columns=['student_id', df_predictive_model_predict.columns[-1]])
            predictions = clf_loaded.predict(df_predictive_model_predict)
            df_predictive_model_predict['Predicted'] = predictions # Добавление к датафрейму предсказанного результата
            df_predictive_model_predict.insert(0, 'student_id', student_id_column) # Возврат сохраненного столбца значений student_id
            
            
            df_predictive_model_predict['Sex'] = df_predictive_model_predict['Sex'].map({1: 'Мужской', 0: 'Женский'})
            df_predictive_model_predict['RegisterCity'] = df_predictive_model_predict['RegisterCity'].map({1: 'Челябинск', 0: 'Иногородний'})
            df_predictive_model_predict['FinancialForm'] = df_predictive_model_predict['FinancialForm'].map({1: 'Бюджет', 0: 'Контракт'})
            
            if model_name_from_file_loaded == 'model4':
                df_predictive_model_predict['Predicted'] = df_predictive_model_predict['Predicted'].map({1: 'Не будет отчислен', 0: 'Будет отчислен'})
            
            print(df_predictive_model_predict)
            
            df_predictive_model_predict_html = df_to_html(df_predictive_model_predict)
            
            df_predictive_model_predict_renamed = rename_dataframe_columns(df_predictive_model_predict)
            df_predictive_model_predict_csv = df_predictive_model_predict_renamed.to_csv(index=False)
            
            context = {
                'classification_quality_assessment_loaded': classification_quality_assessment_loaded,
                'model_name_from_file_loaded': model_name_from_file_loaded,
                'term_from_loaded': term_from_loaded,
                'term_to_loaded': term_to_loaded,
                'institutes_list_item_loaded': institutes_list_item_loaded,
                'year_from_loaded': year_from_loaded,
                'year_to_loaded': year_to_loaded,
                
                'year_from_predict': year_from_predict,
                'year_to_predict': year_to_predict,
                'df_predictive_model_predict_html': df_predictive_model_predict_html,
                'df_predictive_model_predict_csv': df_predictive_model_predict_csv,
            }
            
            if type_of_control_loaded == 'credit':
                type_of_control_inner = 'Зачет'
            else:
                type_of_control_inner = 'Экзамен'
            
            if model_name_from_file_loaded in ['model1', 'model2', 'model3']:
                context['selected_subject_loaded'] = selected_subject_loaded
            if model_name_from_file_loaded in ['model4', 'model5']:
                context['type_of_control_inner'] = type_of_control_inner
                    
            return render(request, 'prediction.html', context)
        except:
            messages.warning(request, 'По указанному диапазону годов невозможно использовать модель или проблема с интернет соединением')
    
    return render(request, 'prediction.html')


def get_info(requset):
    """Генерация интерфейса информации о программе

    Args:
        request: объект запроса

    Returns:
        html: интерфейс с информацией о программе
    """
    
    return render(requset, 'info.html')