import pandas as pd
import asyncio
import json
from .DigitalTrace import DigitalTrace
from tenacity import retry, wait_fixed, stop_after_attempt

class AsyncRequests:

    @staticmethod
    async def journal_by_years(years):
        """Асинхронный запрос журналов по годам

        Args:
            years (List): список годов для запроса

        Returns:
            results: [0] - объединенный dataframe по списку указзанных годов; 
                     [1] - [N] - dataframe с годами по отдельности;
        """
        
        def update_term(row):
            if row['CourseNumber'] == 1:
                return row['Term']
            elif row['CourseNumber'] == 2:
                return 3 if row['Term'] == 1 else 4
            elif row['CourseNumber'] == 3:
                return 5 if row['Term'] == 1 else 6
            elif row['CourseNumber'] == 4:
                return 7 if row['Term'] == 1 else 8
            elif row['CourseNumber'] == 5:
                return 9 if row['Term'] == 1 else 10
            else:
                return row['Term']  # Если CourseNumber не соответствует условиям, оставляем без изменений
        
        tasks = [asyncio.create_task(DigitalTrace.get_journal_by_year(year)) for year in years]
        results = await asyncio.gather(*tasks)
        results = [pd.DataFrame(json.loads(_)) for _ in results]
        
        # Объединение DataFrame
        combined_df = pd.concat(results)
               
        # Вставка объединенного DataFrame в начало списка
        results.insert(0, combined_df)
        
        for df in results:
            df['Term'] = df.apply(update_term, axis=1)
        
        return results
    
    @staticmethod
    async def students_by_journal_id(df):
        """Асинхронный запрос студентов по id в журналах

        Args:
            df (dataframe): dataframe с журналом за год полученный из journal_by_years

        Returns:
            concat_result: dataframe с рейтингом за предметы по id в журналах
        """
        
        semaphore = asyncio.Semaphore(100)  # Ограничение на количество одновременно запущенных задач
        ids = df['Id'].tolist()

        @retry(wait=wait_fixed(1), stop=stop_after_attempt(3))
        async def sem_task(id_, group_id):  # Добавляем group_id в параметры функции
            async with semaphore:
                data = await DigitalTrace.get_students_by_journal_id(id_)
                students_df = pd.DataFrame(json.loads(data))
                students_df['journal_id'] = id_  # Добавление столбца с ID журнала
                students_df['GroupId'] = group_id  # Добавление столбца с GroupId
                return students_df

        tasks = [sem_task(id_, group_id) for id_, group_id in zip(ids, df['GroupId'])]  # Передаем group_id вместе с id_
        results = await asyncio.gather(*tasks)
        concat_result = pd.concat(results)

        return concat_result


    @staticmethod
    async def rating_by_journal_id(df):
        """Асинхронный запрос рейтинга за предметы по id в журналах

        Args:
            df (dataframe): dataframe с журналом за год полученный из journal_by_years

        Returns:
            concat_result: dataframe с рейтингом за предметы по id в журналах
        """
        
        semaphore = asyncio.Semaphore(100)  # Ограничение на количество одновременно запущенных задач
        ids = df['Id'].tolist()
        
        @retry(wait=wait_fixed(1), stop=stop_after_attempt(3))
        async def sem_task(id_):
            async with semaphore:
                data = await DigitalTrace.get_rating_by_journal_id(id_)
                students_df = pd.DataFrame(json.loads(data))
                students_df['journal_id'] = id_  # Добавление столбца с ID журнала
                return students_df

        tasks = [sem_task(id_) for id_ in ids]
        results = await asyncio.gather(*tasks)
        concat_result = pd.concat(results)
        
        return concat_result
    
    @staticmethod
    async def ege_marks_by_student_id(df):
        """Асинхронный запрос результатов ЕГЭ студентов по id в students_by_journal_id

        Args:
            df (dataframe): dataframe с журналом за год полученный из students_by_journal_id

        Returns:
            concat_result: dataframe с результатом ЕГЭ студентов по id в students_by_journal_id
        """
        
        semaphore = asyncio.Semaphore(100)  # Ограничение на количество одновременно запущенных задач
        dfWithoutDuplicates = df.drop_duplicates(subset=['Id'])
        ids = dfWithoutDuplicates['Id'].tolist()
        
        @retry(wait=wait_fixed(1), stop=stop_after_attempt(3))
        async def sem_task(id_):
            async with semaphore:
                data = await DigitalTrace.get_ege_marks_by_student_id(id_)
                students_df = pd.DataFrame(json.loads(data))
                students_df['student_id'] = id_  # Добавление столбца с ID студента
                return students_df
        
        tasks = [sem_task(id_) for id_ in ids]
        results = await asyncio.gather(*tasks)
        concat_result = pd.concat(results)
        
        return concat_result