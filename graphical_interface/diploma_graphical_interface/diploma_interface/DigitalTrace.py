import aiohttp
import json

class DigitalTrace:
    __BASE_URL = "https://digtrace.susu.ru/integration/DigitalTrace/"

    @staticmethod
    async def _get_data(endpoint):
        """Общий метод для выполнения HTTP-запросов и возврата JSON-ответа
        
        """
        
        url = DigitalTrace.__BASE_URL + endpoint
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(url) as response:
                data = await response.json()
                return json.dumps(data, ensure_ascii=False, indent=4)

    @staticmethod
    async def get_journal_by_year(year):
        """Получить журнал по году

        Args:
            year (int): год, по которому нужно получить журнал

        Returns:
            json: результат запроса
        """
        
        endpoint = f"GetJournals/Year/{year}"
        return await DigitalTrace._get_data(endpoint)

    @staticmethod
    async def get_teachers_by_journal_id(id_subject):
        """Получить преподавателей по id из журнала по годам

        Args:
            id_subject (string): id из журнала по годам

        Returns:
            json: результат запроса
        """
        
        endpoint = f"GetTeachersByJournalId/{id_subject}"
        return await DigitalTrace._get_data(endpoint)

    @staticmethod
    async def get_students_by_journal_id(id_subject):
        """Получить студентов по id из журнала по годам

        Args:
            id_subject (string): id из журнала по годам

        Returns:
            json: результат запроса
        """
        
        endpoint = f"GetStudentsByJournalId/{id_subject}"
        return await DigitalTrace._get_data(endpoint)

    @staticmethod
    async def get_grades_by_journal_id(id_subject):
        """Задания предметов студетов по id журнала

        Args:
            id_subject (string): id из журнала по годам

        Returns:
            json: результат запроса
        """
        
        endpoint = f"GetGradesByJournalId/{id_subject}"
        return await DigitalTrace._get_data(endpoint)

    @staticmethod
    async def get_rating_by_journal_id(id_subject):
        """Оценки за предметы по id журнала

        Args:
            id_subject (string): id из журнала по годам

        Returns:
            json: результат запроса
        """
        
        endpoint = f"GetRatingsByJournalId/{id_subject}"
        return await DigitalTrace._get_data(endpoint)

    @staticmethod
    async def get_ege_marks_by_student_id(id_subject):
        """Оценки ЕГЭ студентов по id студентов

        Args:
            id_subject (string): id из журнала по студентам

        Returns:
            json: результат запроса
        """
        
        endpoint = f"GetEgeMarksByStudentId/{id_subject}"
        return await DigitalTrace._get_data(endpoint)
