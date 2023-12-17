import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import sqlalchemy
import numpy as np


class Pipeline:
    def __init__(self, query=None, database=None):
        '''
        Класс Pipeline инициализируется с запросом к БД,
        по-умолчанию запрос к нашей БД
        '''

        # подключение к БД
        if database is None:
            self.database = {
                'drivername': 'postgresql',
                'username': 'postgres',
                'password': '123456',
                'host': 'localhost',
                'port': 5433,
                'database': 'leads_data',
                'query': {}
            }
        else:
            self.database = database

        # базовый запрос к БД
        if query is None:
            self.query = '''
                SELECT 
                    ads.created_at,
                    ads.d_utm_source,
                    ads.d_utm_medium,
                    ads.d_utm_campaign,
                    ads.m_clicks,
                    ads.m_cost,
                    leads.lead_id,
                    purchases.purchase_id,
                    purchases.m_purchase_amount
                FROM 
                    ads
                INNER JOIN 
                    leads ON ads.created_at = leads.lead_created_at
                    AND ads.d_utm_source = leads.d_lead_utm_source
                    AND ads.d_utm_medium = leads.d_lead_utm_medium
                    AND ads.d_utm_campaign = leads.d_lead_utm_campaign
                    AND ads.d_utm_content = leads.d_lead_utm_content
                LEFT JOIN 
                    purchases ON leads.client_id = purchases.client_id
                WHERE 
                    ads.created_at IS NOT NULL
                    AND leads.lead_created_at IS NOT NULL
                    AND purchases.m_purchase_amount IS NOT NULL
                    AND EXTRACT(DAY FROM (purchases.purchase_created_at - ads.created_at)) <= 15;
                    '''
        else:
            self.query = query

    def get_data(self):
        '''
        Метод get_data возвращает наш запрос из БД
        '''

        engine = create_engine(
            f"{self.database['drivername']}://{self.database['username']}:{self.database['password']}@"
            f"{self.database['host']}:{self.database['port']}/{self.database['database']}",
            connect_args=self.database['query']
        )
        return pd.read_sql_query(self.query, con=engine)

    def check_data(self, df: pd.DataFrame = None, expected_types: dict = None,
                   utm_source_list: list = None,
                   utm_medium_list: list = None,
                   utm_campaign_list: list = None,
                   outlier_threshold: float = 1):
        '''
        Метод check_data проверяет данные на корректность:
        1. Соответствие типу данных, по-умолчанию схема указана, можно свою
        2. Соответствие источнику, медиумов и компаний. Можно использовать свои списки для проверок
        3. Проверку дубликатов 
        4. Проверка корректности дат: не больше чем сегодня и не раньше 01.01.2022
        5. Проверку на отрицательные значения для числовых столбцов
        6. Проверка на выбросы, параметр outlier_threshold отвечает за процентильный порог 
        '''
        if df is None:
            df = self.get_data()

        if expected_types is None:
            expected_types = {
                'created_at': np.dtype('<M8[ns]'),
                'd_utm_source': np.dtype('O'),
                'd_utm_medium': np.dtype('O'),
                'd_utm_campaign': np.dtype('O'),
                'm_clicks': np.dtype('float64'),
                'm_cost': np.dtype('float64'),
                'lead_id': np.dtype('O'),
                'purchase_id': np.dtype('O'),
                'm_purchase_amount': np.dtype('float64')
            }

        if utm_source_list is None:
            utm_source_list = ['yandex']

        if utm_medium_list is None:
            utm_medium_list = ['cpc']

        if utm_campaign_list is None:
            utm_campaign_list = ['48306450', '48306518', '48306494',
                                 '48306435', '48306487', '48306473', '72000794', '48306461']

        # Проверка соответствия типов данных
        for col, dtype in expected_types.items():
            if df[col].dtype != dtype:
                print(
                    f"Столбец '{col}' Не соответствует ожидаемому типу данных {dtype}.")

        # Проверка на наличие NaN значений
        nan_values = df.isnull().sum()
        if nan_values.any():
            print(
                f"Обнаружены NaN значения в следующих столбцах:\n{nan_values[nan_values > 0]}")

        # Проверка на наличие дубликатов
        duplicate_values = df.duplicated().sum()
        if duplicate_values:
            print(
                f"Обнаружено {duplicate_values} дубликатов в данных.")

        # Проверка дат на превышение текущей даты и дат до 01.01.2022
        today = datetime.today()
        date_2022 = datetime(2022, 1, 1)
        date_columns = [col for col, dtype in expected_types.items()
                        if dtype == np.dtype('<M8[ns]')]

        for col in date_columns:
            # Проверка на даты после текущей даты
            dates_after_today = df[df[col] > today]
            if not dates_after_today.empty:
                print(
                    f"Обнаружены даты в столбце '{col}', превышающие текущую дату.")
                print(dates_after_today)

            # Проверка на даты до 01.01.2022
            dates_before_2022 = df[df[col] < date_2022]
            if not dates_before_2022.empty:
                print(
                    f"Обнаружены даты в столбце '{col}', до 01.01.2022:")
                print(dates_before_2022)

        # Проверка источников, кампаний и медиумов
        if 'd_utm_source' in df.columns:
            invalid_utm_source = df[~df['d_utm_source'].isin(utm_source_list)]
            if not invalid_utm_source.empty:
                print(f"Обнаружены неверные значения в 'd_utm_source':")
                print(invalid_utm_source)

        if 'd_utm_medium' in df.columns:
            invalid_utm_medium = df[~df['d_utm_medium'].isin(utm_medium_list)]
            if not invalid_utm_medium.empty:
                print(f"Обнаружены неверные значения в 'd_utm_medium':")
                print(invalid_utm_medium)

        if 'd_utm_campaign' in df.columns:
            invalid_utm_campaign = df[~df['d_utm_campaign'].isin(
                utm_campaign_list)]
            if not invalid_utm_campaign.empty:
                print(f"Обнаружены неверные значения в 'd_utm_campaign':")
                print(invalid_utm_campaign)

        # Проверка на отрицательные значения и выбросы в числовых столбцах
        numeric_columns = df.select_dtypes(
            include=['float64', 'int64']).columns.tolist()

        for col in numeric_columns:
            if df[col].min() < 0:
                print(
                    f"Обнаружены отрицательные значения в столбце '{col}'.")

            percentile_99 = df[col].quantile(outlier_threshold)
            outliers = df[df[col] > percentile_99]
            if not outliers.empty:
                print(f"Обнаружены выбросы в столбце '{col}':")
                display(outliers)
        return df

    def aggregate_data(self, df: pd.DataFrame = None):
        '''
        Метод aggregate_data принимает df и агрегирует данные к нужному нам формату,
        так же он проверяет наличие необходимых полей.
        По-умолчанию применяет метод   check_data() для создания df и проверки данных
        на случай если в данных содержатся ошибки 
        '''
        if df is None:
            df = self.check_data()

        required_columns = ['created_at', 'd_utm_source', 'd_utm_medium', 'd_utm_campaign',
                            'm_clicks', 'm_cost', 'lead_id', 'purchase_id', 'm_purchase_amount']

        if not all(col in df.columns for col in required_columns):
            raise ValueError("DataFrame doesn't contain all required columns.")

        grouped = df.groupby(
            ['created_at', 'd_utm_source', 'd_utm_medium', 'd_utm_campaign'])
        aggregated = grouped.agg({
            'm_clicks': 'sum',
            'm_cost': 'sum',
            'lead_id': 'nunique',
            'purchase_id': 'nunique',
            'm_purchase_amount': 'sum'
        })
        aggregated['CPL'] = (aggregated['m_cost'] /
                             aggregated['lead_id']).round(2)
        aggregated['ROAS'] = aggregated['m_purchase_amount'] / \
            aggregated['m_cost']
        aggregated['ROAS'] = aggregated['ROAS'].replace(np.inf, 0).round(2)
        aggregated['m_cost'] = aggregated['m_cost'].round(2)
        aggregated.reset_index(inplace=True)
        return aggregated

    def to_database(self, df: pd.DataFrame = None, table_name='end_to_end'):
        '''
        метод to_database() принимает агрегированные данные,
        по-умолчанию использует метод aggregate_data() для получения данных;
        проверяет соответствию необходимому типу; 
        проверяет есть ли подобная таблица в БД, если нет создает;
        при наличии похожей таблицы в добавляет новые  строчки 
        '''

        if df is None:
            df = self.aggregate_data()

        # Проверка типов данных
        expected_types = {
            'created_at': 'datetime64[ns]',
            'd_utm_source': 'object',
            'd_utm_medium': 'object',
            'd_utm_campaign': 'object',
            'm_clicks': 'float64',
            'm_cost': 'float64',
            'lead_id': 'int64',
            'purchase_id': 'int64',
            'm_purchase_amount': 'float64',
            'CPL': 'float64',
            'ROAS': 'float64'
        }

        for col, dtype in expected_types.items():
            if df[col].dtype != dtype:
                raise ValueError(
                    f"Column '{col}' doesn't match expected data type {dtype}.")

        engine = create_engine(
            f"{self.database['drivername']}://{self.database['username']}:{self.database['password']}@"
            f"{self.database['host']}:{self.database['port']}/{self.database['database']}",
            connect_args=self.database['query']
        )

        table_name = table_name

        # Проверяет наличие таблицы в БД

        if not sqlalchemy.inspect(engine).has_table(table_name):
            df.to_sql(table_name, engine, index=False, if_exists='replace'
                      )
            print(
                f"Таблица {table_name} успешно создана и данные сохранены.")
        else:
            existing_data = pd.read_sql_table(table_name, engine)
            new_rows = df[~df.apply(tuple, 1).isin(
                existing_data.apply(tuple, 1))]
            if not new_rows.empty:
                new_rows.to_sql(table_name, engine, index=False, if_exists='append',
                                )
                print(f"Данные добавлены в таблицу {table_name}.")

            else:
                print(
                    f"Новые данные отсутствуют. Таблица {table_name} остается без изменений.")
