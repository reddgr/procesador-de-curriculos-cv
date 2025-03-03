
import os
import pandas as pd
import json
import textwrap
from scipy import spatial
from datetime import datetime
from openai import OpenAI

class ProcesadorCV:

    def __init__(self, api_key, cv_text, job_text, ner_pre_prompt, system_prompt, user_prompt, ner_schema, response_schema,
                inference_model="gpt-4o-mini", embeddings_model="text-embedding-3-small"):
        """
        Inicializa una instancia de la clase con los parámetros proporcionados.

        Args:
            api_key (str): La clave de API para autenticar con el cliente OpenAI.
            cv_text (str): contenido del CV en formato de texto.
            job_text (str): título de la oferta de trabajo a evaluar.
            ner_pre_prompt (str): instrucción de "reconocimiento de entidades nombradas" (NER) para el modelo en lenguaje natural.
            system_prompt (str): instrucción en lenguaje natural para la salida estructurada final.
            user_prompt (str): instrucción con los parámetros y datos calculados en el preprocesamiento.
            ner_schema (dict): esquema para la llamada con "structured outputs" al modelo de OpenAI para NER.
            response_schema (dict): esquema para la respuesta final de la aplicación.
            inference_model (str, opcional): El modelo de inferencia a utilizar. Por defecto es "gpt-4o-mini".
            embeddings_model (str, opcional): El modelo de embeddings a utilizar. Por defecto es "text-embedding-3-small".

        Atributos:
            inference_model (str): Almacena el modelo de inferencia seleccionado.
            embeddings_model (str): Almacena el modelo de embeddings seleccionado.
            client (OpenAI): Instancia del cliente OpenAI inicializada con la clave de API proporcionada.
            cv (str): Almacena el texto del currículum vitae proporcionado.

        """
        self.inference_model = inference_model
        self.embeddings_model = embeddings_model
        self.ner_pre_prompt = ner_pre_prompt
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt
        self.ner_schema = ner_schema
        self.response_schema = response_schema
        self.client = OpenAI(api_key=api_key)
        self.cv = cv_text
        self.job_text = job_text
        print("Cliente inicializado como",self.client)

    def extraer_datos_cv(self, temperature=0.5):
        """
        Extrae datos estructurados de un CV con OpenAI API.
        Args:
            pre_prompt (str): instrucción para el modelo en lenguaje natural.
            schema (dict): esquema de los parámetros que se espera extraer del CV.
            temperature (float, optional): valor de temperatura para el modelo de lenguaje. Por defecto es 0.5.
        Returns:
            pd.DataFrame: DataFrame con los datos estructurados extraídos del CV.
        Raises:
            ValueError: si no se pueden extraer datos estructurados del CV.
        """
        response = self.client.chat.completions.create(
            model=self.inference_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": self.ner_pre_prompt},
                {"role": "user", "content": self.cv}
            ],
            functions=[
                {
                    "name": "extraer_datos_cv",
                    "description": "Extrae tabla con títulos de puesto de trabajo, nombres de empresa y períodos de un CV.",
                    "parameters": self.ner_schema
                }
            ],
            function_call="auto"
        )

        if response.choices[0].message.function_call:
            function_call = response.choices[0].message.function_call
            structured_output = json.loads(function_call.arguments)
            if structured_output.get("experiencia"):
                df_cv = pd.DataFrame(structured_output["experiencia"]) 
                return df_cv
            else:
                raise ValueError(f"No se han podido extraer datos estructurados: {response.choices[0].message.content}")
        else:
            raise ValueError(f"No se han podido extraer datos estructurados: {response.choices[0].message.content}")
        

    def procesar_periodos(self, df):    
        """
        Procesa los períodos en un DataFrame y añade columnas con las fechas de inicio, fin y duración en meses. 
        Si no hay fecha de fin, se considera la fecha actual.
        Args:
            df (pandas.DataFrame): DataFrame que contiene una columna 'periodo' con períodos en formato 'YYYYMM-YYYYMM' o 'YYYYMM'.
        Returns:
            pandas.DataFrame: DataFrame con columnas adicionales 'fec_inicio', 'fec_final' y 'duracion'.
                - 'fec_inicio' (datetime.date): Fecha de inicio del período.
                - 'fec_final' (datetime.date): Fecha de fin del período.
                - 'duracion' (int): Duración del período en meses.
        """
        # Función lambda para procesar el período
        def split_periodo(periodo):
            dates = periodo.split('-')
            start_date = datetime.strptime(dates[0], "%Y%m")
            if len(dates) > 1:
                end_date = datetime.strptime(dates[1], "%Y%m")
            else:
                end_date = datetime.now()
            return start_date, end_date

        df[['fec_inicio', 'fec_final']] = df['periodo'].apply(lambda x: pd.Series(split_periodo(x)))

        # Formateamos las fechas para mostrar mes, año, y el primer día del mes (dado que el día es irrelevante y no se suele especificar)
        df['fec_inicio'] = df['fec_inicio'].dt.date
        df['fec_final'] = df['fec_final'].dt.date

        # Añadimos una columna con la duración en meses
        df['duracion'] = df.apply(
            lambda row: (row['fec_final'].year - row['fec_inicio'].year) * 12 + 
                        row['fec_final'].month - row['fec_inicio'].month, 
            axis=1
        )

        return df


    def calcular_embeddings(self, df, column='puesto', model_name='text-embedding-3-small'):
        """
        Calcula los embeddings de una columna de un dataframe con OpenAI API.
        Args:
            cv_df (pandas.DataFrame): DataFrame con los datos de los CV.
            column (str, optional): Nombre de la columna que contiene los datos a convertir en embeddings. Por defecto es 'puesto'.
            model_name (str, optional): Nombre del modelo de embeddings. Por defecto es 'text-embedding-3-small'.
        """
        df['embeddings'] = df[column].apply(
            lambda puesto: self.client.embeddings.create(
                input=puesto, 
                model=model_name
            ).data[0].embedding
        )
        return df


    def calcular_distancias(self, df, column='embeddings', model_name='text-embedding-3-small'):
        """
        Calcula la distancia coseno entre los embeddings del texto y los incluidos en una columna del dataframe.
        Params:
        df (pandas.DataFrame): DataFrame que contiene los embeddings.
        column (str, optional): nombre de la columna del DataFrame que contiene los embeddings. Por defecto, 'embeddings'.
        model_name (str, optional): modelo de embeddings de la API de OpenAI. Por defecto "text-embedding-3-small".
        Returns:
        pandas.DataFrame: DataFrame ordenado de menor a mayor distancia, con las distancias en una nueva columna.
        """
        response = self.client.embeddings.create(
            input=self.job_text,
            model=model_name
        )
        emb_compare = response.data[0].embedding

        df['distancia'] = df[column].apply(lambda emb: spatial.distance.cosine(emb, emb_compare))
        df.drop(columns=[column], inplace=True)
        df.sort_values(by='distancia', ascending=True, inplace=True)
        return df


    def calcular_puntuacion(self, df, req_experience, positions_cap=4, dist_threshold_low=0.6, dist_threshold_high=0.7):
        """
        Calcula la puntuación de un CV a partir de su tabla de distancias (con respecto a un puesto dado) y duraciones. 

        Params:
        df (pandas.DataFrame): datos de un CV incluyendo diferentes experiencias incluyendo duracies y distancia previamente calculadas sobre los embeddings de un puesto de trabajo
        req_experience (float): experiencia requerida en meses para el puesto de trabajo (valor de referencia para calcular una puntuación entre 0 y 100 en base a diferentes experiencias)
        positions_cap (int, optional): Maximum number of positions to consider for scoring. Defaults to 4.
        dist_threshold_low (float, optional): Distancia entre embeddings a partir de la cual el puesto del CV se considera "equivalente" al de la oferta.
        max_dist_threshold (float, optional): Distancia entre embeddings a partir de la cual el puesto del CV no puntúa.
        
        Returns:
        pandas.DataFrame: DataFrame original añadiendo una columna con las puntuaciones individuales contribuidas por cada puesto.
        float: Puntuación total entre 0 y 100.
        """
        # A efectos de puntuación, computamos para cada puesto como máximo el número total de meses de experiencia requeridos
        df['duration_capped'] = df['duracion'].apply(lambda x: min(x, req_experience))
        # Normalizamos la distancia entre 0 y 1, siendo 0 la distancia mínima y 1 la máxima
        df['adjusted_distance'] = df['distancia'].apply(
            lambda x: 0 if x <= dist_threshold_low else (
                1 if x >= dist_threshold_high else (x - dist_threshold_low) / (dist_threshold_high - dist_threshold_low)
            )
        )
        # Cada puesto puntúa en base a su duración y a la inversa de la distancia (a menor distancia, mayor puntuación)
        df['position_score'] = round(((1 - df['adjusted_distance']) * (df['duration_capped']/req_experience) * 100), 2)
        # Descartamos puestos con distancia superior al umbral definido (asignamos puntuación 0), y ordenamos por puntuación
        df.loc[df['distancia'] >= dist_threshold_high, 'position_score'] = 0
        df = df.sort_values(by='position_score', ascending=False)
        # Nos quedamos con los puestos con mayor puntuación (positions_cap)
        df.iloc[positions_cap:, df.columns.get_loc('position_score')] = 0
        # Totalizamos (no debería superar 100 nunca, pero ponemos un límite para asegurar) y redondeamos a dos decimales
        total_score = round(min(df['position_score'].sum(), 100), 2)
        return df, total_score
    
    def filtra_experiencia_relevante(self, df):
        """
        Filtra las experiencias relevantes del dataframe y las devuelve en formato diccionario.
        Args:
            df (pandas.DataFrame): DataFrame con la información completa de experiencia.
        Returns:
            dict: Diccionario con las experiencias relevantes.
        """
        df_experiencia =  df[df['position_score'] > 0].copy()
        df_experiencia.drop(columns=['periodo', 'fec_inicio', 'fec_final', 
                                     'distancia', 'duration_capped', 'adjusted_distance'], inplace=True)
        experiencia_dict = df_experiencia.to_dict(orient='list')
        return experiencia_dict
    
    def llamada_final(self, req_experience, puntuacion, dict_experiencia):
        """
        Realiza la llamada final al modelo de lenguaje para generar la respuesta final.
        Args:
        req_experience (int): Experiencia requerida en meses para el puesto de trabajo.
        puntuacion (float): Puntuación total del CV.
        dict_experiencia (dict): Diccionario con las experiencias relevantes.
        Returns:
        dict: Diccionario con la respuesta final.
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self.user_prompt.format(job=self.job_text, req_experience=req_experience,puntuacion=puntuacion, exp=dict_experiencia)
            }
        ]

        functions = [
            {
                "name": "respuesta_formateada",
                "description": "Devuelve el objeto con puntuacion, experiencia y descripcion de la experiencia",
                "parameters": self.response_schema
            }
        ]

        response = self.client.chat.completions.create(
            model=self.inference_model,
            temperature=0.5,
            messages=messages,
            functions=functions,
            function_call={"name": "respuesta_formateada"}
        )

        if response.choices[0].message.function_call:
            function_call = response.choices[0].message.function_call
            structured_output = json.loads(function_call.arguments)
            print("Respuesta:\n", json.dumps(structured_output, indent=4, ensure_ascii=False))
            wrapped_description = textwrap.fill(structured_output['descripcion de la experiencia'], width=120)
            print(f"Descripción de la experiencia:\n{wrapped_description}")
            return structured_output
        else:
            raise ValueError(f"Error. No se ha podido generar respuesta:\n {response.choices[0].message.content}")
    
    def procesar_cv_completo(self, req_experience, positions_cap, dist_threshold_low, dist_threshold_high):
        """
        Procesa un CV y calcula la puntuación final.
        Args:
            req_experience (int, optional): Experiencia requerida en meses para el puesto de trabajo.
            positions_cap (int, optional): Número máximo de puestos a considerar para la puntuación.
            dist_threshold_low (float, optional): Distancia límite para considerar un puesto equivalente.
            dist_threshold_high (float, optional): Distancia límite para considerar un puesto no relevante.
        Returns:
            pd.DataFrame: DataFrame con las puntuaciones individuales contribuidas por cada puesto.
            float: Puntuación total entre 0 y 100.
        """
        df_datos_estructurados_cv = self.extraer_datos_cv()
        df_datos_estructurados_cv = self.procesar_periodos(df_datos_estructurados_cv)
        df_con_embeddings = self.calcular_embeddings(df_datos_estructurados_cv)
        df_con_distancias = self.calcular_distancias(df_con_embeddings)
        df_puntuaciones, puntuacion = self.calcular_puntuacion(df_con_distancias,
                                                                req_experience=req_experience,
                                                                positions_cap=positions_cap,
                                                                dist_threshold_low=dist_threshold_low,
                                                                dist_threshold_high=dist_threshold_high)
        dict_experiencia = self.filtra_experiencia_relevante(df_puntuaciones)
        dict_respuesta = self.llamada_final(req_experience, puntuacion, dict_experiencia)
        return dict_respuesta