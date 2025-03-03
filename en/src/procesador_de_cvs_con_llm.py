
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
        Initializes an instance of the class with the provided parameters.

        Args:
            api_key (str): The API key to authenticate with the OpenAI client.
            cv_text (str): CV content in text format.
            job_text (str): title of the job offer to evaluate.
            ner_pre_prompt (str): "Named Entity Recognition" (NER) instruction for the natural language model.
            system_prompt (str): natural language instruction for the final structured output.
            user_prompt (str): instruction with parameters and data calculated in preprocessing.
            ner_schema (dict): schema for the "structured outputs" call to the OpenAI model for NER.
            response_schema (dict): schema for the final application response.
            inference_model (str, optional): The inference model to use. Default is "gpt-4o-mini".
            embeddings_model (str, optional): The embeddings model to use. Default is "text-embedding-3-small".

        Attributes:
            inference_model (str): Stores the selected inference model.
            embeddings_model (str): Stores the selected embeddings model.
            client (OpenAI): Instance of the OpenAI client initialized with the provided API key.
            cv (str): Stores the provided curriculum vitae text.
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
        Extracts structured data from a CV using OpenAI API.
        Args:
            temperature (float, optional): temperature value for the language model. Default is 0.5.
        Returns:
            pd.DataFrame: DataFrame with structured data extracted from the CV.
        Raises:
            ValueError: if structured data cannot be extracted from the CV.
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
                    "description": "Extracts table with job titles, company names and periods from a CV.",
                    "parameters": self.ner_schema
                }
            ],
            function_call="auto"
        )

        if response.choices[0].message.function_call:
            function_call = response.choices[0].message.function_call
            structured_output = json.loads(function_call.arguments)
            if structured_output.get("experience"):
                df_cv = pd.DataFrame(structured_output["experience"]) 
                return df_cv
            else:
                raise ValueError(f"Unable to extract structured data: {response.choices[0].message.content}")
        else:
            raise ValueError(f"Unable to extract structured data: {response.choices[0].message.content}")
        

    def procesar_periodos(self, df):    
        """
        Process periods in a DataFrame and adds columns with start dates, end dates, and duration in months.
        If there is no end date, the current date is considered.
            df (pandas.DataFrame): DataFrame containing a 'period' column with periods in 'YYYYMM-YYYYMM' or 'YYYYMM' format.
            pandas.DataFrame: DataFrame with additional columns 'fec_inicio', 'fec_final', and 'duracion'.
                - 'fec_inicio' (datetime.date): Start date of the period.
                - 'fec_final' (datetime.date): End date of the period.
                - 'duracion' (int): Duration of the period in months.
        """
        # Función lambda para procesar el período
        def split_period(period):
            dates = period.split('-')
            start_date = datetime.strptime(dates[0], "%Y%m")
            if len(dates) > 1:
                end_date = datetime.strptime(dates[1], "%Y%m")
            else:
                end_date = datetime.now()
            return start_date, end_date

        df[['fec_inicio', 'fec_final']] = df['period'].apply(lambda x: pd.Series(split_period(x)))

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


    def calcular_embeddings(self, df, column='role', model_name='text-embedding-3-small'):
        """
        Calculates the embeddings for a column in a dataframe using the OpenAI API.
        
            df (pandas.DataFrame): DataFrame containing the CV data.
            column (str, optional): Name of the column containing the text to be converted to embeddings. Default is 'role'.
            model_name (str, optional): Name of the embeddings model. Default is 'text-embedding-3-small'.
            
        Returns:
            pandas.DataFrame: DataFrame with an additional 'embeddings' column containing the generated embeddings.
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
        Calculates the cosine distance between the text embeddings and those included in a DataFrame column.
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the embeddings.
        column : str, optional
            Name of the DataFrame column containing the embeddings. Default is 'embeddings'.
        model_name : str, optional
            OpenAI API embedding model. Default is "text-embedding-3-small".
        --------
        pandas.DataFrame
            DataFrame sorted by distance in ascending order, with distances added as a new column.
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

        Calculates the score of a CV based on its distance table (relative to a given position) and durations.
        Parameters:
        ----------
        df : pandas.DataFrame
            CV data including different experiences with durations and distances previously calculated based on the embeddings of a job position.
        req_experience : float
            Required experience in months for the job position (reference value to calculate a score between 0 and 100 based on different experiences).
        positions_cap : int, optional
            Maximum number of positions to consider for scoring. Defaults to 4.
        dist_threshold_low : float, optional
            Distance between embeddings below which the CV position is considered "equivalent" to the job offer. Defaults to 0.6.
        dist_threshold_high : float, optional
            Distance between embeddings above which the CV position does not score. Defaults to 0.7.
        -------
        pandas.DataFrame
            Original DataFrame with an additional column containing individual scores contributed by each position.
        float
            Total score between 0 and 100.
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
        Filters the relevant experiences from the dataframe and returns them in dictionary format.
        Args:
            df (pandas.DataFrame): DataFrame with complete experience information.
        Returns:
            dict: Dictionary with the relevant experiences.
        """
        df_experiencia =  df[df['position_score'] > 0].copy()
        df_experiencia.drop(columns=['period', 'fec_inicio', 'fec_final', 
                                     'distancia', 'duration_capped', 'adjusted_distance'], inplace=True)
        experiencia_dict = df_experiencia.to_dict(orient='list')
        return experiencia_dict
    
    def llamada_final(self, req_experience, puntuacion, dict_experiencia):
        """
        Makes the final call to the language model to generate the final response.
            req_experience (int): Required experience in months for the job position.
            puntuacion (float): Total score of the CV.
            dict_experiencia (dict): Dictionary with relevant experiences.
            dict: Dictionary with the final response.
        Raises:
            ValueError: If no response is generated by the language model.
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
                "description": "Returns an object with score, experience and description of the experience",
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
            print("Response:\n", json.dumps(structured_output, indent=4, ensure_ascii=False))
            wrapped_description = textwrap.fill(structured_output['experience summary'], width=120)
            print(f"Experience summary:\n{wrapped_description}")
            return structured_output
        else:
            raise ValueError(f"Error. No response was generated:\n {response.choices[0].message.content}")
    
    def procesar_cv_completo(self, req_experience, positions_cap, dist_threshold_low, dist_threshold_high):
        '''
        Processes a CV and calculates the final score.
            req_experience (int, optional): Required experience in months for the job position.
            positions_cap (int, optional): Maximum number of positions to consider for scoring.
            dist_threshold_low (float, optional): Distance limit to consider a position equivalent.
            dist_threshold_high (float, optional): Distance limit to consider a position not relevant.
        Returns:
            dict: Dictionary with the final answer.
        '''
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