import sys
import os
import json
import gradio as gr
sys.path.append('src')
from procesador_de_cvs_con_llm import ProcesadorCV

use_dotenv = False
if use_dotenv:
    from dotenv import load_dotenv
    load_dotenv("../../../../../../../apis/.env")
    api_key = os.getenv("OPENAI_API_KEY")

else:
    api_key = os.getenv("OPENAI_API_KEY")

unmasked_chars = 8
masked_key = api_key[:unmasked_chars] + '*' * (len(api_key) - unmasked_chars*2) + api_key[-unmasked_chars:]
print(f"API key: {masked_key}")

def process_cv(job_text, cv_text, req_experience, req_experience_unit, positions_cap, dist_threshold_low, dist_threshold_high):
    if dist_threshold_low >= dist_threshold_high:
        return {"error": "dist_threshold_low must be lower than dist_threshold_high."}
    
    if not isinstance(cv_text, str) or not cv_text.strip():
        return {"error": "Please provide the CV or upload a file."}
    
    # Convertir la experiencia requerida a meses si se introduce en años
    if req_experience_unit == "years":
        req_experience = req_experience * 12

    try:
        procesador = ProcesadorCV(api_key, cv_text, job_text, ner_pre_prompt, 
                                  system_prompt, user_prompt, ner_schema, response_schema)
        dict_respuesta = procesador.procesar_cv_completo(
            req_experience=req_experience,
            positions_cap=positions_cap,
            dist_threshold_low=dist_threshold_low,
            dist_threshold_high=dist_threshold_high
        )
        return dict_respuesta
    except Exception as e:
        return {"error": f"Error en el procesamiento: {str(e)}"}

# Parámetros de ejecución:
job_text = "Generative AI engineer"
cv_sample_path = 'cv_examples/reddgr_cv.txt' # Ruta al fichero de texto con un currículo de ejemplo
with open(cv_sample_path, 'r', encoding='utf-8') as file:
    cv_text = file.read()
# Prompts:
with open('prompts/ner_pre_prompt.txt', 'r', encoding='utf-8') as f:
    ner_pre_prompt = f.read()
with open('prompts/system_prompt.txt', 'r', encoding='utf-8') as f:
    system_prompt = f.read()
with open('prompts/user_prompt.txt', 'r', encoding='utf-8') as f:
    user_prompt = f.read()
# Esquemas JSON:
with open('json/ner_schema.json', 'r', encoding='utf-8') as f:
    ner_schema = json.load(f)
with open('json/response_schema.json', 'r', encoding='utf-8') as f:
    response_schema = json.load(f)

# Fichero de ejemplo para autocompletar (opción que aparece en la parte de abajo de la interfaz de usuario):
with open('cv_examples/reddgr_cv.txt', 'r', encoding='utf-8') as file:
    cv_example = file.read()

default_parameters = [4, "years", 10, 0.5, 0.7] # Parámetros por defecto para el reinicio de la interfaz y los ejemplos predefinidos 

# Código CSS para truncar el texto de ejemplo en la interfaz (bloque "Examples" en la parte de abajo):
css = """
        table tbody tr {
            height: 2.5em; /* Set a fixed height for the rows */
            overflow: hidden; /* Hide overflow content */
        }

        table tbody tr td {
            overflow: hidden; /* Ensure content within cells doesn't overflow */
            text-overflow: ellipsis; /* Add ellipsis for overflowing text */
            white-space: nowrap; /* Prevent text from wrapping */
            vertical-align: middle; /* Align text vertically within the fixed height */
        }
        """

# Interfaz Gradio:
with gr.Blocks(css=css) as interface:
    # Inputs
    job_text_input = gr.Textbox(label="Vacancy Title", lines=1, placeholder="Enter the vacancy title")
    gr.Markdown("Required Experience")
    with gr.Row():
        req_experience_input = gr.Number(label="Required Experience", value=default_parameters[0], precision=0, elem_id="req_exp", show_label=False)
        req_experience_unit = gr.Dropdown(label="Period", choices=["months", "years"], value=default_parameters[1], elem_id="req_exp_unit", show_label=False)
    cv_text_input = gr.Textbox(label="CV in Text Format", lines=5, max_lines=5, placeholder="Enter the CV text")
    
    # Opciones avanzadas ocultas en un objeto "Accordion"
    with gr.Accordion("Advanced options", open=False):
        positions_cap_input = gr.Number(label="Maximum number of positions to extract", value=default_parameters[2], precision=0)
        dist_threshold_low_slider = gr.Slider(
            label="Minimum embedding distance threshold (equivalent position)", 
            minimum=0, maximum=1, value=default_parameters[3], step=0.05
        )
        dist_threshold_high_slider = gr.Slider(
            label="Maximum embedding distance threshold (irrelevant position)", 
            minimum=0, maximum=1, value=default_parameters[4], step=0.05
        )
    
    submit_button = gr.Button("Process")
    clear_button = gr.Button("Clear")
    
    output_json = gr.JSON(label="Result")

    # Ejemplos:
    examples = gr.Examples(
        examples=[
            ["Supermarket cashier", "Deli worker since 2021. Previously worked 2 months as a waiter in a tapas bar."] + default_parameters,
            ["Generative AI Engineer", cv_example] + default_parameters
        ],
        inputs=[job_text_input, cv_text_input, req_experience_input, req_experience_unit, positions_cap_input, dist_threshold_low_slider, dist_threshold_high_slider]
    )

    # Botón "Procesar"
    submit_button.click(
        fn=process_cv,
        inputs=[
            job_text_input, 
            cv_text_input, 
            req_experience_input, 
            req_experience_unit,
            positions_cap_input, 
            dist_threshold_low_slider, 
            dist_threshold_high_slider
        ],
        outputs=output_json
    )

    # Botón "Limpiar"
    clear_button.click(
        fn=lambda: ("","",*default_parameters),
        inputs=[],
        outputs=[
            job_text_input, 
            cv_text_input, 
            req_experience_input, 
            req_experience_unit,
            positions_cap_input, 
            dist_threshold_low_slider, 
            dist_threshold_high_slider
        ]
    )

    # Footer
    gr.Markdown("""
        <footer>
        <p>You can view the complete code for this app and the explanatory notebooks on 
        <a href='https://github.com/reddgr/procesador-de-curriculos-cv' target='_blank'>GitHub</a></p>
        <p>© 2024 <a href='https://talkingtochatbots.com' target='_blank'>talkingtochatbots.com</a></p>
        </footer>
    """)

# Lanzar la aplicación:
if __name__ == "__main__":
    interface.launch()