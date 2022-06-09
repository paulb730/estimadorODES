import numpy as np


titulo_text = ["Modeling Data", "Modeling Description", "Algoritmo"]

algoritmos = ["PSO", "AP Monitor", "Mínimos cuadrados", "Gauss-Newton", "Algoritmos Genéticos"]

id_botones=["pso","ap","min","ganew","ga"]

subtitulo_text = ["Dataset del Modelo", "Figure Data", "Ecuaciones diferenciales", "Variables", "Parámetros",
                  "Condiciones Iniciales", "Fitting Data"]

titulo_dict = {"t0": "Datos del modelo experimentales ",
               "t1": "Descripción del modelo",
               "t2": "Optimización"}

algo_dict = {"t":"Algoritmos",
             "t0": "PSO",
             "t1": "AP Monitor",
             "t2": "Mínimos cuadrados",
             "t3": "Método Gauss Newton ",
             "t4":"Algoritmo genético (GA)"
             }

subtitulo_dict = {
    "t0": "Data experimental",
    "t1": "Gráfico de datos (Sin Optimización PSO)",
    "t2": "Ecuaciones diferenciales",
    "t3": "Variables",
    "t4": "Paràmetros",
    "t5": "Condiciones Iniciales",
    "t6": "Función Objetivo",
    "t7": "Fitting Data Process",
    "t8": "Tabla de Parámetros",
    "t9": "Valor función objetivo",
    "t10": "Gráfica Función Objetivo",
    "t11": "Gráfico Datos Experimentales ",
    "t12":"Figure Fitting Model with Data"
}

titulo_PSO_dict = {
    "t0": "Espacio de bùsqueda",
    "t1": "Partìculas",
    "t2": "Poblaciòn",
    "t3": "Iteraciones",
    "t4": "Constante de aceleracion ",
    "t5": " Cognitiva",
    "t6": "Social"

}

"""
Estructura de datos ecuaciones parametros y condiciones de los modelos estudiados 

"""
columns_ob = {

    0: [
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""}
    ],
    1: [
        {"Variables": "$$y_{1} $$",
         "Parametros": "$$p_{1} $$",
         "Ecuaciones": '$$\dfrac{dy_{1}}{dt}=p_{1}(27.8-y_{1})+\dfrac{p_{4}}{2.6}(y_{2}-y_{1})+\dfrac{4991}{t\sqrt{2\pi}}exp(-0.5(\dfrac{ln(t)-p_{2}}{p_{3}})^{2}) $$',
         "Condiciones Iniciales": "$$ y_{1}(0.1)=21.00 $$"},
        {"Variables": "$$y_{2} $$",
         "Parametros": "$$p_{2} $$",
         "Ecuaciones": '$$ \dfrac{dy_{2}}{dt}=\dfrac{p_{4}}{2.7}(y_{1}-y_{2}) $$ ',
         "Condiciones Iniciales": "$$ y_{2}(0.1)=38.75 $$"},
        {"Variables": "$$ t $$",
         "Parametros": "$$ p_{3} $$",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Parametros": "$$p_{4} $$", }
    ],
    2: [
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""}
    ],
    3: [
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""}
    ],
    4: [
        {"Variables":"",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""}
    ],
    5: [
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""}
    ],
    6: [
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""}
    ]

}

init_model_case_1 = {'z': [21.00, 30], 'thetha': [np.random.uniform(0.0001, 0.1), np.random.uniform(0.0001, 0.1),
                                           np.random.uniform(0.0001, 0.1), np.random.uniform(0.0001, 0.1)]}



