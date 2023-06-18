import numpy as np
import collect_data as colldat

titulo_text = ["Modeling Data", "Modeling Description", "Algoritmo"]

algoritmos = ["PSO", "Grey Wolf Optimizer", "Colonia Abejas", "Differential Evolution", "Simple Genético"]

id_botones = ["pso", "ap", "min", "ganew", "ga"]

super_data_Vector=[[{colldat.parametros_titulos[0]:"0",
                    colldat.parametros_titulos[1]:"0",
                    colldat.parametros_titulos[2]:"0",
                    colldat.parametros_titulos[3]:"0",
                    colldat.parametros_titulos[4]:"0",
                    colldat.parametros_titulos[5]:"0",
                    colldat.parametros_titulos[6]:"0"}]]*len(algoritmos)

super_data_algoritmo_usado=["None"]*len(algoritmos)

super_data_valores_estimados=[["0"]]*len(algoritmos)

super_data_valores_referenciales=[["0"]]*len(algoritmos)

super_data_fitness_estimado=["0"]*len(algoritmos)

super_data_fitness_referenciales=["0"]*len(algoritmos)

super_data_fitness_error_=[["0"]]*len(algoritmos)

super_data_param_name=[["0"]]*len(algoritmos)

#super_timing=["0"]*len(algoritmos)

cont_1=0
cont_2=0
cont_3=0
cont_4=0
cont_5=0




# id algoritmos parametros editables
id_edit_form = ["slider_w", "slider_c1", "slider_c2", "input_GEN", "input_POP"]

id_edit_algo_form_0 = ["slider_gen_MC","slider_pop_MC"]

id_edit_algo_form_1 = ["slider_gen_BE", "pop_BE", "limit_bee"]

id_edit_algo_form_2 = ["slider_gen_1", "pop_1", "f_", "cr_"]

id_edit_algo_form_3 = ["slider_gen_2", "pop_2", "f_1", "cr_2"]


##############################################

subtitulo_text = ["Dataset del Modelo", "Figure Data", "Ecuaciones diferenciales", "Variables", "Parámetros",
                  "Condiciones Iniciales", "Fitting Data"]

titulo_dict = {"t0": "Datos del modelo experimentales ",
               "t1": "Descripción del modelo",
               "t2": "Optimización"}




id_estadisticas = "estadisticas"

titulo_PSO_dict = {
    "t0": "Espacio de bùsqueda",
    "t1": "Partìcudasdlas",
    "t2": "Poblaciòn",
    "t3": "Iteraciones",
    "t4": "Constante de aceleracion ",
    "t5": " Cognitiva",
    "t6": "Social"

}

"""
Dimension de parametros de acuerdo al caso 
"""

VECTOR_DIM_PARAM = {
    0: 1,
    1: 4,
    2: 8,
    3: 4,
    4: 12,
    5: 6,
    6: 2,

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
        {"Variables": "$$ H_{s} (Hepatocitos Sanos)$$",
         "Parametros": "$$ \\beta_{s}, \\beta_{T} $$",
         "Ecuaciones": "$$\dot{H_{s}}=\\beta _{s}-kH_{s}V-\mu _{s}H_{s} $$",
         "Condiciones Iniciales": "$$H_{s0}= 4300 \dfrac {celulas}{mm^{3}}$$"},
        {"Variables": "$$ H_{i} (Hepatocitos Infectados)$$",
         "Parametros": "$$ \mu_{s}, \mu_{i}, \mu_{V}, \mu_{T} $$",
         "Ecuaciones": "$$ \dot{H_{i}}= kH_{s}V - \delta H_{i}T - \mu _{i}H_{i} $$",
         "Condiciones Iniciales": "$$H_{i0}= 700 \dfrac {celulas}{mm^{3}} $$"},
        {"Variables": "$$ V (Carga Viral)$$",
         "Parametros": "$$k,p,\delta$$",
         "Ecuaciones": "$$\dot{V}= pH_{i} - \mu_{V}V $$",
         "Condiciones Iniciales": "$$V_{0}= 1200 \dfrac {UI}{mL}$$"},
        {"Variables": "$$ T (Células T)$$",
         "Parametros": "$$T_{máx}$$",
         "Ecuaciones": "$$\dot{T}= \\beta _{T}(1-\dfrac{T}{T_{max}})V - \mu_{T}T $$",
         "Condiciones Iniciales": "$$T_{0} = 400 \dfrac {celulas}{mm^{3}} $$"},
        {"Variables": "$$ t (tiempo)$$",
         "Parametros": "",
         "Ecuaciones": "",
         "Condiciones Iniciales": "$$ T_{máx}=1000 \dfrac {celulas}{mm^{3}} $$"}
    ],
    3: [
        {"Variables": "$$x$$",
         "Parametros": "$$a$$",
         "Ecuaciones": "$$\dfrac{dx}{dt}= y$$",
         "Condiciones Iniciales": "$$x=2$$"},
        {"Variables": "$$y$$",
         "Parametros": "$$b$$",
         "Ecuaciones": "$$\dfrac{dy}{dt}= z$$",
         "Condiciones Iniciales": ""},
        {"Variables": "$$z$$",
         "Parametros": "$$c$$",
         "Ecuaciones": "$$ \dfrac{dz}{dt}= az+by+cx+d $$",
         "Condiciones Iniciales": ""},
        {"Variables": "",
         "Parametros": "$$d$$",
         "Ecuaciones": "",
         "Condiciones Iniciales": ""}
    ],
    4: [
        {"Variables": "$$ x $$",
         "Parametros": "$$ a_{10}, a_{11} , a_{12},a_{13}$$",
         "Ecuaciones": "$$\dfrac{dx}{dt}=x(a_{10}+a_{11}x+a_{12}y+a_{13}z)$$",
         "Condiciones Iniciales": "$$ x(0)=0.52 $$"},
        {"Variables": "$$ y $$",
         "Parametros": "$$ a_{20}, a_{21} , a_{22},a_{23}$$",
         "Ecuaciones": " $$ \dfrac{dy}{dt}=y(a_{20}+a_{21}x+a_{22}y+a_{23}z) $$",
         "Condiciones Iniciales": "$$ y(0)=0.15 $$"},
        {"Variables": "$$ z $$",
         "Parametros": "$$ a_{30}, a_{31} , a_{32},a_{33} $$",
         "Ecuaciones": " $$ \dfrac{dz}{dt}=z(a_{30}+a_{31}x+a_{32}y+a_{33}z) $$",
         "Condiciones Iniciales": "$$ z(0)=0.33 $$"}

    ],
    5: [
        {"Variables": "$$ H $$",
         "Parametros": "$$ kr_{1} $$",
         "Ecuaciones": "$$ \dfrac {dH}{dt}=kr_{1} - kr_{2}H - kr_{3}HV $$",
         "Condiciones Iniciales": "$$H= =10000000 (Células_{saludables} ) $$"},
        {"Variables": "$$ I $$",
         "Parametros": "$$  kr_{2} $$",
         "Ecuaciones": "$$ \dfrac {dI}{dt}=kr_{3}HV - kr_{4}I $$",
         "Condiciones Iniciales": "$$I=0 (Células_{Infectadas})$$"},
        {"Variables": "$$ V $$",
         "Parametros": "$$  kr_{3} $$",
         "Ecuaciones": "$$  \dfrac {dV}{dt}=-kr_{3}HV - kr_{5}V + kr_{6}I  $$",
         "Condiciones Iniciales": "$$V=100 (Virus)$$"},
        {"Variables": "$$$$",
         "Parametros": "$$ kr_{4}$$",
         "Ecuaciones": "$$LV= log_{10}(V)$$",
         "Condiciones Iniciales": "$$ LV=2 (log_{virus})$$"},
        {"Variables": "$$$$",
         "Parametros": "$$ kr_{5}$$",
         "Ecuaciones": "$$$$",
         "Condiciones Iniciales": "$$$$"},
        {"Variables": "$$$$",
         "Parametros": "$$  kr_{6}$$",
         "Ecuaciones": "$$$$",
         "Condiciones Iniciales": "$$$$"},

    ],
    6: [
        {"Variables": "$$ y_{1} $$",
         "Parametros": "$$ \\beta_{1} $$",
         "Ecuaciones": "$$ \dfrac{dy_{1}}{dt}=-\\beta_{1}y_{1} $$",
         "Condiciones Iniciales": "$$ y_{1}(0)=1 $$"},
        {"Variables": "$$ y_{2} $$",
         "Parametros": "$$ \\beta_{2} $$",
         "Ecuaciones": "$$ \dfrac{dy_{2}}{dt}=\\beta_{1}y_{1}-\\beta_{2}y_{2} $$",
         "Condiciones Iniciales": "$$ y_{2}(0)=0 $$"},

    ]

}

initial_conditions_ob = {
    0: {'z': [0, 0],
        'thetha': [0,0,0,0]

        },
    1: {'z': [27.8000, 0.2339],
        'thetha': [0.2729,
                   2.6539,
                   0.3710,
                   0.2004
                   ]
        },
    2: {
        'z': [4300, 700, 1000, 425],
        'thetha': [3.0e-4,
                   2.8e-5,
                   10,
                   0.14,
                   2.0e-2,
                   5.83,
                   10e-5,
                   1000
                   ]

    },
    3: {
        'z': [2, 0, 0],
        'thetha': [-37.699565828,
                   -2310.5055562,
                   -6173.4615908,
                   123.00783964
                   ]

    },
    4: {
        'z': [0.52, 0.15, 0.33],
        'thetha': [0.45, -0.6, -0.2, -0.66,
                   0.86, -0.02, -1.8, -0.59,
                   0.2, -0.06, -0.13, -0.5
                   ]

    },
    5: {
        'z': [1e6, 0, 100],
        'thetha': [1e5, 0.1, 2e-7, 0.5, 5, 100]

    },
    6: {
        'z': [1, 0],
        'thetha': [1.120e-2 , 6.8e-3]

    },
}

inital_guesses_paramters = {
    0: {'thetha': [np.random.uniform(0.0001, 0.1),
                   np.random.uniform(0.0001, 0.1),
                   np.random.uniform(0.0001, 0.1),
                   np.random.uniform(0.0001, 0.1)]},
    1: {'thetha': [np.random.uniform(0.1, 1),
                   np.random.uniform(0.1, 3),
                   np.random.uniform(0.1, 1),
                   np.random.uniform(0.1, 0.3)]},
    2: {'thetha': [4e-4,
                   1e-5,
                   13,
                   0.3,
                   1e-2,
                   5,
                   8e-5,
                   1200]},
    3: {'thetha': [np.random.uniform(-50, -20),
                   np.random.uniform(-9999, -1000),
                   np.random.uniform(-9999, -1000),
                   np.random.uniform(100, 200),
                   ]},
    4: {'thetha': [np.random.uniform(0.1, 0.5), np.random.uniform(-0.7, -0.3), np.random.uniform(-0.3, -0.1),
                   np.random.uniform(-0.7, -0.3),
                   np.random.uniform(0.5, 0.9), np.random.uniform(-0.09, -0.01), np.random.uniform(-2, -1),
                   np.random.uniform(-0.8, -0.4),
                   np.random.uniform(0.1, 0.3), np.random.uniform(-0.09, -0.03), np.random.uniform(-0.30, -0.10),
                   np.random.uniform(-0.9, -0.4)

                   ]},
    5: {'thetha': [1e5, 0.1, 2e-7, 0.5, 5, 100]},
    6: {'thetha': [0.01120, 0.0068]},

}



parametros_columns = {
    0: [{colldat.parametros_titulos[0]: "",
         colldat.parametros_titulos[1]: "",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]:""}],
    1: [{colldat.parametros_titulos[0]: "$$p_{1}$$",
         colldat.parametros_titulos[1]: "$$ 0.2729 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]:""
         },
        {colldat.parametros_titulos[0]: "$$p_{2}$$",
         colldat.parametros_titulos[1]: "$$ 2.6539 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4] : "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: "",
         },
        {colldat.parametros_titulos[0]: "$$p_{3}$$",
         colldat.parametros_titulos[1]: "$$ 0.3710 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]:""
         },
        {colldat.parametros_titulos[0]: "$$p_{4}$$",
         colldat.parametros_titulos[1]: "$$ 0.2004 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]:""
         }],
    2: [
        {colldat.parametros_titulos[0]: "$$ \\beta_{T} $$",
         colldat.parametros_titulos[1]: "$$ 3.0e-4 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]:""
         },
        {colldat.parametros_titulos[0]: "$$ k $$",
         colldat.parametros_titulos[1]: "$$ 2.8e-5 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]:""
         },
        { colldat.parametros_titulos[0]: "$$ p $$",
         colldat.parametros_titulos[1]: "$$ 10 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]:""
         },

        {colldat.parametros_titulos[0]: "$$ \mu_{i} $$",
         colldat.parametros_titulos[1]: "$$ 0.14 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]:""
         },
        {colldat.parametros_titulos[0]: "$$ \mu_{T} $$",
         colldat.parametros_titulos[1]: "$$ 2.0e-2 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ \mu_{V} $$",
         colldat.parametros_titulos[1]: "$$ 5.83 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ \delta $$",
         colldat.parametros_titulos[1]: "$$ 10e-5 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ T_{máx} $$",
         colldat.parametros_titulos[1]: "$$ 1000 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         }
    ],
    3: [{colldat.parametros_titulos[0]: "$$ a  $$",
         colldat.parametros_titulos[1]: "$$ -37.699565828 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ b $$",
         colldat.parametros_titulos[1]: "$$ -2310.5055562 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ c $$",
         colldat.parametros_titulos[1]: "$$ -6173.4615908 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ d $$",
         colldat.parametros_titulos[1]: "$$  123.00783964 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         }
        ],
    4: [{colldat.parametros_titulos[0]: "$$ a_{10}$$",
         colldat.parametros_titulos[1]: "$$0.45 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ a_{11}$$",
         colldat.parametros_titulos[1]: "$$ -0.6$$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: "",
         },
        {colldat.parametros_titulos[0]: "$$ a_{12}$$",
         colldat.parametros_titulos[1]: "$$ -0.2$$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: "",
         },
        {colldat.parametros_titulos[0]: "$$ a_{13}$$",
         colldat.parametros_titulos[1]: "$$ -0.66$$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: "",
         },
        {colldat.parametros_titulos[0]: "$$ a_{20}$$",
         colldat.parametros_titulos[1]: "$$  0.86$$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ a_{21}$$",
         colldat.parametros_titulos[1]: "$$  -0.02$$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ a_{22}$$",
         colldat.parametros_titulos[1]: "$$-1.8$$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ a_{23}$$",
         colldat.parametros_titulos[1]: "$$-0.59$$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ a_{30}$$",
         colldat.parametros_titulos[1]: "$$ 0.2$$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ a_{31}$$",
         colldat.parametros_titulos[1]: "$$ -0.06$$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ a_{32}$$",
         colldat.parametros_titulos[1]: "$$  -0.13 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ a_{33}$$",
         colldat.parametros_titulos[1]: "$$ -0.5$$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         }
        ],
    5: [{colldat.parametros_titulos[0]: "$$ k_{1}$$",
         colldat.parametros_titulos[1]: "$$ 1e5 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ k_{2}$$",
         colldat.parametros_titulos[1]: "$$ 0.1  $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]:""
         },
        {colldat.parametros_titulos[0]: "$$ k_{3}$$",
         colldat.parametros_titulos[1]: "$$ 2e-7 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]:""

         },
        {colldat.parametros_titulos[0]: "$$ k_{4}$$",
         colldat.parametros_titulos[1]: "$$ 0.5 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ k_{5}$$",
         colldat.parametros_titulos[1]: "$$ 5 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ k_{6}$$",
         colldat.parametros_titulos[1]: "$$ 100 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]:""
         },
        ],
    6: [{colldat.parametros_titulos[0]: "$$ \\beta_{1} $$",
         colldat.parametros_titulos[1]: "$$ 1.120e-2 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]: ""
         },
        {colldat.parametros_titulos[0]: "$$ \\beta_{2} $$",
         colldat.parametros_titulos[1]: "$$ 6.8e-3 $$",
         colldat.parametros_titulos[2]: "",
         colldat.parametros_titulos[3]: "",
         colldat.parametros_titulos[4]: "",
         colldat.parametros_titulos[5]: "",
         colldat.parametros_titulos[6]:"",
         }
        ]

}


