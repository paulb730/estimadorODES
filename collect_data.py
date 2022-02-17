import dash
from dash import dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc
import math

# Se plantearán 4 casos de estudio para la aplicación de las técnicas de Estimación a usar

# Author: Paul Benavides
# Derechos Reservados
"""
1. Caso número uno

"""

model_data = {
    0: px.data.iris(),
    1: pd.read_csv('csv/enzimatic_model.csv', sep=";", usecols=['t', 'y1']),
    2: pd.read_csv('csv/hepatitis.csv', sep=';', usecols=['tiempo', 'Hs', 'He', 'V', 'T']),
    3: pd.read_csv('csv/hepatitis10ruido.csv', sep=";", usecols=['tiempo', 'Hs', 'He', 'V', 'T']),
    4: pd.read_csv('csv/hepatitis15%.csv', sep=";", usecols=['tiempo', 'Hs', 'He', 'V', 'T']),
    5: pd.read_csv('csv/hiv.csv', sep=",", usecols=['time', 'lv']),
    6: pd.read_csv('csv/kinetic_chemistry_model.csv', sep=";", usecols=['t', 'exp_1', 'exp_2', 'vectProm'])

}

model_analytic = ["Variables", "Parametros", "Ecuaciones", "Condiciones Iniciales"]

columns_ob = {

    0: [
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]}
    ],
    1: [
        {"Variables": "y1", "Parametros": "p1", "Ecuaciones": 'dy1dt = p1 * (27.8 - y1) +'
                                                              ' (p4 / 2.6) * (y2 - y1) + (4991 /'
                                                              ' t * math.sqrt(2 * math.pi)) * '
                                                              'math.exp(-0.5*(math.log(t) - p2 / p3))',
         "Condiciones Iniciales": "y(0.1)=(21.00,38.75)"},
        {"Variables": "y2", "Parametros": "p2", "Ecuaciones": ' dy2dt = (p4 / 2.7) * (y1 - y2)',
         " Condiciones Iniciales": ""},
        {"Variables": "t", "Parametros": "p3", "Ecuaciones": "f(t,y,y',p)= -(5.3/7.02*p4*y')- p1*y'"
                                                             "+ (p1*p4/2.7)*(27.8-y)+(1991/t^2)*((p4/2.7)*t-1-w/p3)*math.exp(-w^2/2)   ",
         "Condiciones Iniciales": ""},
        {"Parametros": "p4", }
    ],
    2: [
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]}
    ],
    3: [
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]}
    ],
    4: [
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]}
    ],
    5: [
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]}
    ],
    6: [
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]},
        {"Variables": [0, 0, 0], "Parametros": [0, 0, 0], "Ecuaciones": [0, 0, 0], "Condiciones Iniciales": [0, 0, 0]}
    ]

}


def prom(exp_1, exp_2, prom):
    for i in range(len(exp_1)):
        prom.append((exp_1[i] + exp_2[i]) / 2)
    return prom


def modelo_enzi(z, t, p1, p2, p3, p4):
    """
    :param z: vector de valores iniciales 
    :param t: tiempo
    :param p1: parametro p1
    :param p2: parametro p2
    :param p3: parametro p3 
    :param p4: parametro p4
    :return: 
    """""
    thetha = [p1, p2, p3, p4]
    y1 = z[0]
    y2 = z[1]
    dy1dt = p1 * (27.8 - y1) + (p4 / 2.6) * (y2 - y1) + (4991 / t * math.sqrt(2 * math.pi)) * math.exp(
        -0.5 * (math.log(t) - p2 / p3))
    dy2dt = (p4 / 2.7) * (y1 - y2)

    return [dy1dt, dy2dt]


def model_kinetic_chemistry(z, t, b1, b2):
    """

    :param z: vector de valores iniciales
    :param t: tiempo
    :param b1: paràmetro a estimar b1
    :param b2: paràmetro a estimar b2
    :return: modelo configurado
    """
    y1 = z[0]
    y2 = z[1]

    dy1dt = -b1 * y1
    dy2dt = b1 * y1 - b2 * y2
    return [dy1dt, dy2dt]


def modelo_hep():
    return 1


def modelo_hiv():
    return 1


def modelo_kin():
    return 1


def CASOS(case):
    return dash_table.DataTable(
        id="data_model_1",
        columns=[],
        data=[],
        style_cell={'fontSize': '12px', 'borderRadius': '15px !important'},
        style_header={'background': '#f96d00', 'color': '#fff', 'fontSize': '12px', 'textAlign': 'center'},
        style_table={'overflowY': 'auto', 'height': 'auto', 'width': 'auto'},
        style_data={'whiteSpace': 'normal'},
        virtualization=True
    )


def GRAFICOS(id):
    return dcc.Graph(
        id=str(id),
        figure={},
        className='',
        clickData=None,
        hoverData=None,
        style={
            'width': 'auto', 'height': '70vh'
        },
        config={
            'staticPlot': False,
            'scrollZoom': True,
            'doubleClick': 'reset',
            'showTips': True,
            'displayModeBar': True,
            'watermark': True,
            'responsive': True
        },
    )


def MODELO(case):
    return dash_table.DataTable(
        id="data_model_2",
        columns=[{"name": i, "id": i} for i in model_analytic],
        data=[],
        style_cell={'fontSize': '12px', 'borderRadius': '15px !important'},
        style_header={'background': '#f96d00', 'color': '#fff', 'fontSize': '12px', 'textAlign': 'center'},
        style_table={'height': 'auto', 'width': 'auto', 'borderRadius': '15px !important'},
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
        virtualization=False
    )
