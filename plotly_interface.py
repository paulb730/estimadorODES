import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import nav_bar_ODE as nav
import content_interface as ctif
import collect_data as colldat
import pandas as pd
from dash.exceptions import PreventUpdate
import constantes as const
import time
import numpy as np
# Save data in memory Cache
import json
from dash_extensions.callback import CallbackCache, Trigger
from flask_caching.backends import FileSystemCache
import re



# Author: Paul Benavides
# Derechos Reservados

# Create app

MATHJAX_CDN = '''
https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/
MathJax.js?config=TeX-MML-AM_CHTML'''

external_scripts = [
    {'type': 'text/javascript',
     'id': 'MathJax-script',
     'src': MATHJAX_CDN,
     },
]

# Create APP
application = dash.Dash(__name__, suppress_callback_exceptions=True, external_scripts=external_scripts,
                        external_stylesheets=[dbc.themes.BOOTSTRAP],
                        prevent_initial_callbacks=True)
server = application.server

application.config.suppress_callback_exceptions = True

"""
1. Create componenents for the app

"""
# Barra de navegacion
app_nav = nav.nav_bar()
# Contenido del modelo
value_change = 0
value_change_2 = 0
value_change_3 = 0
# Grafico de la data
graph = "my_graph"
# Grafico integrador
graph2 = "my_graph_2"
# Grafico funcion objetiva
graph3 = "my_graph_3"
# Grafico proceso de algoritmo(solo PSO)
graph4 = "my_graph_4"
# Grafico fitting  experimental with simulated data
graph5 = "my_graph_5"

# Graf_1est
gr_est_0 = "my_graph_est_0"
# Graf_2est
gr_est_1 = "my_graph_est_1"
# Graf_3est
gr_est_2 = "my_graph_est_2"
# Graf_4est
gr_est_3 = "my_graph_est_3"
# Graf_5est
gr_est_4 = "my_graph_est_4"
# Graf_6est
gr_est_5 = "my_graph_est_5"

# Table est 1
tb_est_0 = "my_tb_est_0"

# Nombre del algoritmo
algo = "t"




# Estadisticas
estadisticas = ctif.estadisticas(tb_est_0, gr_est_0, gr_est_1, gr_est_2)

# Model data card
app_cont_1 = ctif.model_data(value_change, graph)

# Tabla de parametros dimension var


# Model description card
app_cont_2 = ctif.model_description(value_change_2, graph2)

# Optimización card
app_cont_3 = ctif.algoritmo_proceso(value_change_3, graph5)

# ID dropdown
id = "dpdw"
# Barra lateral


app_sidebar = nav.side_bar(id)

"""
2. App layout 
"""

application.layout = dbc.Card([

    dbc.Row(
        [
            dbc.Col(
                [app_sidebar], sm=2, style=nav.NAV_BAR_STYLE
            ),

            dbc.Col([app_nav, app_cont_1, app_cont_2, app_cont_3, estadisticas],
                    style=nav.NAV_BAR_STYLE, sm=10),

        ]
    ),

], style=nav.NAV_BAR_STYLE, )



# Validar Botones

@application.callback(
    [Output("but_estadis", "disabled"),Output("odeint", "disabled"),Output("algoexe", "disabled")],
    [Input(id,"value"),Input("algo_list","value")]
)
def enabled_estadisticas_integracion(case, algo_num):
    if (str(case) == '0' and str(algo_num) != '0'):
        is_off_est = True
        is_off_int = True
        is_off_algoexe = True
        return is_off_est, is_off_int, is_off_algoexe

    if not ((case != '0' and case is not None) and (algo_num != '0' and algo_num is not None)):
        is_off_est = True
        is_off_int = True
        is_off_algoexe = True
        return is_off_est, is_off_int, is_off_algoexe
    else:
        is_off_est = False
        is_off_int = False
        is_off_algoexe = False
        return is_off_est, is_off_int, is_off_algoexe


# Descripción de la data experimental y grafico de la misma

@application.callback(
    Output(graph, 'figure'),
    Input(id, 'value')
)
def update_graph(case):
    if case is None or str (case) == '0':
        dff = {'t': [0, 0, 0, 0, 0], 'y': [0, 0, 0, 0, 0]}
        x = 't'
        y = 'y'
    else:
        dff = colldat.model_data[int(case)]
        x = ''
        y = ''
        if case == '1':
            x = 't'
            y = 'y1'

        if case == '2':
            x = 'tiempo'
            y = ['V', 'T']

        if case == '3':
            x = 't_data'
            y = 'x_data'

        if case == '4':
            x = 't'
            y = ['competidor x', 'competidor y', 'competidor z']

        if case == '5':
            x = 'time'
            y = 'lv'

        if case == '6':
            x = 't'
            y = 'vectProm'
    return colldat.FIGURE_Scatter(dff, x, y)


@application.callback(
    [Output("data_model_1", 'columns'), Output('data_model_1', 'data'),Output('main_tit','children')],
    Input(id, 'value')
)
def update_dataTable_names(case):
    dfc = {'t': [0, 0, 0, 0, 0], 'y': [0, 0, 0, 0, 0]}
    dfdt = {'t': [0, 0, 0, 0, 0], 'y': [0, 0, 0, 0, 0]}
    dfd1 = pd.DataFrame().from_dict(dfdt)
    titulo = 'Estimador de  Parámetros en EDOs '
    if case is None:
        dfc = [{"name": i, "id": i} for i in dfc]
        dfdt = dfd1.to_dict("records")
        titulo = 'Estimador de  Parámetros en EDOs '
    else:
        if str(case) == "0":
            dfc = [{"name": i, "id": i} for i in dfc]
            dfdt = dfd1.to_dict("records")
            titulo = 'Estimador de  Parámetros en EDOs '
        if case != "0" :
            dfc = [{"name": i, "id": i} for i in colldat.model_data[int(case)].columns]
            dfdt = colldat.model_data[int(case)].to_dict("records")
            titulo = nav.names[int(case), 0]

    return dfc, dfdt,titulo


# Descripción del Modelo y grafico de la integracion con parametros randomicos


@application.callback(
    Output('name_param_id', 'children'),
    Output('param_inputs_id', 'children'),
    Input(id, 'value'),
    prevent_initial_call=True,
)
def create_inputs(case):
    len = 0
    if case is None or case == '0':
        len = 0
    else:
        if case != "0":
            len = const.VECTOR_DIM_PARAM[int(case)]

    return colldat.NAME_INPUT(len, case), colldat.DATA_INPUT(len, case)


@application.callback(
    Output('data_model_2', 'data'),
    Input(id, 'value')
)
def update_model_description(case):
    df = pd.DataFrame().from_dict(const.columns_ob[0])
    dfmd = df.to_dict("records")
    if case is None or case == '0':

        df = pd.DataFrame(data=const.columns_ob[0])
        dfmd = df.to_dict("records")
    else:
        if case != '0':
            df = pd.DataFrame(data=const.columns_ob[int(case)])
            dfmd = df.to_dict("records")
    return dfmd


@application.callback(
    Output(graph2, 'figure'),
    Input('odeint', 'n_clicks'),
    Input(id, 'value'),
    Input('param_inputs_id', 'children'),
    prevent_initial_call=True,

)
def figure_ode_int(n_clicks, case, value):
    input_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if case is None or n_clicks is None or value=="":
        init_cond = []
        init_tiempo = []
        init_param = []
        filter_parameter_list = []
        parameters_list = []
    else:
        time.sleep(2.0)
        df = colldat.model_data[int(case)]
        df_init = const.initial_conditions_ob[int(case)]
        init_tiempo = ''
        init_cond = df_init['z']
        init_param = df_init['thetha']
        len_param_2 = 0 if case is None or case == '0' else const.VECTOR_DIM_PARAM[int(case)]

        parameters_list = [value[_]["props"]["children"]["props"]["value"] for _ in range(len_param_2)]

        filter_parameter_list = [1 if i is None else i for i in parameters_list]

        if input_id == 'odeint':
            if case == '1':
                init_tiempo = df['t']
            if case == '2':
                init_tiempo = df['tiempo']
            if case == '3':
                init_tiempo = df['t_data']
            if case == '4':
                init_tiempo = df['t']
            if case == '5':
                init_tiempo = df['time']
            if case == '6':
                init_tiempo = df['t']

    return colldat.FIGURE_line_ODEINT(input_id, case, init_cond, init_tiempo, filter_parameter_list)


# Proceso de optimización y uso de algoritmos
@application.callback(
    Output('algo_list', 'value'),
    Input(id, 'value')
)
def clean_algoritmos(case):
    value = '0'

    if case != None:
        value = None
        const.super_data_Vector = [0] * len(const.algoritmos)
        const.super_data_fitness_error_=[["0"]]*len(const.algoritmos)
        const.super_data_fitness_referenciales=["0"]*len(const.algoritmos)
        const.super_data_fitness_estimado=["0"]*len(const.algoritmos)
        const.super_data_valores_estimados=[["0"]]*len(const.algoritmos)
        const.super_data_valores_referenciales=[["0"]]*len(const.algoritmos)
        const.super_data_algoritmo_usado=["None"]*len(const.algoritmos)

    return value


@application.callback(
    [Output('parameter_model', 'data'), Output(graph5, 'figure'), Output("textarea_console", "value")],
    Input('algoexe', 'n_clicks'),
    Input(id, 'value'),
    State('algo_list', 'value'),
    # Parameters
    # PSO
    [State('{}'.format(i), 'value') for i in const.id_edit_form],
    # NSPSO
    [State('{}'.format(i), 'value') for i in const.id_edit_algo_form_0],
    # Bee colony
    [State('{}'.format(i), 'value') for i in const.id_edit_algo_form_1],
    # DE
    [State('{}'.format(i), 'value') for i in const.id_edit_algo_form_2],
    # GA
    [State('{}'.format(i), 'value') for i in const.id_edit_algo_form_3],

    prevent_initial_call=True
)
def algo_proceso(num_clicks, case, algo_num,
                 # PSO
                 slider_w, slider_c1, slider_c2, input_GEN, input_POP,
                 # MONTECARLO
                 gen_NS, popNS,
                 # Bee colony
                 gen_be, popBE, limit_be,
                 # DE
                 gen_1, pop, fr, cr,
                 # GA
                 gen_2, pop_1, m, cr_1):
    # Triggered inputs
    input_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]


    # dataframe y fig de optimizacion
    datafram = []
    param_fitted = []
    dfT1 = []
    error = 0
    data_area = ""
    param_referencial = ""
    # Condiciones iniciales de los modelos
    df_init = const.initial_conditions_ob[0]

    if (case is None and algo_num is None) or (str(case) == '0' and algo_num != None) or (
            case is None and algo_num != None):
        datafram = []
        param_fitted = []
        dfT1 = []

    else:
        # tiempo muerto
        time.sleep(2.0)
        # data de cada modelo de acuerdo al caso de estudio
        df = colldat.model_data[int(case)]
        # creacion de dataframe para la visualizacion de tabla de parametros
        df_1 = pd.DataFrame(data=const.parametros_columns[int(case)])
        # condiciones y parametros teoricos del modelo
        df_init = const.initial_conditions_ob[int(case)]
        # initial guesses de los parametros para comenzar la optimizacion
        df_init_parametros = const.inital_guesses_paramters[int(case)]
        param_fitted = df_init_parametros['thetha']

        if input_id == 'algoexe':


            datafram = df_1.to_dict("records")
            param_referencial = colldat.fitness_referencial(df_init['thetha'], case)
            datafram[0][colldat.parametros_titulos[3]] = "None" if algo_num is None else "$$" + algo_num + "$$"
            datafram[0][colldat.parametros_titulos[6]] = "$$" + str(param_referencial) + "$$"
            vector = np.vectorize(float)
            if algo_num == const.algoritmos[0]:
                if case == '1':

                    dfT1 = df['t']

                    best_param, gbest, swarm = colldat.objective_function_enzi(1 if input_GEN is None else input_GEN,
                                                                               1 if input_POP is None else input_POP,
                                                                               slider_c1,
                                                                               slider_c2,
                                                                               slider_w)


                    best_param_1 = vector(best_param)
                    # Parametros ajustados a la data experimental
                    param_fitted = best_param_1
                    # Valor funcion objetivo

                    datafram[0][colldat.parametros_titulos[5]] = "$$" + str(gbest[0]) + "$$"

                    data_area = str(swarm)

                    for i in range(len(datafram)):
                        datafram[i]['ValorEstimado'] = "$$" + str(param_fitted[i]) + "$$"
                        error = 100 * (np.abs(param_fitted[i] - df_init['thetha'][i]) / np.abs(df_init['thetha'][i]))
                        datafram[i]['Error %'] = "$$" + str(round(error, 3)) + "$$"

                if case == '2':

                    dfT1 = df['tiempo']
                    best_param, gbest, swarm = colldat.objective_function_hepa(1 if input_GEN is None else input_GEN,
                                                                               1 if input_POP is None else input_POP,
                                                                               slider_c1,
                                                                               slider_c2,
                                                                               slider_w)

                    best_param_1 = vector(best_param)
                    param_fitted = best_param_1
                    datafram[0][colldat.parametros_titulos[5]] = "$$" + str(gbest[0]) + "$$"
                    data_area = str(swarm)
                    for i in range(len(datafram)):
                        datafram[i]['ValorEstimado'] = "$$" + str(param_fitted[i]) + "$$"
                        error = (np.abs(param_fitted[i] - df_init['thetha'][i]) / np.abs(df_init['thetha'][i])) * 100
                        datafram[i]['Error %'] = "$$" + str(round(error, 3)) + "$$"

                if case == '3':

                    dfT1 = df['t_data']
                    best_param, gbest, swarm = colldat.objective_function_bench(1 if input_GEN is None else input_GEN,
                                                                                1 if input_POP is None else input_POP,
                                                                                slider_c1,
                                                                                slider_c2,
                                                                                slider_w)

                    best_param_1 = vector(best_param)
                    param_fitted = best_param_1
                    datafram[0][colldat.parametros_titulos[5]] = "$$" + str(gbest[0]) + "$$"
                    data_area = str(swarm)
                    for i in range(len(datafram)):
                        datafram[i]['ValorEstimado'] = "$$" + str(param_fitted[i]) + "$$"
                        error = (np.abs(param_fitted[i] - df_init['thetha'][i]) / np.abs(df_init['thetha'][i])) * 100
                        datafram[i]['Error %'] = "$$" + str(round(error, 3)) + "$$"

                if case == '4':

                    dfT1 = df['t']
                    best_param, gbest, swarm = colldat.objective_function_lotka(1 if input_GEN is None else input_GEN,
                                                                                1 if input_POP is None else input_POP,
                                                                                slider_c1,
                                                                                slider_c2,
                                                                                slider_w)

                    best_param_1 = vector(best_param)
                    param_fitted = best_param_1
                    datafram[0][colldat.parametros_titulos[5]] = "$$" + str(gbest[0]) + "$$"
                    data_area = str(swarm)
                    for i in range(len(datafram)):
                        datafram[i]['ValorEstimado'] = "$$" + str(param_fitted[i]) + "$$"
                        error = (np.abs(param_fitted[i] - df_init['thetha'][i]) / np.abs(df_init['thetha'][i])) * 100
                        datafram[i]['Error %'] = "$$" + str(round(error, 3)) + "$$"
                if case == '5':

                    dfT1 = df['time']
                    best_param, gbest, swarm = colldat.objective_function_HIV(1 if input_GEN is None else input_GEN,
                                                                              1 if input_POP is None else input_POP,
                                                                              slider_c1,
                                                                              slider_c2,
                                                                              slider_w)

                    best_param_1 = vector(best_param)
                    param_fitted = best_param_1
                    datafram[0][colldat.parametros_titulos[5]] = "$$" + str(gbest[0]) + "$$"
                    data_area = str(swarm)
                    for i in range(len(datafram)):
                        datafram[i]['ValorEstimado'] = "$$" + str(param_fitted[i]) + "$$"
                        error = (np.abs(param_fitted[i] - df_init['thetha'][i]) / np.abs(df_init['thetha'][i])) * 100
                        datafram[i]['Error %'] = "$$" + str(round(error, 3)) +  "$$"

                if case == '6':
                    dfT1 = df['t']
                    best_param, gbest, swarm = colldat.objetive_function_kinetic(1 if input_GEN is None else input_GEN,
                                                                                 1 if input_POP is None else input_POP,
                                                                                 slider_c1,
                                                                                 slider_c2,
                                                                                 slider_w)

                    best_param_1 = vector(best_param)
                    param_fitted = best_param_1
                    datafram[0][colldat.parametros_titulos[5]] = "$$" + str(gbest[0]) + "$$"
                    data_area = str(swarm)

                    for i in range(len(datafram)):
                        datafram[i]['ValorEstimado'] = "$$" + str(param_fitted[i]) + "$$"
                        error = 100 * (np.abs(param_fitted[i] - df_init['thetha'][i]) / df_init['thetha'][i])
                        datafram[i]['Error %'] = "$$" + str(error) +  "$$"

            if algo_num == const.algoritmos[1]:
                best_param, gbest, swarm = colldat.objetivo_compare_F(case, const.algoritmos[1], gen_NS,

                                                                      popNS,
                                                                      fr, cr,
                                                                      limit_be)
                best_param_1 = vector(best_param)
                data_area = const.algoritmos[1] + str(swarm)
                param_fitted = best_param_1
                datafram[0][colldat.parametros_titulos[5]] = "$$" + str(gbest[0]) + "$$"

                for i in range(len(datafram)):
                    datafram[i]['ValorEstimado'] = "$$" + str(param_fitted[i]) + "$$"
                    error = (np.abs(param_fitted[i] - df_init['thetha'][i]) / np.abs(df_init['thetha'][i])) * 100
                    datafram[i]['Error %'] = "$$" + str(round(error, 3)) +  "$$"

                if case == '1':
                    dfT1 = df['t']
                if case == '2':
                    dfT1 = df['tiempo']
                if case == '3':
                    dfT1 = df['t_data']
                if case == '4':
                    dfT1 = df['t']
                if case == '5':
                    dfT1 = df['time']
                if case == '6':
                    dfT1 = df['t']

            if algo_num == const.algoritmos[2]:
                best_param, gbest, swarm = colldat.objetivo_compare_F(case, const.algoritmos[2], gen_be,

                                                                      popBE,
                                                                      fr, cr,
                                                                      limit_be
                                                                      )

                best_param_1 = vector(best_param)
                data_area = const.algoritmos[2] + str(swarm)
                param_fitted = best_param_1
                datafram[0][colldat.parametros_titulos[5]] = "$$" + str(gbest[0]) + "$$"

                for i in range(len(datafram)):
                    datafram[i]['ValorEstimado'] = "$$" + str(param_fitted[i]) + "$$"
                    error = (np.abs(param_fitted[i] - df_init['thetha'][i]) / np.abs(df_init['thetha'][i])) * 100
                    datafram[i]['Error %'] = "$$" + str(round(error, 3)) +  "$$"

                if case == '1':
                    dfT1 = df['t']
                if case == '2':
                    dfT1 = df['tiempo']
                if case == '3':
                    dfT1 = df['t_data']
                if case == '4':
                    dfT1 = df['t']
                if case == '5':
                    dfT1 = df['time']
                if case == '6':
                    dfT1 = df['t']

            if algo_num == const.algoritmos[3]:
                best_param, gbest, swarm = colldat.objetivo_compare_F(case, const.algoritmos[3], gen_1,

                                                                      pop,
                                                                      fr, cr,
                                                                      limit_be)

                best_param_1 = vector(best_param)
                param_fitted = best_param_1
                datafram[0][colldat.parametros_titulos[5]] = "$$" + str(gbest[0]) + "$$"
                data_area = str(swarm)
                for i in range(len(datafram)):
                    datafram[i]['ValorEstimado'] = "$$" + str(param_fitted[i]) + "$$"
                    error = (np.abs(param_fitted[i] - df_init['thetha'][i]) / np.abs(df_init['thetha'][i])) * 100
                    datafram[i]['Error %'] = "$$" + str(round(error, 3)) +  "$$"

                if case == '1':
                    dfT1 = df['t']
                if case == '2':
                    dfT1 = df['tiempo']
                if case == '3':
                    dfT1 = df['t_data']
                if case == '4':
                    dfT1 = df['t']
                if case == '5':
                    dfT1 = df['time']
                if case == '6':
                    dfT1 = df['t']

            if algo_num == const.algoritmos[4]:
                best_param, gbest, swarm = colldat.simple_genetic(case, gen_2, pop_1, m, cr_1)
                gbest = vector(gbest)
                best_param_1 = vector(best_param)
                param_fitted = best_param_1[0]
                datafram[0][colldat.parametros_titulos[5]] = "$$" + str(gbest[0][0]) + "$$"
                data_area = str(swarm)

                for i in range(len(datafram)):
                    datafram[i]['ValorEstimado'] = "$$" + str(param_fitted[i]) + "$$"
                    error = (np.abs(param_fitted[i] - df_init['thetha'][i]) / np.abs(df_init['thetha'][i])) * 100
                    datafram[i]['Error %'] = "$$" + str(round(error, 3)) + "$$"

                if case == '1':
                    dfT1 = df['t']
                if case == '2':
                    dfT1 = df['tiempo']
                if case == '3':
                    dfT1 = df['t_data']
                if case == '4':
                    dfT1 = df['t']
                if case == '5':
                    dfT1 = df['time']
                if case == '6':
                    dfT1 = df['t']

    return datafram, colldat.Figure_fitted_OPT(input_id, case, algo_num, df_init['z'], dfT1, param_fitted), data_area






@application.callback(

    Output("PSO_edit_container", "is_open"),
    Output("MC_edit_container", "is_open"),
    Output("BEE_edit_container", "is_open"),
    Output("DEALGO_edit_container", "is_open"),
    Output("GAALGO_edit_container", "is_open"),
    Input("algo_list", 'value'),
    [State("PSO_edit_container", "is_open"),
     State("MC_edit_container", "is_open"),
     State("BEE_edit_container", "is_open"),
     State('DEALGO_edit_container', 'is_open'),
     State("GAALGO_edit_container", "is_open")],
)
def toggle_collapse_ALGO(value, is_open_PSO, is_open_NSPSO, is_open_BEE, is_open_algo_de, is_open_algo_ga):
    is_open_PSO = False
    is_open_NSPSO = False
    is_open_BEE = False
    is_open_algo_de = False
    is_open_algo_ga = False
    if value == const.algoritmos[0]:
        return not is_open_PSO, is_open_NSPSO, is_open_BEE, is_open_algo_de, is_open_algo_ga
    else:
        is_open_PSO = False
        is_open_NSPSO = False
        is_open_BEE = False
        is_open_algo_de = False
        is_open_algo_ga = False
        if value == const.algoritmos[1]:
            return is_open_PSO, not is_open_NSPSO, is_open_BEE, is_open_algo_de, is_open_algo_ga
        else:
            is_open_PSO = False
            is_open_NSPSO = False
            is_open_BEE = False
            is_open_algo_de = False
            is_open_algo_ga = False

            if value == const.algoritmos[2]:
                return is_open_PSO, is_open_NSPSO, not is_open_BEE, is_open_algo_de, is_open_algo_ga
            else:
                is_open_PSO = False
                is_open_NSPSO = False
                is_open_BEE = False
                is_open_algo_de = False
                is_open_algo_ga = False
                if value == const.algoritmos[3]:
                    return is_open_PSO, is_open_NSPSO, is_open_BEE, not is_open_algo_de, is_open_algo_ga
                else:
                    is_open_PSO = False
                    is_open_NSPSO = False
                    is_open_BEE = False
                    is_open_algo_de = False
                    is_open_algo_ga = False
                    if value == const.algoritmos[4]:
                        return is_open_PSO, is_open_NSPSO, is_open_BEE, is_open_algo_de, not is_open_algo_ga
                    else:
                        is_open_PSO = False
                        is_open_NSPSO = False
                        is_open_BEE = False
                        is_open_algo_de = False
                        is_open_algo_ga = False

    return is_open_PSO, is_open_NSPSO, is_open_BEE, is_open_algo_de, is_open_algo_ga







#Estadisticas
@application.callback(
    Output(const.id_estadisticas, "is_open"),
    [Input("but_estadis", "n_clicks")],
    [State(const.id_estadisticas, "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open




# Create (server side) cache. Works with any flask caching backend.



cc = CallbackCache(cache=FileSystemCache(cache_dir='cache'), session_check=True, instant_refresh=True)

@cc.cached_callback(
    Output("memory_storage_param_table", "data"),
    [Input("parameter_model", "data"), Input("algoexe", "n_clicks"), Input(id, 'value'), Input("algo_list", 'value'), ],

)
def store_data(data_p, n_clicks, case, value_algo):
    time.sleep(1)  # sleep to emulate a database call / a long calculation

    PSO = ""
    NS_PSO = ""
    Colonia_Abejas = ""
    DE = ""
    GA = ""
    split_data = [{'Algoritmo Usado': "0"}]

    super_vector = const.super_data_Vector
    super_vector_valores_referenciales=const.super_data_valores_referenciales
    super_vector_valores_estimados=const.super_data_valores_estimados
    super_vector_fitness_estimada=const.super_data_fitness_estimado
    super_vector_fitness_referencial= const.super_data_fitness_referenciales
    super_vector_error=const.super_data_fitness_error_
    super_vector_algoritmos=const.super_data_algoritmo_usado
    super_vector_param_names=const.super_data_param_name

    input_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if str(case) == '0' or case is None or value_algo is None:
        raise PreventUpdate
    else:
        if data_p is not []:
            split_data = data_p
            if case != '0':
                if input_id == "algoexe":
                    if split_data[0]['Algoritmo Usado'] == "$$PSO$$":
                        PSO = data_p
                        super_vector_param_names[0] = [PSO[_][colldat.parametros_titulos[0]] for _ in range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_valores_referenciales[0] = [PSO[_][colldat.parametros_titulos[1]] for _ in
                                                                 range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_valores_estimados[0] = [PSO[_][colldat.parametros_titulos[2]] for _ in
                                                             range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_algoritmos[0] = PSO[0][colldat.parametros_titulos[3]]
                        super_vector_error[0] = [PSO[_][colldat.parametros_titulos[4]] for _ in
                                                 range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_fitness_estimada[0] = PSO[0][colldat.parametros_titulos[5]]
                        super_vector_fitness_referencial[0] = PSO[0][colldat.parametros_titulos[6]]


                    if split_data[0]['Algoritmo Usado'] == "$$Grey Wolf Optimizer$$":
                        NS_PSO = data_p
                        super_vector_param_names[1] = [NS_PSO[_][colldat.parametros_titulos[0]] for _ in
                                                       range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_valores_referenciales[1] = [NS_PSO[_][colldat.parametros_titulos[1]] for _ in
                                                                 range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_valores_estimados[1] = [NS_PSO[_][colldat.parametros_titulos[2]] for _ in
                                                             range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_error[1] = [NS_PSO[_][colldat.parametros_titulos[4]] for _ in
                                                 range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_fitness_estimada[1] = NS_PSO[0][colldat.parametros_titulos[5]]
                        super_vector_fitness_referencial[1] = NS_PSO[0][colldat.parametros_titulos[6]]
                        super_vector_algoritmos[1] = NS_PSO[0][colldat.parametros_titulos[3]]
                    if split_data[0]['Algoritmo Usado'] == "$$Colonia Abejas$$":
                        Colonia_Abejas = data_p
                        super_vector_param_names[2] = [Colonia_Abejas[_][colldat.parametros_titulos[0]] for _ in
                                                       range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_valores_referenciales[2] = [Colonia_Abejas[_][colldat.parametros_titulos[1]] for _
                                                                 in
                                                                 range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_valores_estimados[2] = [Colonia_Abejas[_][colldat.parametros_titulos[2]] for _ in
                                                             range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_error[2] = [Colonia_Abejas[_][colldat.parametros_titulos[4]] for _ in
                                                 range(const.VECTOR_DIM_PARAM[int(case)])]

                        super_vector_fitness_estimada[2] = Colonia_Abejas[0][colldat.parametros_titulos[5]]
                        super_vector_fitness_referencial[2] = Colonia_Abejas[0][colldat.parametros_titulos[6]]
                        super_vector_algoritmos[2] = Colonia_Abejas[0][colldat.parametros_titulos[3]]
                    if split_data[0]['Algoritmo Usado'] == "$$Differential Evolution$$":
                        DE = data_p
                        super_vector_param_names[3] = [DE[_][colldat.parametros_titulos[0]] for _ in
                                                       range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_valores_referenciales[3] = [DE[_][colldat.parametros_titulos[1]] for _ in
                                                                 range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_valores_estimados[3] = [DE[_][colldat.parametros_titulos[2]] for _ in
                                                             range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_error[3] = [DE[_][colldat.parametros_titulos[4]] for _ in
                                                 range(const.VECTOR_DIM_PARAM[int(case)])]

                        super_vector_fitness_estimada[3] = DE[0][colldat.parametros_titulos[5]]
                        super_vector_fitness_referencial[3] = DE[0][colldat.parametros_titulos[6]]
                        super_vector_algoritmos[3] = DE[0][colldat.parametros_titulos[3]]
                    if split_data[0]['Algoritmo Usado'] == "$$Simple Genético$$":
                        GA = data_p
                        super_vector_param_names[4] = [GA[_][colldat.parametros_titulos[0]] for _ in
                                                       range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_valores_referenciales[4] = [GA[_][colldat.parametros_titulos[1]] for _ in
                                                                 range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_valores_estimados[4] = [GA[_][colldat.parametros_titulos[2]] for _ in
                                                             range(const.VECTOR_DIM_PARAM[int(case)])]
                        super_vector_error[4] = [GA[_][colldat.parametros_titulos[4]] for _ in
                                                 range(const.VECTOR_DIM_PARAM[int(case)])]

                        super_vector_fitness_estimada[4] = GA[0][colldat.parametros_titulos[5]]
                        super_vector_fitness_referencial[4] = GA[0][colldat.parametros_titulos[6]]
                        super_vector_algoritmos[4] = GA[0][colldat.parametros_titulos[3]]

        else:
            raise PreventUpdate

    return super_vector_algoritmos,\
           super_vector_fitness_estimada,\
           super_vector_fitness_referencial, \
           super_vector_valores_referenciales, \
           super_vector_valores_estimados, \
           super_vector_error,\
           super_vector_param_names


@cc.callback(
    [Output(gr_est_0, "figure"),Output(gr_est_1,"figure"),Output(gr_est_2,"figure")],
    [Input("memory_storage_param_table", 'modified_timestamp'), Input("algoexe", "n_clicks")],
    [State("memory_storage_param_table", 'data'), State(id, 'value'), State('algo_list', 'value'), ]
)
def on_data(ts, n_clicks, data, case, algonum):
    input_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    algo_data_0 = ["None"]
    graph_data_0 = [0]
    graph_data_1=[0]
    algo_data_0 = "0"

    if ts is None or (case is None and algonum is None) or (str(case) == '0' and algonum != None) or (case is None and algonum != None):
        raise PreventUpdate
    else:
        # G(theta) data for PSO
        algo_data_0 = data[0]
        graph_data_0 = data[1]
        graph_data_1=data[2]
        graph_data_2=data[3]
        graph_data_3=data[4]
        graph_data_4=data[5]
        graph_data_5=data[6]

        algo_data_0 = [re.sub('[!@#$]', '', _) for _ in algo_data_0]
        graph_data_0 = [float(re.sub('[!@#$]', '', _)) for _ in graph_data_0]
        graph_data_1=[float(re.sub('[!@#$]', '', _)) for _ in graph_data_1]

        


    return colldat.Figure_estadistics_1(case, algonum, algo_data_0, graph_data_0),\
           colldat.Figure_estadistics_2(case,algonum,algo_data_0,graph_data_0,graph_data_1),\
           colldat.Figure_estadistics_3(case,algonum,graph_data_2,graph_data_3,graph_data_5,graph_data_4)


# This call registers the callbacks on the application.
cc.register(application)

if __name__ == '__main__':
    application.run_server(debug=True)
