import numpy as np
import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import nav_bar_ODE as nav
import content_interface as ctif
import collect_data as colldat
import pandas as pd
from dash.exceptions import PreventUpdate
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache
import odes_solver as ODE_sol
import constantes as const
import time

import plotly.graph_objects as go


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

cache = diskcache.Cache('cache')
long_callback_manager = DiskcacheLongCallbackManager(cache)

application = dash.Dash(__name__,external_scripts=external_scripts,external_stylesheets=[dbc.themes.BOOTSTRAP], prevent_initial_callbacks=True)
server=application.server

"""
1. Create componenents for the app

"""
# Barra de navegacion
app_nav = nav.nav_bar()
# Contenido del modelo
value_change = 0
value_change_2 = 0
value_change_3=0
#Grafico de la data
graph = "my_graph"
#Grafico integrador
graph2 = "my_graph_2"
#Grafico funcion objetiva
graph3="my_graph_3"
#Grafico proceso de algoritmo(solo PSO)
graph4="my_graph_4"
#Grafico fitting  experimental with simulated data
graph5="my_graph_5"
#Nombre del algoritmo
algo="t"
#Model data card
app_cont_1 = ctif.model_data(value_change, graph)
#Model description card
app_cont_2 = ctif.model_description(value_change_2, graph2)
#Optimización card
app_cont_3=ctif.algoritmo_proceso(algo,value_change_3,graph3,graph4,graph5)
#ID dropdown
id = "dpdw"
# Barra lateral
app_sidebar = nav.side_bar(id)

"""
2. App layout 
"""



application.layout = dbc.Card([
    dbc.Container([
        dbc.Row(
            [
                dbc.Col(
                    app_sidebar, sm=2, style=nav.NAV_BAR_STYLE
                ),

                dbc.Col([app_nav, app_cont_1, app_cont_2,app_cont_3], style=nav.NAV_BAR_STYLE, sm=10),

            ]
        ),

    ])

], style=nav.NAV_BAR_STYLE
)


@application.callback(
    Output(graph, 'figure'),
    Input(id, 'value')
)
def update_graph(case):
    dff = {'t': [0, 1, 2, 3, 4], 'y': [1, 1, 1, 1, 1]}
    fig = px.line(data_frame=dff, x='t', y='y', color='t')
    if case is None or case == '0':
        dff = {'t': [0, 1, 2, 3, 4], 'y': [1, 1, 1, 1, 1]}
        fig = px.line(data_frame=dff, x='t', y='y', color='t')
    else:
        dff = colldat.model_data[int(case)]
        if case == '1':

            fig = px.scatter(data_frame=dff, x='t', y='y1', color='t')
            fig.update_yaxes(
                autorange=True,
                type='linear'
            )
            fig.update_xaxes(
                autorange=True,
                type='linear'
            )
        if case == '2':

            fig = px.scatter(data_frame=dff, x='tiempo', y='T', color='tiempo')
            fig.update_yaxes(
                autorange=True,
                type='linear'
            )
            fig.update_xaxes(
                autorange=True,
                type='linear'
            )
        if case == '3':

            fig = px.scatter(data_frame=dff, x='tiempo', y='T', color='tiempo')
            fig.update_yaxes(
                autorange=True,
                type='linear'
            )
            fig.update_xaxes(
                autorange=True,
                type='linear'
            )
        if case == '4':

            fig = px.scatter(data_frame=dff, x='tiempo', y='T', color='tiempo')
            fig.update_yaxes(
                autorange=True,
                type='linear'
            )
            fig.update_xaxes(
                autorange=True,
                type='linear'
            )
        if case == '5':

            fig = px.scatter(data_frame=dff, x='time', y='lv', color='time')
            fig.update_yaxes(
                autorange=True,
                type='linear'
            )
            fig.update_xaxes(
                autorange=True,
                type='linear'
            )
        if case == '6':

            fig = px.scatter(data_frame=dff, x='t', y='vectProm', color='t')
            fig.update_yaxes(
                autorange=True,
                type='linear'
            )
            fig.update_xaxes(
                autorange=True,
                type='linear'
            )

    return fig


@application.callback(
    Output("data_model_1", 'columns'),
    Input(id, 'value')
)
def update_dataTable(case):
    dfc = {'t': [0, 1, 2, 3, 4], 'y': [1, 1, 1, 1, 1]}
    if case is None:
        raise PreventUpdate
    else:
        if case == "0":
            dfc = [{"name": i, "id": i} for i in dfc]
        if case != "0" and case != 'NoneType':
            dfc = [{"name": i, "id": i} for i in colldat.model_data[int(case)].columns]

    return dfc


@application.callback(
    Output('data_model_1', 'data'),
    Input(id, 'value')
)
def update_dataTable(case):
    dfdt = {'t': [0, 1, 2, 3, 4], 'y': [1, 1, 1, 1, 1]}
    dfd1=pd.DataFrame().from_dict(dfdt)

    if case is None:

        raise PreventUpdate
    else:
        if case == "0":
            dfdt = dfd1.to_dict("records")
        if case != '0':
            dfdt = colldat.model_data[int(case)].to_dict("records")

    return dfdt


@application.callback(
    Output('tilt_model', 'children'),
    Input(id, 'value')

)
def update_title(case):
    dc = nav.names[0, 0]
    if case is None:
        dc = nav.names[0, 0]
    else:
        dc = nav.names[int(case), 0]
    return dc


@application.callback(
    Output('data_model_2', 'data'),
    Input(id, 'value')
)
def update_model_description(case):
    df = pd.DataFrame().from_dict(const.columns_ob[0])
    dfmd = df.to_dict("records")
    if case is None or case == '0':
        raise PreventUpdate

    else:
        if case != '0':
            df = pd.DataFrame(data=const.columns_ob[int(case)])
            dfmd = df.to_dict("records")

    return dfmd


@application.long_callback(
    Output(graph2, 'figure'),
    Input('odeint', 'n_clicks'),
    [State(id, 'value')],
    manager=long_callback_manager

)
def figure_ode_int(n_clicks, case):
    dff = {'t': [0,1,2,3,4], 'y': [1,1,1,1,1]}
    fig = px.line(data_frame=dff, x='t', y='y', color='t')
    fig.update_yaxes(
        autorange=True,
        type='linear'
    )
    fig.update_xaxes(
        autorange=True,
        type='linear'
    )

    if case is None or case == '0':
        dff = {'t': [0, 1, 2, 3, 4], 'y': [1, 1, 1, 1, 1]}
        fig = px.line(data_frame=dff, x='t', y='y', color='t')
    else:
        df = colldat.model_data[int(case)]
        if case == '1':
            ymodel = ODE_sol.odeint(colldat.modelo_enzi, const.init_model_case_1['z'], df['t'], args=tuple(const.init_model_case_1['thetha']))
            #dataframe
            dff = {'t1': df['t'], 'y1': ymodel[:, 0]}
            #graphicsettings
            fig = px.line(dff, x="t1", y="y1", title='ODEINT')
        if case== '2':
            print(1)

    return fig

@application.long_callback(
    Output(graph5, 'figure'),
    Input('algoexe', 'n_clicks'),
    [State(id, 'value'),State('algo_list', 'value')],
    manager=long_callback_manager

)

def algo_PSO(num_clicks,case,algo_num):
    """
     df = {'t': [0, 1, 2, 3, 4], 'y': [1, 1, 1, 1, 1]}
    fig = px.line(data_frame=df, x='t', y='y', color='t')

    if case is None or case == '0':
        raise PreventUpdate
    else:
        df = colldat.model_data[int(case)]
        if case == '1':
            if  list == 'PSO':
                time.sleep(1)
                objectivefunction = ODE_sol.objective_function(const.init_model_case_1['thetha'],
                                                               const.init_model_case_1['z'],
                                                               df['t'], colldat.modelo_enzi,
                                                               df['y1'], 0)
                print(objectivefunction)




                 PSO_vect = ODE_sol.PSO_test(objectivefunction, 50, 50, [-1, -1, -1,-1],[1, 1, 1,1], 0.1, 0.1, 1)

                fig = px.scatter(data_frame=df, x='t', y='y1', color='t')
                fig.update_yaxes(
                    autorange=True,
                    type='linear'
                )
                fig.update_xaxes(
                    autorange=True,
                    type='linear'
                )

                ymodel_1 = ODE_sol.odeint(colldat.modelo_enzi, const.init_model_case_1['z'], df['t'],
                                          args=tuple(PSO_vect[:, 1]))
                # dataframe
                dff = {'t1': df['t'], 'y1': ymodel_1[:, 0]}

                # graphicsettings
                fig = px.line(dff, x="t1", y="y1", title='PSO')
                
                """
    df = {'t': [0, 1, 2, 3, 4], 'y': [1, 1, 1, 1, 1]}
    fig = px.line(data_frame=df, x='t', y='y', color='t')


    return fig

if __name__ == '__main__':
    application.run_server(debug=True)
""""
def prom (exp_1,exp_2,prom) :
    for i in range(len(exp_1)):
       prom.append((exp_1[i]+exp_2[i])/2)
    return prom

 
y2_prom = prom(exp_1, exp_2, vect_prom)  # promedio de las observaciones iniciales
df_1=pd.DataFrame(y2_prom,columns=['PROM_EXP'])
df_1.head()
def data_graph_plotly(x, y,str='def'):
    return go.Scatter(
        x=x,
        y=y,
        name=str
    )

exp1=data_graph_plotly(t_data,exp_1,'exp1')
exp2=data_graph_plotly(t_data,exp_2,'exp2')
prom_1=data_graph_plotly(t_data,y2_prom,'prom')

data=[exp1,exp2,prom_1]
ply.plot(data,filename='basicline.html',auto_open=True)

"""
