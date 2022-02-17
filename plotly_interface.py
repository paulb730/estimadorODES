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

# Author: Paul Benavides
# Derechos Reservados

# Create app

cache = diskcache.Cache('cache')
long_callback_manager = DiskcacheLongCallbackManager(cache)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], prevent_initial_callbacks=True)
server=app.server

"""
1. Create componenents for the app

"""

# Barra de navegacion
app_nav = nav.nav_bar()
# Contenido del modelo
value_change = 0
value_change_2 = 0
graph = "my_graph"
graph2 = "my_graph_2"
app_cont_1 = ctif.model_data(value_change, graph)
app_cont_2 = ctif.model_description(value_change_2, graph2)
id = "dpdw"
# Barra lateral
app_sidebar = nav.side_bar(id)

"""
2. App layout 
"""

app.layout = dbc.Card([
    dbc.Container([
        dbc.Row(
            [
                dbc.Col(
                    app_sidebar, sm=3, style=nav.NAV_BAR_STYLE
                ),

                dbc.Col([app_nav, app_cont_1, app_cont_2], style=nav.NAV_BAR_STYLE, sm=9),

            ]
        ),

    ])

], style=nav.NAV_BAR_STYLE
)


@app.callback(
    Output(graph, 'figure'),
    Input(id, 'value')

)
def update_graph(case):
    dff = px.data.iris()
    fig = px.scatter(data_frame=dff, x='sepal_width', y='sepal_length', color='sepal_width')
    if case == 'NoneType':
        dff = px.data.iris()
        fig = px.scatter(data_frame=dff, x='sepal_width', y='sepal_length', color='sepal_width')
    else:
        if case == '0':
            dff = px.data.iris()
            fig = px.scatter(data_frame=dff, x='sepal_width', y='sepal_length', color='sepal_width')

        if case == '1':
            dff = colldat.model_data[int(case)]

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
            dff = colldat.model_data[int(case)]
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
            dff = colldat.model_data[int(case)]
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
            dff = colldat.model_data[int(case)]
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
            dff = colldat.model_data[int(case)]
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
            dff = colldat.model_data[int(case)]
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


@app.callback(
    Output("data_model_1", 'columns'),
    Input(id, 'value')
)
def update_dataTable(case):
    if case is None:
        dfc = [{"name": i, "id": i} for i in px.data.iris()]
    else:
        if case == "0":
            dfc = [{"name": i, "id": i} for i in px.data.iris()]
        if case != "0" and case != 'NoneType':
            dfc = [{"name": i, "id": i} for i in colldat.model_data[int(case)].columns]

    return dfc


@app.callback(
    Output('data_model_1', 'data'),
    Input(id, 'value')
)
def update_dataTable(case):
    dfdt = px.data.iris().to_dict("records")
    if case is None:
        dfdt = px.data.iris().to_dict("records")
    else:
        if case == "0":
            dfdt = px.data.iris().to_dict("records")
        if case != '0':
            dfdt = colldat.model_data[int(case)].to_dict("records")

    return dfdt


@app.callback(
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


@app.callback(
    Output('data_model_2', 'data'),
    Input(id, 'value')
)
def update_model_description(case):
    df = pd.DataFrame().from_dict(colldat.columns_ob[0])
    dfmd = df.to_dict("records")
    if case is None or case == '0':
        df = pd.DataFrame(data=colldat.columns_ob[0])
        dfmd = df.to_dict("records")
    else:
        if case != '0':
            df = pd.DataFrame(data=colldat.columns_ob[int(case)])
            dfmd = df.to_dict("records")

    return dfmd


@app.long_callback(
    Output(graph2, 'figure'),
    Input('odeint', 'n_clicks'),
    [State(id, 'value')],
    manager=long_callback_manager

)
def figure_ode_int(n_clicks, case):
    dff = px.data.iris()
    fig = px.scatter(data_frame=dff, x='sepal_width', y='sepal_length', color='sepal_width')
    if case is None or case == '0':
        raise PreventUpdate
    else:
        if case == '1':
            df = colldat.model_data[int(1)]
            init_model = {'z': [21.00, 30], 'thetha': [np.random.uniform(0.0001, 1), np.random.uniform(0.0001, 1),
                                                       np.random.uniform(0.0001, 1), np.random.uniform(0.0001, 1)]}
            ymodel = ODE_sol.odeint(colldat.modelo_enzi, init_model['z'], df['t'], args=tuple(init_model['thetha']))
            print(ymodel)
            data = {'t1': df['t'], 'y1': 1}
            dff = {'t1': df['t'], 'y1': ymodel[:, 0]}
            fig = px.line(dff, x="t1", y="y1", title='ODEINT')
            fig.update_yaxes(
                autorange=True,
                type='linear'
            )
            fig.update_xaxes(
                autorange=True,
                type='linear'
            )
        if case == '2':
            df = colldat.model_data[int(2)]

        if case == '3':
            df = colldat.model_data[int(3)]
        if case == '4':
            df = colldat.model_data[int(4)]
        if case == '5':
            df = colldat.model_data[int(5)]
        if case == '6':
            df = colldat.model_data[int(6)]

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
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
