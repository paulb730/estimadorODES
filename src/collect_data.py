from dash import dash_table
import pandas as pd
import plotly.express as px
from dash import dcc
import math
import plotly.graph_objects as go
import numpy as np
from src import constantes as const
from src import Funcionamiento_PSO as f_pso
import dash_bootstrap_components as dbc
from dash import html
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src import odes_solver as ODE_sol


# Author: Paul Benavides
# Derechos Reservados


model_data = {
    0: px.data.iris(),
    1: pd.read_csv('csv/enzimatic_model.csv', sep=";", usecols=['t', 'y1']),
    2: pd.read_csv('csv/hepatitis.csv', sep=';', usecols=['tiempo', 'Hs', 'He', 'V', 'T']),
    3: pd.read_csv('csv/benchmark.csv', sep=";", usecols=['t_data', 'x_data']),
    4: pd.read_csv('csv/lotka_volterra_model.csv', sep=";",
                   usecols=['period', 't', 'competidor x', 'competidor y', 'competidor z']),
    5: pd.read_csv('csv/hiv.csv', sep=";", usecols=['time', 'lv']),
    6: pd.read_csv('csv/kinetic_chemistry_model.csv', sep=";", usecols=['t', 'exp_1', 'exp_2', 'vectProm'])

}

model_analytic = ["Variables", "Parametros", "Ecuaciones", "Condiciones Iniciales"]

parametros_titulos = ["Parametros", "Valor Referencial", "ValorEstimado", "Algoritmo Usado","", "$$ g( θ ) (Parametros Estimados) $$",
                      "$$ g( θ ) (Parámetros Referenciales) $$"]

parametros_cond_tit=["$$ Algoritmos $$","$$Condiciones$$"]


estadisticas_tit=["Parametros",""]

def prom(exp_1, exp_2):
    prom_1=[]
    for i in range(len(exp_1)):
        prom_1.append((exp_1[i] + exp_2[i]) / 2)
    return prom_1





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
    dy1dt = p1 * (27.8 - y1) + (p4 / 2.6) * (y2 - y1) + (4991 / (t * math.sqrt(2 * math.pi))) * math.exp(
        -0.5 * math.pow((math.log(t) - p2) / p3, 2))
    dy2dt = (p4 / 2.7) * (y1 - y2)

    return [dy1dt, dy2dt]


def modelo_hepatitis(z, t, beta_t, k, p, mu_i, mu_T, mu_v, delta, t_max):
    """
    :param z:
    :param t:
    :param beta_s:
    :param beta_t:
    :param k:
    :param p:
    :param mu_s:
    :param mu_i:
    :param mu_t:
    :param mu_v:
    :param delta:
    :return: dHs/dt;dHi/dt;dV/dt;dT/dt
    """
    Hs = z[0]
    Hi = z[1]
    V = z[2]
    T = z[3]

    dHsdt = 15.4 - k * Hs * V - 3.08e-3 * Hs
    dHidt = k * Hs * V - delta * Hi * T - mu_i * Hi
    dVdt = p * Hi - mu_v * V
    dTdt = beta_t * (1 - (T / t_max)) * V - mu_T * T
    return [dHsdt, dHidt, dVdt, dTdt]


def modelo_benchmark(zinit, t, a, b, c, d):
    x = zinit[0]
    y = zinit[1]
    z = zinit[2]
    dxdt = y
    dydt = z
    dzdt = a * z + b * y + c * x + d
    return [dxdt, dydt, dzdt]


def modelo_lotka(zinit, t, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33):
    x = zinit[0]
    y = zinit[1]
    z = zinit[2]
    dxdt = x * (a10 + a11 * x + a12 * y + a13 * z)
    dydt = y * (a20 + a21 * x + a22 * y + a23 * z)
    dzdt = z * (a30 + a31 * x + a32 * y + a33 * z)
    return [dxdt, dydt, dzdt]


def modelo_hiv(zinit, t, kr1, kr2, kr3, kr4, kr5, kr6):
    H = zinit[0]
    I = zinit[1]
    V = zinit[2]
    dHdt = kr1 - kr2 * H - kr3 * H * V
    dIdt = kr3 * H * V - kr4 * I
    dVdt = -kr3 * H * V - kr5 * V + kr6 * I
    return [dHdt, dIdt, dVdt]


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


def CASOS(case):
    return dash_table.DataTable(
        id="data_model_1",
        columns=[],
        data=[],
        style_cell={'fontSize': '12px', 'borderRadius': '15px !important'},
        style_header={'background': '#008cf9', 'color': '#fff', 'fontSize': '12px', 'textAlign': 'center'},
        style_table={'overflowY': 'auto', 'height': 'auto', 'width': 'auto', 'overflowX': 'auto'},
        style_data={'whiteSpace': 'normal'},
        virtualization=True
    )


def algo_table_cond(id):
    return dash_table.DataTable(
        id=str(id),
        columns=[{"name": i, "id": i} for i in parametros_cond_tit],
        data=[],
        style_cell={'fontSize': '12px', 'borderRadius': '15px !important'},
        style_header={'background': '#008cf9', 'color': '#fff', 'fontSize': '12px', 'textAlign': 'center'},
        style_table={'overflowY': 'auto', 'height': 'auto', 'width': 'auto', 'overflowX': 'auto'},
        style_data={'whiteSpace': 'normal'},
        virtualization=True
    )


def GRAFICOS(id):
    return dcc.Graph(
        id=str(id),
        figure={},
        className='graphstyle',
        clickData=None,
        hoverData=None,
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


def NAME_INPUT(len_param: int, case):
    return [html.Th(const.parametros_columns[int(case)][_]['Parametros']) for _ in range(len_param)]


def DATA_INPUT(len_param: int, case):
    return [html.Th(dbc.Input(
        id=str(_), type="number",
        value=float(re.sub('[!@#$]', '', const.parametros_columns[int(case)][_][parametros_titulos[1]]))
        , step=0.000000000001, placeholder=re.sub('[!@#$]', '', const.parametros_columns[int(case)][_][parametros_titulos[1]])))
        for _ in range(len_param)]


def PARAM_INPUT(len_param: int, case):
    layout_test = html.Table \
            ([
            html.Tbody(
                [html.Tr(
                    "", id="name_param_id")] +
                [html.Tr("", id="param_inputs_id")]
            )
        ])
    return layout_test


def GRAFICOS_2(id):
    return dcc.Graph(
        id=str(id),
        figure={},
        className='graphstyle',
        clickData=None,
        hoverData=None,
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


def GRAFICOS_3(id):
    return dcc.Graph(
        id=str(id),
        figure={},
        className='graphstyle',
        clickData=None,
        hoverData=None,
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


def FIGURE_Scatter(dff, x, y):
    fig = px.scatter(data_frame=dff, x=x, y=y)
    fig.update_yaxes(
        autorange=True,
        type='linear'
    )
    fig.update_xaxes(
        autorange=True,
        type='linear'
    )
    return fig


def FIGURE_line_ODEINT(id, case, condicionesiniciales, tiempo, parametros):
    fig_1 = {}
    if case == None or case == '0':
        fig_1 = {}
    else:
        if id == "odeint":
            if case is None:
                dff = {'t1': [0, 0, 0, 0, 0], 'y': [0, 0, 0, 0, 0]}
                fig_1 = {}
            if case == '0':
                dff = {'t1': [1, 1, 1, 1, ], 'y1': [1, 2, 3, 4]}
                fig_1 = px.line(dff, x="t1", y="y1", title='', markers=True)
            if case == '1':
                del fig_1
                ymodel = ODE_sol.odeint(modelo_enzi, condicionesiniciales, tiempo, args=tuple(parametros))
                dff = {'t1': tiempo, 'Y1': ymodel[:, 0], 'Y2': ymodel[:, 1]}
                fig_1 = px.line(dff, x="t1", y=['Y1', 'Y2'], title='Módelo Enzimático', markers=True)
            if case == '2':
                del fig_1
                ymodel = ODE_sol.odeint(modelo_hepatitis, condicionesiniciales, tiempo, args=tuple(parametros))
                dff = {'t1': tiempo, 'Hs': ymodel[:, 0], 'Hi': ymodel[:, 1], 'V': ymodel[:, 2], 'T': ymodel[:, 3]}
                fig_1 = px.line(dff, x="t1", y=['Hs', 'Hi', 'V', 'T'], title='Dinámica Hepatitis', markers=True)
            if case == '3':
                del fig_1
                ymodel = ODE_sol.odeint(modelo_benchmark, condicionesiniciales, tiempo, args=tuple(parametros))
                dff = {'t1': tiempo, 'X': ymodel[:, 0], 'Y': ymodel[:, 1], 'Z': ymodel[:, 2]}
                fig_1 = px.line(dff, x="t1", y=['X', 'Y', 'Z'], title='Benchmark', markers=True)
            if case == '4':
                del fig_1
                ymodel = ODE_sol.odeint(modelo_lotka, condicionesiniciales, tiempo, args=tuple(parametros))
                dff = {'t1': tiempo, 'competidor_X': ymodel[:, 0], 'competidor_Y': ymodel[:, 1],
                       'competidor_Z': ymodel[:, 2]}
                fig_1 = px.line(dff, x="t1", y=['competidor_X', 'competidor_Y', 'competidor_Z'], title='Lotka Volterra',
                                markers=True)
            if case == '5':
                del fig_1
                ymodel = ODE_sol.odeint(modelo_hiv, condicionesiniciales, tiempo, args=tuple(parametros))
                log_ymodel_V = np.log10(ymodel[:, 2])
                log_ymodel_H = np.log10(ymodel[:, 0])
                # log_ymodel_I = np.log10(ymodel[:, 1])
                dff = {'t1': tiempo, 'H': log_ymodel_H, 'I': ymodel[:, 1], 'V': log_ymodel_V}
                fig_1 = px.line(dff, x='t1', y=['H', 'I', 'V'], title='Dinámica HIV', log_y=True, markers=True)
            if case == '6':
                del fig_1
                ymodel = ODE_sol.odeint(model_kinetic_chemistry, condicionesiniciales, tiempo, args=tuple(parametros))
                dff = {'t1': tiempo, 'Y1': ymodel[:, 0], 'Y2': ymodel[:, 1]}
                fig_1 = px.line(dff, x="t1", y=['Y1', 'Y2'], title='Cinética de una reacción Química', markers=True)
        else:
            fig_1 = {}

    return fig_1


def Figure_fitted_OPT(id, case, algo_num, condicionesiniciales, tiempo, parametros):
    fig = {}
    if case == None or case == '0':
        fig = {}
    else:
        if id == 'algoexe':
            if case == None or algo_num == None:
                fig = {}
            else:
                dff_exp = model_data[int(case)]
                if case == '0':
                    fig = {}
                if case == '1':
                    ymodel = ODE_sol.odeint(modelo_enzi, condicionesiniciales, tiempo, args=tuple(parametros))
                    dff = {'t1': tiempo, 'Y1': ymodel[:, 0], 'Y2': ymodel[:, 1]}
                    fig_line = px.line(dff, x="t1", y=['Y1', 'Y2'], title='Módelo Enzimático', markers=True)
                    fig_scatter = px.scatter(dff_exp, x='t', y="y1", color='t', color_continuous_scale="HSV", )
                    fig = go.Figure(data=fig_line.data + fig_scatter.data).update_layout(
                        coloraxis=fig_scatter.layout.coloraxis
                        , legend=dict(yanchor="top", y=0.99,
                                      xanchor="left", x=0.01))
                if case == '2':
                    ymodel = ODE_sol.odeint(modelo_hepatitis, condicionesiniciales, tiempo, args=tuple(parametros))
                    dff = {'t1': tiempo, 'Hs': ymodel[:, 0], 'He': ymodel[:, 1], 'V': ymodel[:, 2], 'T': ymodel[:, 3]}
                    fig_line = px.line(dff, x="t1", y=['Hs', 'He', 'V', 'T'], title='Dinámica Hepatitis', markers=True)
                    fig_scatter = px.scatter(dff_exp, x='tiempo', y=['V', 'T'], color='tiempo',
                                             color_continuous_scale="turbo", )
                    fig = go.Figure(data=fig_line.data + fig_scatter.data).update_layout(
                        coloraxis=fig_scatter.layout.coloraxis
                        , legend=dict(yanchor="top", y=0.99,
                                      xanchor="left", x=0.01))
                if case == '3':
                    ymodel = ODE_sol.odeint(modelo_benchmark, condicionesiniciales, tiempo, args=tuple(parametros))
                    dff = {'t1': tiempo, 'X': ymodel[:, 0], 'Y': ymodel[:, 1], 'Z': ymodel[:, 2]}
                    fig_line = px.line(dff, x="t1", y=['X', 'Y', 'Z'], title='Benchmark', markers=True)
                    fig_scatter = px.scatter(dff_exp, x='t_data', y="x_data", color='t_data',
                                             color_continuous_scale="inferno")
                    fig = go.Figure(data=fig_line.data + fig_scatter.data).update_layout(
                        coloraxis=fig_scatter.layout.coloraxis
                        , legend=dict(yanchor="top", y=0.99,
                                      xanchor="left", x=0.01))
                if case == '4':
                    ymodel = ODE_sol.odeint(modelo_lotka, condicionesiniciales, tiempo, args=tuple(parametros))
                    dff = {'t1': tiempo, 'Competidor X': ymodel[:, 0], 'Competidor Y': ymodel[:, 1],
                           'Competidor Z': ymodel[:, 2]}
                    fig_line = px.line(dff, x='t1', y=['Competidor X', 'Competidor Y', 'Competidor Z'],
                                       title='Lotka-Volterra',
                                       markers=True)
                    fig_scatter = px.scatter(dff_exp, x='t', y=['competidor x', 'competidor y', 'competidor z'])
                    fig = go.Figure(data=fig_line.data + fig_scatter.data).update_layout(
                        coloraxis=fig_scatter.layout.coloraxis)
                if case == '5':
                    ymodel = ODE_sol.odeint(modelo_hiv, condicionesiniciales, tiempo, args=tuple(parametros))
                    log_ymodel = np.log10(ymodel[:, 2])
                    dff = {'t1': tiempo, 'H': ymodel[:, 0], 'I': ymodel[:, 1], 'V': log_ymodel}
                    fig_line = px.line(dff, x='t1', y='V', title='Dinámica VIH', log_y=True, markers=True)
                    fig_scatter = px.scatter(dff_exp, x='time', y='lv', color='time', color_continuous_scale="inferno")
                    fig = go.Figure(data=fig_line.data + fig_scatter.data).update_layout(
                        coloraxis=fig_scatter.layout.coloraxis)
                if case == '6':
                    ymodel = ODE_sol.odeint(model_kinetic_chemistry, condicionesiniciales, tiempo,
                                            args=tuple(parametros))
                    dff = {'t1': tiempo, 'exp1': ymodel[:, 0], 'exp2': ymodel[:, 1]}
                    fig_line = px.line(dff, x='t1', y=['exp1', 'exp2'], title='Modelo de Cinética Química ',
                                       markers=True)
                    fig_scatter = px.scatter(dff_exp, x='t', y=['exp_1', 'exp_2', 'vectProm'])
                    fig = go.Figure(data=fig_line.data + fig_scatter.data).update_layout(
                        coloraxis=fig_scatter.layout.coloraxis)
        else:
            fig = {}

    return fig




def Figure_estadistics_1(case, algo_num, data, data_2):
    fig = {}
    if case == None or case == '0' or algo_num == None or (case == None and algo_num != None) \
               or (case != None and algo_num == None):

        fig = {}

    else:

        dff = {"algo": data, "g(thetha)": data_2}
        fig = px.bar(dff, x="algo", y="g(thetha)", color="algo",title="Algoritmos vs Función Objetiva")

    return fig



def Figure_estadistics_2(case,algo_num,data,data_2,data_3):
    fig={}

    if case == None or case == '0' or algo_num == None or (case == None and algo_num != None) \
            or (case != None and algo_num == None):

        fig = {}

    else:


        dff = {"algo": data, "g(thetha)": data_2, "g(thetha) REF": data_3}
        fig1=px.bar(dff,x='algo',y="g(thetha) REF",width=10, color='g(thetha) REF', color_continuous_scale=px.colors.qualitative.G10 )
        fig2=px.bar(dff,x="algo",y="g(thetha)",width=10,color='g(thetha)',color_continuous_scale=px.colors.qualitative.Light24 )
        fig = go.Figure(data=fig1.data+fig2.data).update_layout( coloraxis=fig2.layout.coloraxis, barmode='group', bargap=0.55, bargroupgap=0.1)

    return fig


def Figure_estadistics_3(case,algo_num,data_2,data_3,data_4,data_5):
    fig = {}


    if case == None or case == '0' or algo_num == None or (case == None and algo_num != None) \
            or (case != None and algo_num == None):

        fig = {}

    else:
        #Valores referenciales
        bp_1 = data_2[0][:]
        bp_2 = data_2[1][:]
        bp_3 = data_2[2][:]
        bp_4 = data_2[3][:]
        bp_5 = data_2[4][:]

        bp_1=[float(re.sub('[!@#$]', '', _)) for _ in bp_1]
        bp_2 = [float(re.sub('[!@#$]', '', _)) for _ in bp_2]
        bp_3 = [float(re.sub('[!@#$]', '', _)) for _ in bp_3]
        bp_4 = [float(re.sub('[!@#$]', '', _)) for _ in bp_4]
        bp_5 = [float(re.sub('[!@#$]', '', _)) for _ in bp_5]

        #Valores Estimados
        bp2_1 = data_3[0][:]
        bp2_2 = data_3[1][:]
        bp2_3 = data_3[2][:]
        bp2_4 = data_3[3][:]
        bp2_5 = data_3[4][:]

        bp2_1 = [float(re.sub('[!@#$]', '', _)) for _ in bp2_1]
        bp2_2 = [float(re.sub('[!@#$]', '', _)) for _ in bp2_2]
        bp2_3 = [float(re.sub('[!@#$]', '', _)) for _ in bp2_3]
        bp2_4 = [float(re.sub('[!@#$]', '', _)) for _ in bp2_4]
        bp2_5 = [float(re.sub('[!@#$]', '', _)) for _ in bp2_5]

        # Nombre de los parametros
        bp3_1 = data_4[0][:]
        bp3_2 = data_4[1][:]
        bp3_3 = data_4[2][:]
        bp3_4 = data_4[3][:]
        bp3_5 = data_4[4][:]

        bp3_1 = [re.sub('[!@#$]', '', _) for _ in bp3_1]
        bp3_2 = [re.sub('[!@#$]', '', _) for _ in bp3_2]
        bp3_3 = [re.sub('[!@#$]', '', _) for _ in bp3_3]
        bp3_4 = [re.sub('[!@#$]', '', _) for _ in bp3_4]
        bp3_5 = [re.sub('[!@#$]', '', _)for _ in bp3_5]





        dff_1 = {"param": bp3_1,  "p(estim)": bp2_1, "p(ref)":bp_1}
        dff_2={"param": bp3_2,  "p(estim)": bp2_2, "p(ref)":bp_2}
        dff_3={"param": bp3_3,  "p(estim)": bp2_3, "p(ref)":bp_3}
        dff_4={"param": bp3_4,  "p(estim)": bp2_4, "p(ref)":bp_4}
        dff_5={"param": bp3_5,  "p(estim)": bp2_5, "p(ref)":bp_5}

        fig = go.Figure()


        fig.add_trace(go.Scatter(x=dff_1['param'], y=dff_1["p(ref)"],name="Valor Referencial",marker={'size': 30,'opacity':0.5} ))


        fig.add_trace( go.Scatter(x=dff_1['param'], y=dff_1["p(estim)"]
                                  , mode="markers",name=const.algoritmos[0],marker={'size': 30,'opacity':0.5},))

        fig.add_trace(go.Scatter(x=dff_2['param'], y=dff_2["p(estim)"]
                                 , mode="markers", name=const.algoritmos[1],marker={'size': 30,'opacity':0.5}
                                 ))

        fig.add_trace(go.Scatter(x=dff_3['param'], y=dff_3["p(estim)"],mode="markers",name=const.algoritmos[2],marker={'size': 30,'opacity':0.5}))

        fig.add_trace(go.Scatter(x=dff_4['param'], y=dff_4["p(estim)"] , mode="markers",name=const.algoritmos[3],marker={'size': 30,'opacity':0.5}  ))

        fig.add_trace(go.Scatter(x=dff_5['param'], y=dff_5["p(estim)"],mode="markers",name=const.algoritmos[4],marker={'size': 30,'opacity':0.5} ))

    return fig




def get_y_data_1():
    data = model_data[1]
    y_mod = data['y1']
    return y_mod


def ymodel_enzi(param):
    df = model_data[1]
    df_init = const.initial_conditions_ob[1]
    x0 = df_init['z']
    t = df['t']
    ymodelo = ODE_sol.odeint(modelo_enzi, x0, t, args=tuple(param))
    return ymodelo[:, 0]


def fitness_goal(param):

    ymeas = get_y_data_1()
    residual = np.sum((ymodel_enzi(param) - ymeas) ** 2)
    print("residual Enzi", residual)
    return residual


def fitness_referencial(param,case):
    residual=[]
    if case is None:
        residual=0
    if case == '0':
        residual=0
    if case =='1':
        residual=fitness_goal(param)
    if case =='2':
        residual=fitness_goal_1(param)
    if case =='3':
        residual=fitness_goal_2(param)
    if case =='4':
        residual=fitness_goal_3(param)
    if case =='5':
        residual=fitness_goal_4(param)
    if case =='6':
        residual=fitness_goal_5(param)


    return residual

def objective_function_enzi(gen: float, pop: float, c1: float, c2: float, w: float):
    objetivo = f_pso.init_algorithm_pso(ymodel_enzi, get_y_data_1, [0.1, 0.1, 0.1, 0.1], [1, 3, 1, 0.3], gen, pop, c1,
                                        c2, w, 1, '1')
    # objetivo = ODE_sol.PSO_test(fitness_goal, 20, 100, [0.1, 0.1, 0.1, 0.1],[1, 3, 1, 0.3], 0.5, 0.3, 0.9)
    return objetivo


def get_y_data_2():
    data = model_data[2]
    y_mod = data['V']
    y_mod_1 = data['T']
    return y_mod, y_mod_1


def ymodel_hepatitis(param):
    df = model_data[2]
    df_init = const.initial_conditions_ob[2]
    x0 = df_init['z']
    t = df['tiempo']
    ymodelo = ODE_sol.odeint(modelo_hepatitis, x0, t, args=tuple(param))
    return ymodelo


def fitness_goal_1(param):
    ymeas, ymeas_1 = get_y_data_2()
    YMODEL = ymodel_hepatitis(param)
    u = 1e9
    # residual=(1/2)*np.array(np.sum((ymodel_hepatitis(param)[:,2]-ymeas)**2),np.sum((ymodel_hepatitis(param)[:,3]-ymeas_1)**2))
    residual_1 = (1 / 2) *np.log10((np.sum(((YMODEL[:, 2] - ymeas) ** 2)) + np.sum((YMODEL[:, 3] - ymeas_1) ** 2)))
    print("hepatitis residual ", residual_1)
    return residual_1


def objective_function_hepa(gen: float, pop: float, c1: float, c2: float, w: float):
    objetivo = f_pso.init_algorithm_pso(ymodel_hepatitis, get_y_data_2, [0.00001, 1.1e-5, 5, 0.1, 1e-2, 4, 8e-5, 900],
                                        [4e-4, 1.9e-5, 14, 0.4, 2.5e-2, 6, 10e-5, 1100], gen, pop, c1, c2, w,
                                        1, '2')

    return objetivo


def get_y_data_3():
    data = model_data[3]
    y_mod = data['x_data']
    return y_mod


def ymodel_benchmark(param):
    df = model_data[3]
    df_init = const.initial_conditions_ob[3]
    x0 = df_init['z']
    t = df['t_data']
    ymodelo = ODE_sol.odeint(modelo_benchmark, x0, t, args=tuple(param))
    return ymodelo[:, 0]


def fitness_goal_2(param):
    ymeas = get_y_data_3()
    residual = np.sum((ymodel_benchmark(param) - ymeas) ** 2)
    return residual


def objective_function_bench(gen: float, pop: float, c1: float, c2: float, w: float):
    # objetivo=ODE_sol.PSO_test(fitness_goal_2,100,100,[-50,-9999,-9999,100],[-20,-1000,-1000,200],1.49445, 1.49445, 0.9)
    objetivo = f_pso.init_algorithm_pso(ymodel_benchmark, get_y_data_3, [-50, -9999, -9999, 100],
                                        [-20, -1000, -1000, 200], gen, pop, c1, c2, w, 1, '1')
    return objetivo


def get_y_data_4():
    data = model_data[4]
    y_mod = data['competidor x']
    y_mod_2 = data['competidor y']
    y_mod_3 = data['competidor z']
    return y_mod, y_mod_2, y_mod_3


def ymodel_lotka(param):
    df = model_data[4]
    df_init = const.initial_conditions_ob[4]
    x0 = df_init['z']
    t = df['t']
    ymodelo = ODE_sol.odeint(modelo_lotka, x0, t, args=tuple(param))
    return ymodelo


def fitness_goal_3(param):
    ymeas1, ymeas2, ymeas3 = get_y_data_4()
    residual = np.sum((ymodel_lotka(param)[:, 0] - ymeas1) ** 2) / len(ymeas1)
    residual2 = np.sum((ymodel_lotka(param)[:, 1] - ymeas2) ** 2) / len(ymeas2)
    residual3 = np.sum((ymodel_lotka(param)[:, 2] - ymeas3) ** 2) / len(ymeas3)
    TOT_res = (residual + residual2 + residual3) / 3
    print("lotka residual", TOT_res)
    return TOT_res


def objective_function_lotka(gen: float, pop: float, c1: float, c2: float, w: float):
    objetivo = f_pso.init_algorithm_pso(ymodel_lotka, get_y_data_4,
                                        [0.1, -0.7, -0.3, -0.7, 0.5, -0.04, -2, -0.7, 0.1, -0.09, -0.30, -0.9],
                                        [0.5, -0.3, -0.1, -0.3, 0.9, -0.01, -1, -0.4, 0.3, -0.03, -0.10, -0.4], gen,
                                        pop, c1, c2, w, 1, '3')
    return objetivo


def get_y_data_5():
    data = model_data[5]
    y_mod = data['lv']
    return y_mod


def ymodel_hiv(param):
    df = model_data[5]
    df_init = const.initial_conditions_ob[5]
    x0 = df_init['z']
    t = df['time']
    ymodelo = ODE_sol.odeint(modelo_hiv, x0, t, args=tuple(param))
    return ymodelo


def fitness_goal_4(param):
    ymeas = get_y_data_5()
    log_ymodel = np.log10(ymodel_hiv(param)[:, 2])
    #(1/len(self.y_data()))*np.sum((np.log10(self.y_modelo(thetha)[:,2]) - self.y_data()) ** 2)
    residual = 1/len(ymeas)*np.sum((log_ymodel - ymeas) ** 2)
    print("residual HIV", residual)
    return residual


def objective_function_HIV(gen: float, pop: float, c1: float, c2: float, w: float):
    objetivo = f_pso.init_algorithm_pso(ymodel_hiv, get_y_data_5, [1e1, 0, 1e-7, 0, 1, 10],
                                        [1e6, 1, 10e-7, 1, 50, 10000], gen, pop, c1, c2, w, 1, '4')
    return objetivo


def get_y_data_6():
    data = model_data[6]
    y_mod = data['exp_1']
    y_mod_1 = prom(data['exp_1'],data['exp_2'])
    return y_mod, y_mod_1


def ymodel_kinetic(param):
    df = model_data[6]
    df_init = const.initial_conditions_ob[6]
    x0 = df_init['z']
    t = df['t']
    ymodelo = ODE_sol.odeint(model_kinetic_chemistry, x0, t, args=tuple(param))
    return ymodelo


def fitness_goal_5(param):
    ymeas_1, ymeas = get_y_data_6()

    residual_1 = np.sum(np.power((ymodel_kinetic(param)[:, 1] - ymeas),2))
    print("residual Kinetic", residual_1)
    return residual_1


def objetive_function_kinetic(gen: float, pop: float, c1: float, c2: float, w: float):
    objetivo = f_pso.init_algorithm_pso(ymodel_kinetic, get_y_data_6, [0, 0], [1, 1], gen, pop,
                                        c1, c2, w, 1, '5')
    return objetivo


# algoritmos a comparar con PSO

def objetivo_compare_F(case, method, gen,

                       pop: int,
                       f, cr,
                       limit):
    df_init_param = const.inital_guesses_paramters[int(case)]
    g = np.zeros(10)
    if method == const.algoritmos[1]:
        if case == '1':
            g = f_pso.init_nspso(ymodel_enzi, get_y_data_1, [0.1, 0.1, 0.1, 0.1], [1, 3, 1, 0.3], gen, pop,1,'1')
        if case == '2':
            g = f_pso.init_nspso(ymodel_hepatitis, get_y_data_2, [3e-4, 1.1e-5, 1, 0.1, 1e-2, 4, 8e-5, 900],
                                  [4e-4, 1.9e-5, 14, 0.4, 2.5e-2, 6, 10e-5, 1100], gen, pop,1,'2')
        if case == '3':
            g = f_pso.init_nspso(ymodel_benchmark, get_y_data_3, [-50, -9999, -9999, 100],
                                  [-20, -1000, -1000, 200], gen, pop,1,'1')
        if case == '4':
            g = f_pso.init_nspso(ymodel_lotka, get_y_data_4,
                                  [0.1, -0.7, -0.3, -0.7, 0.5, -0.04, -2, -0.7, 0.1, -0.09, -0.30, -0.9],
                                  [0.5, -0.3, -0.1, -0.3, 0.9, -0.01, -1, -0.4, 0.3, -0.03, -0.10, -0.4], gen, pop,1,'3')
        if case == '5':
            #[1e1, 0, 1e-7, 0, 1, 10], [1e6, 1, 10e-7, 1, 50, 10000]
            g =f_pso.init_nspso(ymodel_hiv, get_y_data_5, [1e1, 0, 1e-7, 0, 1, 10],
                                  [1e6, 1, 10e-7, 1, 50, 10000], gen, pop,1,'4')
        if case == '6':
            g =f_pso.init_nspso(ymodel_kinetic, get_y_data_6, [-10, -10], [10, 10], gen, pop,1,'5')

    if method == const.algoritmos[2]:
        if case == '1':
            g = f_pso.init_bee_colony(ymodel_enzi, get_y_data_1, [0.1, 0.1, 0.1, 0.1], [1, 3, 1, 0.3], gen, pop,limit,1,'1')
        if case == '2':
            g = f_pso.init_bee_colony(ymodel_hepatitis, get_y_data_2, [0, 0, 1, 0.1, 1e-2, 4, 8e-5, 900],
                                  [1, 1, 14, 0.4, 2.5e-2, 6, 10e-5, 1100],gen, pop,limit,1,'2')
        if case == '3':
            g = f_pso.init_bee_colony(ymodel_benchmark, get_y_data_3, [-50, -9999, -9999, 100],
                                         [-20, -1000, -1000, 200], gen, pop,limit,1,'1')
        if case == '4':
            g = f_pso.init_bee_colony(ymodel_lotka, get_y_data_4,
                                  [0.1, -0.7, -0.3, -0.7, 0.5, -0.04, -2, -0.7, 0.1, -0.09, -0.30, -0.9],
                                  [0.5, -0.3, -0.1, -0.3, 0.9, -0.01, -1, -0.4, 0.3, -0.03, -0.10, -0.4], gen, pop,limit,1,'3')
        if case == '5':
            # [1e1, 0, 1e-7, 0, 1, 10], [1e6, 1, 10e-7, 1, 50, 10000]
            g = f_pso.init_bee_colony(ymodel_hiv, get_y_data_5, [1e1, 0, 1e-7, 0, 1, 10],
                                  [1e6, 1, 10e-7, 1, 50, 10000], gen, pop,limit,1,'4')
        if case == '6':
            g = f_pso.init_bee_colony(ymodel_kinetic, get_y_data_6, [-10, -10], [10, 10], gen, pop,limit,1,'5')

    if method == const.algoritmos[3]:
        if case == '1':
            g = f_pso.init_diif_e(ymodel_enzi, get_y_data_1, [0.1, 0.1, 0.1, 0.1], [1, 3, 1, 0.3], gen, pop, f, cr, '1',100)
        if case == '2':
            g = f_pso.init_diif_e(ymodel_hepatitis, get_y_data_2, [3e-4, 1.1e-5, 12, 0.1, 1e-2, 4, 8e-5, 900],
                                  [4e-4, 1.9e-5, 14, 0.4, 2.5e-2, 6, 10e-5, 1100], gen, pop, f, cr, '2',100)
        if case == '3':
            g = f_pso.init_diif_e(ymodel_benchmark, get_y_data_3, [-50, -9999, -9999, 100],
                                  [-20, -1000, -1000, 200], gen, pop, f, cr, '1',100)
        if case == '4':
            g = f_pso.init_diif_e(ymodel_lotka, get_y_data_4,
                                  [0.1, -0.7, -0.3, -0.7, 0.5, -0.04, -2, -0.7, 0.1, -0.09, -0.30, -0.9],
                                  [0.5, -0.3, -0.1, -0.3, 0.9, -0.01, -1, -0.4, 0.3, -0.03, -0.10, -0.4], gen, pop, f,
                                  cr, '3',100)
        if case == '5':
            # [1e1, 0, 1e-7, 0, 1, 10], [1e6, 1, 10e-7, 1, 50, 10000]
            g = f_pso.init_diif_e(ymodel_hiv, get_y_data_5, [1e1, 0, 1e-7, 0, 1, 10],
                                  [1e6, 1, 10e-7, 1, 50, 10000], gen, pop, f, cr, '4',100)
        if case == '6':
            g = f_pso.init_diif_e(ymodel_kinetic, get_y_data_6, [-10, -10], [10, 10], gen, pop, f, cr,
                                  '5',100)

    return g


def simple_genetic(case, gen, pop, m, cr):
    # [1e1, 0, 1e-7, 0, 1, 10], [1e6, 1, 10e-7, 1, 50, 10000]
    if case == '1':
        g = f_pso.init_genetic_algorithm(ymodel_enzi, get_y_data_1, [0.1, 0.1, 0.1, 0.1], [1, 3, 1, 0.3], gen, pop, m,
                                         cr, '1')
    if case == '2':
        g = f_pso.init_genetic_algorithm(ymodel_hepatitis, get_y_data_2, [0, 1.1e-5, 12, 0.1, 1e-2, 4, 8e-5, 900],
                                         [1, 1.9e-5, 14, 0.4, 2.5e-2, 6, 10e-5, 1100], gen, pop, m, cr, '2')
    if case == '3':
        g = f_pso.init_genetic_algorithm(ymodel_benchmark, get_y_data_3, [-50, -9999, -9999, 100],
                                         [-20, -1000, -1000, 200], gen, pop, m, cr, '1')
    if case == '4':
        g = f_pso.init_genetic_algorithm(ymodel_lotka, get_y_data_4,
                                         [0.1, -0.7, -0.3, -0.7, 0.5, -0.04, -2, -0.7, 0.1, -0.09, -0.30, -0.9],
                                         [0.5, -0.3, -0.1, -0.3, 0.9, -0.01, -1, -0.4, 0.3, -0.03, -0.10, -0.4], gen,
                                         pop, m, cr, '3')
    if case == '5':
        g = f_pso.init_genetic_algorithm(ymodel_hiv, get_y_data_5, [1e1, 0, 1e-7, 0, 1, 10],
                                         [1e6, 1, 10e-7, 1, 50, 10000], gen, pop, m, cr, '4')
    if case == '6':
        g = f_pso.init_genetic_algorithm(ymodel_kinetic, get_y_data_6, [1e-5, 6e-6], [1e-1, 6e-1], gen, pop, m, cr, '5')

    return g


# tablas de descripcion de modelo y parametros
def MODELO(case):
    return dash_table.DataTable(
        id="data_model_2",
        columns=[{"name": i, "id": i} for i in model_analytic],
        data=[],
        style_cell={'fontSize': '10px'},
        style_header={'background': '#008cf9', 'color': '#fff', 'fontSize': '10px', 'textAlign': 'center'},
        style_table={'height': 'auto', 'width': 'auto', 'overflowX': 'auto', 'overflowY': 'auto'},
        style_data={'whiteSpace': 'normal', 'height': 'auto','fontSize': '10px'},
        virtualization=False
    )


def parametros(case):
    return dash_table.DataTable(
        id="parameter_model",
        columns=[{"name": i, "id": i} for i in parametros_titulos],
        data=[],
        style_cell={'fontSize': '10px','padding': '0px' ,'borderRadius': '14px !important'},
        style_header={'background': '#008cf9', 'color': '#fff', 'fontSize': '10px', 'textAlign': 'center'},
        style_table={'height': 'auto', 'width': 'auto', 'overflowX': 'auto', 'overflowY': 'auto'},
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
        virtualization=False
    )

def parametros_saved():
    return dash_table.DataTable(
        id="parameter_saved",
        columns=[{"name": i, "id": i} for i in parametros_titulos],
        data=[],
        style_cell={'fontSize': '12px', 'borderRadius': '11px !important'},
        style_header={'background': '#008cf9', 'color': '#fff', 'fontSize': '11px', 'textAlign': 'center'},
        style_table={'height': 'auto', 'width': 'auto', 'overflowX': 'auto', 'overflowY': 'auto'},
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
        virtualization=False
    )


def test():
    return html.Div(children="",id="parameter_saved_1")



