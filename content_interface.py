from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import constantes as const
import nav_bar_ODE as nav
import collect_data as colldat
from dash import dash_table

# Author: Paul Benavides
# Derechos Reservados

CENTER = {
    "text-align": "center",
    "font-weight": "800"

}

data = []


def model_data(case, graph):
    casos = colldat.CASOS(case)
    graficos = colldat.GRAFICOS(graph)
    return dbc.Col([
        dbc.Row(
            [dbc.Card([
                dbc.Container([
                    dbc.Row([
                        dbc.Col([html.H1(const.titulo_dict["t0"]),
                                 html.H4(id="tilt_model", style=CENTER),
                                 dbc.Row([
                                     dbc.Col(html.Div([html.H6(const.subtitulo_dict["t0"]), casos]), sm=3),
                                     dbc.Col(html.Div([html.H6(const.subtitulo_dict["t11"]), graficos]), sm=9)
                                 ])]),
                    ])
                ])
            ])
            ]
        )
    ], style=nav.CONTENT_STYLE)


def model_description(case, graph):
    graficos_2 = colldat.GRAFICOS(graph)
    return dbc.Col([
        dbc.Row(
            [
                dbc.Card([
                    dbc.Container([
                        dbc.Row([
                            dbc.Col([html.H1(const.titulo_dict["t1"]),
                                     dbc.Row([
                                         dbc.Col(html.Div(
                                             [

                                                 colldat.MODELO(case)

                                             ],

                                         ), sm=12),

                                         dbc.Col(
                                             html.Div([
                                                 dbc.NavLink(dbc.Button("Ejecutar Integrador", className="button-link",
                                                                        id="odeint", n_clicks=0))
                                             ]), sm=12, style={"padding-right": 0, "padding-left": 0})

                                         ,
                                         dbc.Col(html.Div([

                                             html.Div(graficos_2),

                                         ]

                                         ), sm=12, style={"padding-top": 50})

                                     ])]),

                        ])

                    ], fluid=True, class_name="container")

                ])

            ]

        )

    ], style=nav.CONTENT_STYLE)


def algoritmo_proceso(algo,case,graph,graph_1,graph_2):
    #funcion objetivo
    graficos_3 = colldat.GRAFICOS(graph)
    #espacio de busqueda PSO
    graficos_4=colldat.GRAFICOS(graph_1)
    #fitting data
    graficos_5=colldat.GRAFICOS(graph_2)
    #tabla_deparametros
    parametros=colldat.parametros(case)

    return dbc.Col([
        dbc.Row(
            [
                dbc.Card([
                    dbc.Container([
                        dbc.Row([
                            dbc.Col([html.H1(const.titulo_dict["t2"]),
                                     dbc.Row([
                                        dbc.Col(html.Div([

                                             dbc.Col(html.Div([dbc.NavLink(dbc.Button("Ejecutar Algoritmo",
                                                                                     className="button-link",
                                                                                     id="algoexe", n_clicks=0))]),
                                                    sm=12, style={"padding-right": 0, "padding-left": 0}),
                                             html.Div(graficos_4)


                                         ]

                                         ), sm=12, style={"padding-top": 50}),





                                         dcc.Loading(
                                             id="load",
                                             children=[
                                                 dbc.Col(html.Div([

                                                 dbc.Col(html.Div([

                                                 html.Div(colldat.parametros(case))]), sm=12,style={"padding-top": 50}),
                                                 html.Div(graficos_5)]), sm=12, style={"padding-top": 50}),



                                                       ],

                                             type="circle",
                                         ),


                                     ])]),])

                    ], fluid=True, class_name="container")

                ])

            ]

        )

    ], style=nav.CONTENT_STYLE)
