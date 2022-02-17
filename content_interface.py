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
    "text-align":"center",
    "font-weight":"800"

}

data=[]

def model_data(case,graph):
    casos=colldat.CASOS(case)
    graficos=colldat.GRAFICOS(graph)

    return dbc.Col([
        dbc.Row(
            [dbc.Card([
                    dbc.Container([
                        dbc.Row([
                            dbc.Col([html.H1(const.titulo_dict["t0"]),
                                    html.H4(id="tilt_model",style=CENTER)
                                    ,
                                dbc.Row([
                                    dbc.Col(html.Div([html.H6(const.subtitulo_dict["t0"]),
                                                casos]),sm=6),
                                    dbc.Col(html.Div([
                                            html.H6(const.subtitulo_dict["t11"]),
                                            graficos]),sm=6)


                            ])]),


                        ])

                    ])

                ])

            ]

        )

    ],style=nav.CONTENT_STYLE)

def model_description(case,graph):
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
                                                 html.H6(const.subtitulo_dict["t0"]),
                                                 colldat.MODELO(case)

                                             ],

                                         ), sm=12),
                                         dbc.Col(html.Div([
                                             html.H6(const.subtitulo_dict["t1"]),
                                             html.Div(graficos_2),


                                         ]

                                         ), sm=12, style={"padding-top": 50}),
                                         dbc.Col(
                                             html.Div([
                                                 dbc.NavLink(dbc.Button("Ejecutar Integrador", className="button-link",id="odeint",n_clicks=0))
                                             ]),sm=12, style={"padding-right":0,"padding-left":0})

                                     ])]),

                        ])

                    ], fluid=True, class_name="container")

                ])

            ]

        )

    ], style=nav.CONTENT_STYLE)

def algoritmo_proceso():
    return 1

