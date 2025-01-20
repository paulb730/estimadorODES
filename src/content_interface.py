from dash import html, dcc
import dash_bootstrap_components as dbc
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src import constantes as const
from src import nav_bar_ODE as nav
from src import collect_data as colldat

# Author: Paul Benavides
# Derechos Reservados

CENTER = {
    "text-align": "center",
    "font-weight": "100"

}


def slider_data(items):
    return dbc.Carousel(
        items=items,
        controls=False,
        indicators=False,
        interval=2000,
        ride="carousel",
    )


def collapse_component(component, id,on_off):
    return dbc.Collapse(
        dbc.Container([component],fluid=True, class_name="container"),
        id=id,
        is_open=on_off, )




def model_data(case, graph):
    casos = colldat.CASOS(case)
    graficos = colldat.GRAFICOS(graph)
    return dbc.Col([
        dbc.Row(
            [dbc.Card([
                dbc.Container([
                    dbc.Row([
                        dbc.Col(html.H6(const.titulo_dict["t0"]), class_name="col-sm-9 mt-2"),
                        dbc.Col(dbc.Button("Min", class_name="button-link_1 mt-2", id="butmin_0"), class_name="col-sm-3"),
                        collapse_component(dbc.Col([
                                 dbc.Row([
                                     dbc.Col(html.Div(casos), sm=3), dbc.Col(html.Div(graficos), sm=9)
                                 ])]),"datacontainer",True),

                    ])
                ], fluid=True, class_name="container")
            ])
            ]
        )
    ], style=nav.CONTENT_STYLE,id="colu_dat")


def model_description(case, graph):
    graficos_2 = colldat.GRAFICOS(graph)
    table_inputs = colldat.PARAM_INPUT(0, 0)
    return dbc.Col([
        dbc.Row(
            [
                dbc.Card([
                    dbc.Container([
                        dbc.Row([
                            dbc.Col(html.H6(const.titulo_dict["t1"]), class_name="col-sm-9 mt-2"),
                            dbc.Col(dbc.Button("Min",class_name="button-link_1 mt-2",id="butmin_1"), class_name="col-sm-3"),

                            collapse_component(dbc.Col([
                                     dbc.Row([
                                         dbc.Col(html.Div(
                                             [

                                                 colldat.MODELO(case),
                                                 html.Br(),

                                             ],

                                         ), sm=12),

                                         dbc.Col(html.H6("Edición de parámetros")),
                                         html.Hr(),
                                         dbc.Col(dbc.Row(html.Div(children=table_inputs, id='param_div', style={})),
                                                 sm=12),

                                         dbc.Col(

                                             html.Div([
                                                 html.Br(),
                                                 dbc.NavLink(dbc.Button("Integrar Modelo", className="button-link",
                                                                        id="odeint", n_clicks=0,disabled=True))]), sm=12,
                                             style={"padding-right": 0, "padding-left": 0}),
                                         dbc.Col(html.Div([
                                             dbc.Spinner(graficos_2, show_initially=True,fullscreen=True)

                                         ]

                                         ), sm=12, style={"padding-top": 50})

                                     ])]),"description_container",True),



                        ])

                    ], fluid=True, class_name="container")

                ])

            ]

        )

    ], style=nav.CONTENT_STYLE,id="col_desc")


def ALGO_edit_PSO():
    # Proceso PSO editable
    return collapse_component(dbc.Col([
        dbc.Row([
            dbc.Col([
                dbc.Label("Coeficiente de Inercia", html_for="slider_w"),
                dcc.Slider(0, 1, 0.0001,
                           value=0.7804,
                           marks={0: {'label': 'W:0'},
                                  0.7804: {'label': 'W:0.7804'},
                                  1: {'label': 'W:1'}},
                           id=const.id_edit_form[0], tooltip={"placement": "bottom",
                                                              "always_visible": True})

            ], sm=4),
            dbc.Col([
                dbc.Label("Coeficiente Social", html_for="slider_c1"),
                dcc.Slider(0, 4, 0.001,
                           value=2.05,
                           marks={0: {'label': 'C1:0'},
                                  2.05: {'label': 'C1:2.05'},
                                  4: {'label': 'C1:4'}},
                           id=const.id_edit_form[1], tooltip={"placement": "bottom",
                                                              "always_visible": True})

            ], sm=4),
            dbc.Col([
                dbc.Label("Coeficiente Cognitivo", html_for="slider_c2"),
                dcc.Slider(0, 4, 0.001,
                           value=2.05,
                           marks={0: {'label': 'C2:0'},
                                  2.05: {'label': 'C2:2.05'},
                                  4: {'label': 'C2:4'}},
                           id=const.id_edit_form[2],
                           tooltip={"placement": "bottom",
                                    "always_visible": True})

            ], sm=4),
        ]),
        dbc.Row([dbc.Col([
            dbc.Label("Generación", html_for="input_GEN"),
            dbc.Input(id=const.id_edit_form[3], type="number", value=100, step=1, max=1000,
                      placeholder="GEN")], sm=6),
            dbc.Col([
                dbc.Label("Población", html_for="input_POP"),
                dbc.Input(id=const.id_edit_form[4], type="number", value=100, step=1,
                          max=500, placeholder="POP")

            ], sm=6)])
    ]), "PSO_edit_container",False)


def ALGO_EDIT_MC():
    # MC
    # gen_NS, popNS, wNS, c1NS, c2NS, v_coeff,
    return collapse_component(
        dbc.Col(
            [
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Generaciones", html_for=const.id_edit_algo_form_0[0]),
                        dcc.Slider(0, 1000, 1,
                                   value=100,
                                   marks={0: {'label': 'GEN:0'},
                                          500: {'label': 'GEN:500'},
                                          1000: {'label': 'GEN:1000'}},
                                   id=const.id_edit_algo_form_0[0], tooltip={"placement": "bottom",
                                                                           "always_visible": True})

                    ], sm=6),

                    dbc.Col([
                        dbc.Label("Población",html_for=const.id_edit_algo_form_0[1]),
                        dcc.Slider(0,500,1,
                                   value=25,
                                   marks={0: {'label': 'POP:0'},
                                          25:{'label':'25'},
                                          100: {'label': '100'},
                                          500: {'label': '500'}},
                                   id=const.id_edit_algo_form_0[1],tooltip={"placement":"bottom",
                                                                            "always_visible":True}


                        )



                    ],sm=6)

                ]),



            ]
            #"ALGO_edit_container"
        ), "MC_edit_container",False)


def ALGO_EDIT_BEE():
    # Bee colony
    # gen_be, limit_be,

    return collapse_component(
        dbc.Col([

            dbc.Row([
                dbc.Col([
                    dbc.Label("Generacion", html_for=const.id_edit_algo_form_1[0]),
                    dcc.Slider(0, 1000, 1,
                               value=100,
                               marks={0: {'label': 'GEN:0'},
                                      100: {'label': '100'},
                                      1000: {'label': '1000'}},
                               id=const.id_edit_algo_form_1[0], tooltip={"placement": "bottom",
                                                                         "always_visible": True}
                               )

                ], sm=6),
                dbc.Col([
                    dbc.Label("Población", html_for=const.id_edit_algo_form_1[1]),
                    dcc.Slider(0, 500, 1,
                               value=25,
                               marks={0: {'label': 'POP:0'},
                                      25: {'label': '25'},
                                      500: {'label': '500'}},
                               id=const.id_edit_algo_form_1[1], tooltip={"placement": "bottom",
                                                                         "always_visible": True}
                               )

                ], sm=6)

            ]),

            dbc.Row([

                dbc.Col([

                    dbc.Label("Limit", html_for=const.id_edit_algo_form_1[2]),
                    dcc.Slider(0, 500, 1,
                               value=1,
                               marks={0: {'label': 'Limit:0'},
                                      25: {'label': '25'},
                                      100: {'label': '100'},
                                      500: {'label': '500'}},
                               id=const.id_edit_algo_form_1[2], tooltip={"placement": "bottom",
                                                                         "always_visible": True}
                               )

                ], sm=12)

            ]),

        ])
        , "BEE_edit_container",False)

def ALGO_EDIT_1():
    return collapse_component(
        dbc.Col(
            [

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Generación", html_for=const.id_edit_algo_form_2[0]),
                        dcc.Slider(0, 1000, 1,
                                   value=100,
                                   marks={0: {'label': 'GEN:0'},
                                          500: {'label': 'GEN:500'},
                                          1000: {'label': 'GEN:1000'}},
                                   id=const.id_edit_algo_form_2[0], tooltip={"placement": "bottom",
                                                                             "always_visible": True})

                    ], sm=6),

                    dbc.Col([
                        dbc.Label("Población", html_for=const.id_edit_algo_form_2[1]),
                        dcc.Slider(0, 500, 1,
                                   value=100,
                                   marks={0: {'label': 'POP:0'},
                                          250: {'label': 'POP:250'},
                                          500: {'label': 'POP:500'}},
                                   id=const.id_edit_algo_form_2[1], tooltip={"placement": "bottom",
                                                                             "always_visible": True})

                    ], sm=6),
                ]),
                dbc.Row([

                    dbc.Col([
                        dbc.Label("Weighting Factor", html_for=const.id_edit_algo_form_2[2]),
                        dcc.Slider(0, 1, 1e-3,
                                   value=0.5,
                                   marks={0: {'label': '0'},
                                          0.5: {'label': '0.5'},
                                          1: {'label': '1'}},
                                   id=const.id_edit_algo_form_2[2], tooltip={"placement": "bottom",
                                                                             "always_visible": True})

                    ], sm=6),

                    dbc.Col([

                        dbc.Label("Crossover", html_for=const.id_edit_algo_form_2[3]),
                        dcc.Slider(0, 1, 1e-3,
                                   value=0.5,
                                   marks={0: {'label': '0'},
                                          0.5: {'label': '0.5'},
                                          1: {'label': '1'}},
                                   id=const.id_edit_algo_form_2[3], tooltip={"placement": "bottom",
                                                                             "always_visible": True})

                    ], sm=6)

                ])
            ]
        ), "DEALGO_edit_container",False)


def ALGO_EDIT_2():
    return collapse_component(
        dbc.Col(
            [

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Generación", html_for=const.id_edit_algo_form_3[0]),
                        dcc.Slider(0, 1000, 1,
                                   value=100,
                                   marks={0: {'label': 'GEN:0'},
                                          500: {'label': 'GEN:500'},
                                          1000: {'label': 'GEN:1000'}},
                                   id=const.id_edit_algo_form_3[0], tooltip={"placement": "bottom",
                                                                             "always_visible": True})

                    ], sm=6),

                    dbc.Col([
                        dbc.Label("Población", html_for=const.id_edit_algo_form_3[1]),
                        dcc.Slider(0, 500, 1,
                                   value=100,
                                   marks={0: {'label': 'POP:0'},
                                          250: {'label': 'POP:250'},
                                          500: {'label': 'POP:500'}},
                                   id=const.id_edit_algo_form_3[1], tooltip={"placement": "bottom",
                                                                             "always_visible": True})

                    ], sm=6),
                ]),
                dbc.Row([

                    dbc.Col([
                        dbc.Label("Mutación", html_for=const.id_edit_algo_form_3[2]),
                        dcc.Slider(0, 1, 1e-3,
                                   value=0.5,
                                   marks={0: {'label': '0'},
                                          0.5: {'label': '0.5'},
                                          1: {'label': '1'}},
                                   id=const.id_edit_algo_form_3[2], tooltip={"placement": "bottom",
                                                                             "always_visible": True})

                    ], sm=6),

                    dbc.Col([

                        dbc.Label("Crossover", html_for=const.id_edit_algo_form_3[3]),
                        dcc.Slider(0, 1, 1e-3,
                                   value=0.5,
                                   marks={0: {'label': '0'},
                                          0.5: {'label': '0.5'},
                                          1: {'label': '1'}},
                                   id=const.id_edit_algo_form_3[3], tooltip={"placement": "bottom",
                                                                             "always_visible": True})

                    ], sm=6)

                ])
            ]
        ), "GAALGO_edit_container",False)


def edit_table_FORM(component: list, id):
    return dbc.Form(component, id=id)


def algoritmo_proceso(case, graph_2):
    # fitting data
    graficos_5 = colldat.GRAFICOS_2(graph_2)
    # tabla_deparametros
    parametros = colldat.parametros(case)

    return dbc.Col([

        dbc.Row(
            [
                dbc.Card([
                    dbc.Container([
                        dbc.Row([
                            dbc.Col(html.H6(const.titulo_dict["t2"]), class_name="col-sm-9 mt-2"),
                            dbc.Col(dbc.Button("Min", class_name="button-link_1 mt-2", id="butmin_2"),
                                    class_name="col-sm-3"),
                            collapse_component(dbc.Col([
                                     dbc.Row([

                                         # EDITABLES

                                         edit_table_FORM([ALGO_edit_PSO()], "PSO_id_form"),
                                         edit_table_FORM([ALGO_EDIT_MC()], "MC_id_form"),
                                         edit_table_FORM([ALGO_EDIT_BEE()], "bee_id_form"),
                                         edit_table_FORM([ALGO_EDIT_1()], "dealgo_id_form"),
                                         edit_table_FORM([ALGO_EDIT_2()], "gaalgo_id_form"),

                                         # Boton Ejecutar
                                         dbc.Col(html.Div([

                                             dbc.Col([dbc.Button("Ejecutar Algoritmo", className="button-link",
                                                                 id="algoexe", n_clicks=0, disabled=True)], sm=12,
                                                     style={"padding-right": 0, "padding-left": 0}),

                                             # proceso callback

                                         ]), sm=12, style={"padding-top": 50}),

                                         # Tabla de parámetros
                                         dbc.Col(html.Div(parametros), sm=12, style={"padding-top": 50}),

                                         # Grafico
                                         dcc.Loading(
                                             children=[
                                                 graficos_5], id="load",fullscreen=True,debug=True, type='graph'

                                         ),

                                         dcc.Store(id='memory_storage_param_table'),
                                         dcc.Store(id='memory_storage_param_table_1'),
                                         dcc.Store(id='memory_storage_param_table_2'),
                                         dcc.Store(id='memory_storage_param_table_3'),
                                         dcc.Store(id='memory_storage_param_table_4'),

                                         dbc.Col(html.Div(id="test_time"))

                                     ])]),"algocontainer",True)])

                    ], fluid=True, class_name="container")

                ])

            ]

        )

    ], style=nav.CONTENT_STYLE,id="col_opt")


def field_data_estadisticas():
    return dbc.Col([

    ], sm=12)


def estadisticas(table_cond_algo,
                 graph_est_0,
                 graph_est_1,
                 graph_est_2,
                 ):
    table_data = colldat.algo_table_cond(table_cond_algo)
    gra_0 = colldat.GRAFICOS_3(graph_est_0)
    gra_1 = colldat.GRAFICOS_3(graph_est_1)
    gra_2 = colldat.GRAFICOS_3(graph_est_2)

    return dbc.Col([
        dbc.Row(
            [
                dbc.Card([
                    dbc.Container([
                        dbc.Row([

                            dbc.Col([
                                dbc.Row([
                                    # Abrir panel de estadísticas
                                    dbc.NavLink(
                                        dbc.Button("Mostrar Estadísticas", className="button-link ", id="but_estadis",
                                                   color="primary", n_clicks=0, disabled=True)),

                                ])], sm=12),
                        ]),

                        html.Br(),

                        collapse_component(dbc.Row([dbc.Row([
                            dbc.Col([

                                table_data,

                            ], sm=12),

                        ]), dbc.Row([
                            dbc.Col([

                                gra_0,

                            ], sm=12),
                            dbc.Col([

                                gra_1,

                            ], sm=12),
                            dbc.Col([

                                gra_2

                            ], sm=12)

                        ]), ]), const.id_estadisticas,True),

                        html.Br(),

                    ], fluid=True, class_name="container")

                ])

            ]

        )

    ], style=nav.CONTENT_STYLE,id="col_esta")



def VIEW_PSO(graph_view_PSO_1):

    graph_view_PSO_1=colldat.GRAFICOS_2(graph_view_PSO_1)
    return dbc.Col([dbc.Row(
        [dbc.Card([
            dbc.Container([

                dbc.Row([

                    html.H3("Visualización PSO"),
                    html.Br(),
                    dbc.Col([html.Iframe(
                        id=graph_view_PSO_1,
                        srcDoc=None,  # here is where we will put the graph we make
                        style={'border-width': '5', 'width': '100%',
                               'height': '500px'}), ])
                ])
            ])

        ])]


    )],style=nav.CONTENT_STYLE)
