import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
import numpy as np
import constantes as const

# Author: Paul Benavides
# Derechos Reservados

SIDEBAR_STYLE_1 = {
    "position":"sticky",
    "backgroundColor": "#222831",
    "padding": "1rem 1rem",
    "top": "0",
    "left": "0",
    "right":"0",
    "bottom": "0",

    "color": "#ffff",
    "fontFamily": "Times New Roman, Times, serif"
}
COLORS_STYLE = {
    "backgroundColor": "#222831"
}
FONT_COLOR = {
    "color": "#ffff",

}
NAV_LINKS_STYLE = {
    "color": "#ffff",
    "fontWeight": "500",
    "font-size": "18px",
}
FONT_STYLE = {

}
# padding for the page content
CONTENT_STYLE = {

    "padding": "1rem 1rem",
    "fontFamily": "Times New Roman, Times, serif"
}

CARD_STYLE = {


}
NAV_BAR_STYLE = {

    "top": "0",
    "left": "0",
    "bottom": "0",
    "backgroundColor": "#393e46 ",
    "fontFamily": "Times New Roman, Times, serif"

}

names = [["Casos de Estudio", 0], ["Modelo Enzimático", 1], ["Dinámica Hepatitis C (5% ruido) ", 2], ["Benchmark", 3],
         ["Lotka Volterra", 4], ["Dinámica HIV", 5], ["Modelo de Cinética Química", 6]]

names = np.array(names)


def fill_dropdown(namop):
    opciones = []
    for i in namop:
        opciones.append(dbc.DropdownMenuItem(i))
    return opciones


def dropdown(id):
    return dcc.Dropdown(
        id=str(id),
        options=[{'label': name, 'value': val} for name, val in names],
        value=0
    )


def nav_bar(id):
    return dbc.Card(

        dbc.CardHeader( [

            html.Div(id="garbage-output-0"),
            html.Div(id="garbage-output-1"),
            html.Div(id="garbage-output-2"),
            html.Div(id="garbage-output-3"),
            dbc.Row([dbc.Col([html.H2("Estimador de  Parámetros en EDOs", id='main_tit')],
                             class_name="pad_tit col-sm-6"),
                     dbc.Col([dbc.NavLink(dbc.Col(dropdown(id), className="drop-dash col-sm-6"))],
                             )],
                    id="tit",style=FONT_COLOR, align="center"),
        ])
       ,class_name="mediatit"

    )


def side_bar():
    return dbc.Nav(
        [


            dbc.Nav(
                [
                    html.H4("Algoritmos"),
                    dbc.RadioItems(options=[{'label': str(j), 'value': str(j)} for j in const.algoritmos],
                                   id='algo_list'),
                    dbc.Alert(
                        [
                            "Elije un caso de estudio",
                            html.A("link ejemplo", href="#", className="alert-link"),
                        ],
                        id="alert_algo",
                        color="danger",
                        is_open=False,
                        dismissable=True,
                    )


                ],

                className="nav_algo nav_radio",
                vertical="sm",
                id="navradioid"

            ),
            html.Div([ dbc.Col(dbc.Button("Data", class_name="button-link_1 mt-2 col-sm-6", id="butmin_0f"),
                    ),
            dbc.Col(dbc.Button("Descripcion", class_name="button-link_1 mt-2 col-sm-6", id="butmin_1f"),
                    ),
            dbc.Col(dbc.Button("Optimizacion", class_name="button-link_1 mt-2 col-sm-6 ", id="butmin_2f"),
                    ),

            dbc.Col(dbc.Button("Estadisticas", class_name="button-link_1 mt-2 col-sm-6", id="butmin_3f"),
                    )],id="navbut")

            ,
        dbc.Nav(
             [

                 html.Hr(),
                 dbc.NavLink(
                     dbc.Button("Visualización PSO", className="button-link ", id='but_viewpso', color="primary",
                                n_clicks=0, )),

             ],
             vertical="sm"
         ),
         html.Hr(),


         dbc.Nav(

             [
                 html.Hr(),
                 dcc.Textarea(
                     id='textarea_console',
                     value="",
                     className="text_area_1"
                 ),

             ],
             vertical="sm",
             pills=True,
             className="nav_algo"
         ),

         ],
        class_name="SIDEBAR_STYLE",
        vertical="sm",
        pills=True,

    )
