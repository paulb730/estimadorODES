import  dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import numpy as np
import constantes as const
# Author: Paul Benavides
# Derechos Reservados

SIDEBAR_STYLE = {
    "position":"fixed",
    "backgroundColor": "#222831",
    "padding":"1rem 1rem",
    "top":"0",
    "left":"0",
    "bottom":"0",
    "color":"#ffff",
    "fontFamily": "Times New Roman, Times, serif"
}
COLORS_STYLE={
    "backgroundColor": "#222831"
}
FONT_COLOR={
"color":"#ffff",

}
NAV_LINKS_STYLE={
    "color":"#ffff",
    "fontWeight":"500",
    "font-size":"18px",
}
FONT_STYLE={



}
# padding for the page content
CONTENT_STYLE = {

    "padding": "1rem 1rem",
    "fontFamily": "Times New Roman, Times, serif"
}
NAV_BAR_STYLE = {

    "top": "0",
    "left": "0",
    "bottom": "0",
    "backgroundColor": "#393e46 ",
    "fontFamily": "Times New Roman, Times, serif"


}


names=[["Casos de Estudio",0],["Modelo Enzimático",1],["Dinámica Hepatitis",2],["Dinámica Hepatitis 10%",3], ["Dinámica Hepatitis 15%",4],["Dinámica HIV",5],["Modelo de Cinética Química",6]]

names= np.array(names)
def fill_dropdown(namop):
    opciones=[]
    for i in namop:
        opciones.append(dbc.DropdownMenuItem(i))
    return opciones
def dropdown(id):
    return dcc.Dropdown(
        id=str(id),
        options=[{'label' :name,'value':val} for name,val in names ],
        value=0
    )
def nav_bar():
    return dbc.Container(
            [
                dbc.Col(html.H2("Estimador de  Parámetros de  EDO's "), id="tit",class_name="order-12 p-2",style=FONT_COLOR,align="center")



            ],

        )
def side_bar(id):
    return  html.Div(
        [
            html.H2("Funciones", className="display-6"),
            html.Hr(),
            html.P("Algoritmos"),
            dbc.Nav(
                [

                    dbc.NavLink( str(n) , href="/"+str(n) , className="page-link") for n in const.algoritmos




                ],
                vertical=True,
                pills=True,


            ),

            html.Hr(),
            html.P("Aplicaciones"),

            dbc.Nav(

                [
                    dbc.NavLink(dbc.Col(dropdown(id), className="drop-dash "  )),
                    html.Br(),
                    dbc.NavLink( dbc.Button("Benchmark", className="button-link"))



                ],
                vertical=True


            )
        ],
        style=SIDEBAR_STYLE,
    )


