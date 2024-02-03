# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 23:01:23 2023

@author: ANDRES
"""

import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
import locale

pd.options.display.float_format = '{:.2f}'.format

bal_2018 = pd.read_csv('2018.csv', sep=";", dtype={'SALDO_USD': float})
bal_2019 = pd.read_csv('2019.csv', sep=";", dtype={'SALDO_USD': float})
bal_2020 = pd.read_csv('2020.csv', sep=";", dtype={'SALDO_USD': float})
bal_2021 = pd.read_csv('2021 EEFF MEN.txt', sep=";", dtype={'SALDO_USD': float})
bal_2022 = pd.read_csv('EEFF MEN 2022.txt', sep="\t", dtype={'SALDO_USD': float})
bal_2022.columns = ['FECHA_DE_CORTE', 'SEGMENTO', 'RUC',
                    'RAZON_SOCIAL', 'CUENTA', 'DESCRIPCION_CUENTA', 'SALDO_USD']
bal_2023 = pd.read_csv('EEFF MEN 2023.txt', sep="\t",
                       decimal=",", dtype={'SALDO_USD': float})
bal_2023.columns = ['FECHA_DE_CORTE', 'SEGMENTO', 'RUC',
                    'RAZON_SOCIAL', 'CUENTA', 'DESCRIPCION_CUENTA', 'SALDO_USD']
bal_total = pd.concat([bal_2018, bal_2018, bal_2019,
                      bal_2020, bal_2021, bal_2022, bal_2023], axis=0)

bal_total['FECHA_DE_CORTE'] = pd.to_datetime(bal_total['FECHA_DE_CORTE'])  # .dt.strftime('%d-%m-%Y')
bal_total = bal_total.astype({'SALDO_USD': 'float', 'CUENTA': 'int64', 'RUC': 'int64'})

ind_fin = pd.read_excel('data_total.xlsx') ######### indicadores financieros

agencias = pd.read_excel('puntos_atencion.xlsx')
agencias['fecha_corte'] = pd.to_datetime(agencias['fecha_corte'])

calif = pd.read_excel('calificaciones.xlsx')
calif['FECHA'] = pd.to_datetime(calif['FECHA'])

calif_model = pd.read_excel('CALIFICACION_SEPS_2023.xlsx')


locale.setlocale(locale.LC_TIME, 'es_ES')

ruc = 1790866084001
fecha = '2023-06-30'

## Funcón para mostrar en porcentajes

def convertir_a_porcentaje(valor):
    return f'{valor * 100:.2f}%'

## Funcón para mostrar en dólares

def convertir_a_dolares(valor):
    return '${:,.0f}'.format(valor)

## GRÁFICO EVOLUVIÓN DE LA CALIFICACIÓN

calif_model = calif_model[calif_model['RUC_ID'] == ruc].copy()
calif_model['Fecha'] = pd.to_datetime(calif_model['Fecha'])
calif_model.sort_values(by='Fecha', ascending= False, inplace=True)
calif_model = calif_model.iloc[0:24,:]
calif_model.sort_values(by='Fecha', ascending= True, inplace=True)


fig = go.Figure()
fig.add_trace(go.Scatter(x=calif_model['Fecha'], y=calif_model['PRED_BUENO'],
                         mode='lines+markers+text',
                         text=calif_model['CALIFICACION'],  # Agregar valores como etiquetas
                         textposition=['top center', 'bottom center'] * (len(calif_model) // 2) + ['top center'] * (len(calif_model) % 2),  # Posición de las etiquetas
                         textfont=dict(size=9),
                         line=dict(dash='dot', width=4, color='red'),
                         marker=dict(color='darkblue', size=9, opacity=0.8)))

fig.update_layout(title='Evolución de la calificación',
                  yaxis_title='Score')

fig.show()


## Función para generar Tabla de información básica

def inf_bas(fecha,ruc):
    agencias1 = agencias[agencias['ruc'] == ruc].copy()
    agencias1 = agencias1[agencias1['fecha_corte'] == fecha].copy()
    datos_coac = agencias1['canton'].unique()
    datos_coac = ['Nombre de la Insticución:', 'RUC:', 'Provincia:', 'Ciudad:',
                    'Puntos de atención:', 'Segmento cooperativo','Calificación obtenida por una Calificadora de Riesgos:']
    tab_inf = pd.DataFrame(columns=['Información'], index=datos_coac)
    tab_inf.iloc[0,0] = agencias1.loc[agencias1['ruc'] == ruc, 'razon_social'].values[0]
    tab_inf.iloc[1,0] = ruc
    tab_inf.iloc[2,0] = agencias1.loc[agencias1['tipo_punto'] == 'MATRIZ', 'provincia'].values[0]
    tab_inf.iloc[3,0] = agencias1.loc[agencias1['tipo_punto'] == 'MATRIZ', 'canton'].values[0]
    tab_inf.iloc[4,0] = ', '.join(agencias1['canton'].tolist())
    tab_inf.iloc[5,0] = agencias1.loc[agencias1['ruc'] == ruc, 'segmento'].values[0]
    calif1 = (calif[(calif['FECHA'] == calif['FECHA'].max()) & (calif['RUC'] == ruc)]).reset_index(drop=True)
    tab_inf.iloc[6,0] = '\n'.join(calif1.loc[0,['FECHA', 'FIRMA CALIFICADORA DE RIESGO','CALIFICACIÓN']].apply(str).tolist())
    return (tab_inf)


inf_basic = inf_bas(fecha,ruc)


## Función para generar Tabla de indicadores financieros

def tabla_indfin(ruc, fecha, indicadores, ind_fin):
    ind_fin1 = ind_fin[ind_fin['RUC_ID'] == ruc].copy()
    ind_fin1.sort_values(by='Fecha', ascending= True, inplace=True)

    ind_fin1 = ind_fin1.drop(['SEGMENTO', 'RUC_ID','ID_AND'], axis=1)
    ind_fin1 = ind_fin1.set_index('Fecha')
    ind_fin1 = ind_fin1.T

    # Se crea fechas para los 5 últimos meses
    mes = pd.to_datetime(fecha).month
    fechas = pd.date_range(fecha, periods=5, freq='-1Y')
    fechas = [fecha - pd.offsets.MonthEnd(mes) for fecha in fechas]
    fechas.sort()
    mes_ant = fechas[-1].replace(day=1) - timedelta(days=1)
    mes_dic = pd.to_datetime(fecha) - pd.offsets.YearEnd(1)
    fechas.append(mes_dic)
    fechas.append(mes_ant)

    ind_fin2 = ind_fin1.loc[indicadores, fechas]
    ind_fin2.columns = ind_fin2.columns.strftime("%b %Y")
    ind_fin3 = ind_fin2.applymap(convertir_a_porcentaje)

    return ind_fin3

ind_fin = ind_fin
ruc = 1790866084001
fecha = '2023-06-30'


indicadores = ['(_PATRIMONIO_+_RESULTADOS_)_/_ACTIVOS_INMOVILIZADOS', 
            'ACTIVOS_IMPRODUCTIVOS_NETOS_/_TOTAL_ACTIVOS',
            'MOROSIDAD_DE_LA_CARTERA_TOTAL',
            'COBERTURA_DE_LA_CARTERA_PROBLEMÁTICA',
            'GASTOS_DE_OPERACION_ESTIMADOS_/_TOTAL_ACTIVO_PROMEDIO',
            'GASTOS_DE_OPERACION_/_MARGEN_FINANCIERO',
            'RESULTADOS_DEL_EJERCICIO_/_PATRIMONIO_PROMEDIO',
            'RESULTADOS_DEL_EJERCICIO_/_ACTIVO_PROMEDIO',
            'FONDOS_DISPONIBLES_/_TOTAL_DEPOSITOS_A_CORTO_PLAZO']

indicadores = tabla_indfin(ruc, fecha, indicadores, 
                           ind_fin)

# Función para generar tabla de cuentas de balance
def cuenta_balance(ruc, fecha, cuentas, periodos, bal_total):

    bal_total1 = bal_total[bal_total['RUC'] == ruc].copy()
    bal_total1.sort_values(by='FECHA_DE_CORTE', ascending=True, inplace=True)
    bal_total2 = bal_total1.pivot_table(
        values='SALDO_USD', index='FECHA_DE_CORTE', columns='CUENTA')

    mes = pd.to_datetime(fecha).month

    if periodos == 'anuales':#=============> modificar
        fechas = pd.date_range(fecha, periods=5, freq='-1Y')
        fechas = [fecha - pd.offsets.MonthEnd(mes) for fecha in fechas]
    elif periodos == 'mensuales':
        fechas = pd.date_range(fecha, periods=5, freq='-1M')
        fechas = [fecha - pd.offsets.MonthEnd(0) for fecha in fechas]
    elif periodos == 'grafico':
        fechas = pd.date_range(fecha, periods=36, freq='-1M')
        fechas = [fecha - pd.offsets.MonthEnd(0) for fecha in fechas]  

    fechas.sort()
    mes_ant = fechas[-1].replace(day=1) - timedelta(days=1)
    mes_dic = pd.to_datetime(fecha) - pd.offsets.YearEnd(1)
    fechas.append(mes_dic)
    fechas.append(mes_ant)

    bal_total2 = bal_total2.loc[fechas, cuentas]
    bal_total2.index = (pd.to_datetime(bal_total2.index)).strftime("%b %Y") ##### se cambia formato de fecha
    
    if bal_total2.shape[1] >= 36: #=============> modificar
        bal_anual = bal_total2.iloc[:, :]
    else: 
        bal_anual = bal_total2
    
    var_anual = bal_anual.pct_change(periods=1)
    var_anual.dropna(axis=0, inplace=True, how='all')
    var_anual.loc['Variación promedio'] = var_anual.mean(axis=0)
    var_anual = var_anual.T
    var_anual = var_anual.rename(columns={pd.to_datetime(fecha).strftime("%b %Y"): 'Variación interanual'}) ##### se cambia nombre de la columna
    var_anual = var_anual.fillna(0)

    bal_anual = bal_anual.T
    bal_anual = bal_anual.fillna(0) ####################
 

    bal_dic = bal_total2.iloc[4:6, :]
    bal_dic = bal_dic.sort_index()
    bal_dic = bal_dic.fillna(0) ##########################

    var_dic = bal_dic.pct_change(periods=1)
    var_dic.dropna(axis=0, inplace=True, how='all')
    var_dic = var_dic.T
    var_dic = var_dic.set_axis(['Variación cierre último año'], axis=1) #### cambio nombre columna
    var_dic = var_dic.fillna(0) ####################
    

    bal_mensual = bal_total2.iloc[4:7, :]
    bal_mensual = bal_mensual.sort_index()
    bal_mensual = bal_mensual.drop(bal_mensual.index[0])
    var_mensual = bal_mensual.pct_change(periods=1)
    var_mensual.dropna(axis=0, inplace=True, how='all')
    var_mensual = var_mensual.T
    var_mensual = var_mensual.set_axis(['Variación último mes'], axis=1) #### cambio nombre columna
    var_mensual = var_mensual.fillna(0) 
    
    try:#=============> modificar
        bal_total2 = pd.concat([bal_anual, var_anual[[
                                'Variación promedio', 'Variación interanual']], var_mensual,
                                var_dic], axis=1)
    except:
        bal_total2 = pd.concat([bal_anual, var_anual[[
                                'Variación promedio']], var_mensual,
                                var_dic], axis=1)

    bal_total2 = bal_total2.reset_index()
    bal_total1 = bal_total1.drop_duplicates(subset='CUENTA')
    bal_total3 = pd.merge(bal_total2, bal_total1[[
                            'CUENTA', 'DESCRIPCION_CUENTA']], how='left', on='CUENTA')
    #bal_total3 = bal_total3.set_index('DESCRIPCION_CUENTA')
    #bal_total3 = bal_total3.drop('CUENTA', axis=1)

    dfs = {
        'bal_dic':bal_dic,
        'bal_total3':bal_total3,
        'bal_mensual': bal_mensual
    }
    return dfs

# Tabla activos y pasivos
periodos = 'anuales'#=============> modificar
bal_total = bal_total
cuentas = [1, 2, 3]
dfs1 = cuenta_balance(ruc, fecha, cuentas, periodos, bal_total)
activos = dfs1.get('bal_total3')
activos_dic = dfs1.get('bal_dic')

# 3 GRÁFICOS

etiq = activos.columns[0:5]
act = pd.Series(activos.iloc[0, 0:5])
#act = act.str.replace('[$,]', '', regex=True).astype(float)
pas = pd.Series(activos.iloc[1, 0:5])
#pas = pas.str.replace('[$,]', '', regex=True).astype(float)
pat = pd.Series(activos.iloc[2, 0:5])
#pat = pat.str.replace('[$,]', '', regex=True).astype(float)


# Razón de activos y pasivos

fig = go.Figure()
fig.add_trace(go.Bar(
    x=etiq,
    y=act,
    name='Activos',
    marker_color='indianred',
    #text=act,
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=etiq,
    y=pas,
    name='Pasivos',
    marker_color='rgb(55, 83, 109)',
    #text=pas,
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=etiq,
    y=pat,
    name='Patrimonio',
    marker_color='rgb(26, 118, 255)',
    #text=pat,
    textposition='auto'
))

fig.update_layout(barmode='stack', xaxis_tickangle=0, 
                  title="Razón de Activos y Pasivos",
                  xaxis_tickfont_size = 10,
                  height=400, 
                  width=500
                      )
fig.update_traces(textangle=-90, 
                  texttemplate='$%{y:,.0f}')



# Avance de cuentas a la fecha

etiq2 = activos.index[0:4]
dic = activos_dic.iloc[0, 0:3]
actual = activos_dic.iloc[1, 0:3]


fig = go.Figure()
fig.add_trace(go.Bar(
    x=etiq2,
    y=dic,
    name=activos_dic.index[0],
    marker_color='indianred',
))
fig.add_trace(go.Bar(
    x=etiq2,
    y=actual,
    name= 'Avance a '+str(activos_dic.index[1]),
    marker_color='lightslategrey',
))

fig.update_layout(barmode='group', xaxis_tickangle=0, 
                  title="Avance respecto al cierre del último año",
                  xaxis_tickfont_size = 10,
                  height=400, 
                  width=500
                      )
fig.update_traces(textangle=-90, 
                  texttemplate='$%{y:,.0f}')
fig.show()


# Tabla Estado de Resultados

periodos = 'anuales'#=============> modificar
cuentas = [5, 4, 3603, 3604]

dfs1 = cuenta_balance(ruc, fecha, cuentas, periodos, bal_total)
resultados = dfs1.get('bal_total3')
resultados_dic = dfs1.get('bal_dic')




# Tabla Estado de Cuentas de cartera

periodos = 'anuales'#=============> modificar
cuentas = [14, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420,
           1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441,
           1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462,
           1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1475, 1477, 1479, 1481, 1483, 1485, 1487, 1489, 1499,
           149905,149910,149915,149920,149925,149930,149935,149940,149945,149950,149955]

dfs1 = cuenta_balance(ruc, fecha, cuentas, periodos, bal_total)
cartera = dfs1.get('bal_total3')
cartera_dic = dfs1.get('bal_dic')
cartera_mensual = dfs1.get('bal_mensual')


# =============>> Función para calcular participación y variaciones

def calcular_variaciones (tab_datos, tab_dic, tab_mensual):
    
    tab_datos = tab_datos.fillna(0)
    tab_datos = tab_datos.replace([np.inf, -np.inf], [1, -1])
    tab_datos['Participación'] = (tab_datos.iloc[:,-1]/ tab_datos.iloc[:,-1].sum()) # Participación

    tab_actual =tab_datos.iloc[:,0:5].T
    var_anual = tab_actual.pct_change(periods=1)
    var_anual = var_anual.drop(var_anual.index[0])
    var_anual.loc['Variación promedio'] = var_anual.mean(axis=0)
    var_anual = var_anual.T
    var_anual = var_anual.rename(columns={var_anual.columns[-2]: 'Variación interanual'}) ##### ============== cambiar
    var_anual = var_anual.fillna(0)
    var_anual = var_anual.replace([np.inf, -np.inf], [1, -1])

    tab_dic = tab_dic.T
    var_dic = tab_dic.pct_change(periods=1)
    var_dic = var_dic.drop(var_dic.index[0])
    var_dic = var_dic.T
    var_dic = var_dic.set_axis(['Variación cierre último año'], axis=1)
    var_dic = var_dic.fillna(0)
    var_dic = var_dic.replace([np.inf, -np.inf], [1, -1])    

    tab_mensual = tab_mensual.T
    var_mensual = tab_mensual.pct_change(periods=1)
    var_mensual = var_mensual.drop(var_mensual.index[0])
    var_mensual = var_mensual.T
    var_mensual = var_mensual.set_axis(['Variación último mes'], axis=1)
    var_mensual = var_mensual.fillna(0)
    var_mensual = var_mensual.replace([np.inf, -np.inf], [1, -1]) 

    tab_final = pd.concat([tab_datos, var_anual[[
                                'Variación promedio', 'Variación interanual']], var_mensual,
                                var_dic], axis=1)
    tab_final = tab_final.fillna(0)
    tab_final = tab_final.replace([np.inf, -np.inf], [1, -1])
    return tab_final


# COMPOSICIÓN GENERAL DE LA CARTERA

if cartera.shape[1] >= 36: #=============> modificar
        cartera_actual = cartera.iloc[:, :-1]
else: 
        cartera_actual = cartera.iloc[:,0:6]

cartera_actual = cartera_actual.set_index('CUENTA')
cartera_dic = cartera_dic.T
cartera_mensual = cartera_mensual.T


cartera = cartera_actual
def comp_gen_cart(cartera):
    cart_gen = pd.DataFrame(columns=cartera.columns)
    cart_gen.loc['Cartera Bruta'] = cartera.loc[14] - cartera.loc[1499]
    cart_gen.loc['Cartera Neta'] = cartera.loc[14]

#=============================================================================> aumentar catera por vencer
    cart_gen.loc['Cartera por Vencer'] = cartera.loc[[1401,1402,1403,1404,1405,1406,
                                                      1407,1408,1409,1410,1411,1412,
                                                      1413,1414,1415,1416,1417,1418,
                                                      1419,1420,1421,1422,1423,1424,
                                                      1473,1475,1477,]].sum(axis=0)

    cart_gen.loc['Cartera No Devenga'] = cartera.loc[[1425,1426,1427,1428,1429,1430,
                                                    1431,1432,1433,1434,1435,1436,
                                                    1437,1438,1439,1440,1441,1442,
                                                    1443,1444,1445,1446,1447,1448]].sum(axis=0)

    cart_gen.loc['Cartera Vencida'] = cartera.loc[[1449,1450,1451,1452,1453,1454,
                                                    1455,1456,1457,1458,1459,1460,
                                                    1461,1462,1463,1464,1465,1466,
                                                    1467,1468,1469,1470,1471,1472,
                                                    1479,1481,1483,1485,1487,1489]].sum(axis=0)

    cart_gen.loc['Cartera Improductiva'] = cartera.loc[[1425,1426,1427,1428,1429,1430,
                                                    1431,1432,1433,1434,1435,1436,
                                                    1437,1438,1439,1440,1441,1442,
                                                    1443,1444,1445,1446,1447,1448,
                                                    1449,1450,1451,1452,1453,1454,
                                                    1455,1456,1457,1458,1459,1460,
                                                    1461,1462,1463,1464,1465,1466,
                                                    1467,1468,1469,1470,1471,1472,
                                                    1479,1481,1483,1485,1487,1489]].sum(axis=0)
    cart_gen.loc['Provisiones'] = cartera.loc[1499]*-1
    cart_gen.loc['Morosidad Cartera'] = (cart_gen.loc['Cartera Improductiva'] / cart_gen.loc['Cartera Bruta'])*100
    cart_gen.loc['Cobertura'] = (cart_gen.loc['Provisiones'] / cart_gen.loc['Cartera Improductiva'])*100

    return cart_gen

cartera = cartera_actual
cart_gen_actual = comp_gen_cart(cartera)

cartera = cartera_dic
cart_gen_dic = comp_gen_cart(cartera)

cartera = cartera_mensual
cart_gen_mes = comp_gen_cart(cartera)

tab_datos = cart_gen_actual
tab_dic = cart_gen_dic
tab_mensual = cart_gen_mes

cart_gen_fin = calcular_variaciones (tab_datos, tab_dic, tab_mensual)
cart_gen_fin = cart_gen_fin.drop('Participación', axis =1)




# GRAFICO COMPOSICIÓN DE LA  CARTERA ================================>>>>>

cart_gen_graf = cart_gen_fin.drop(index=['Cartera Neta',
                                         'Cartera Improductiva',
                                         'Morosidad Cartera',
                                         'Cobertura'])

fig = go.Figure(data=[go.Pie(labels=cart_gen_graf.index[1:], 
                             values=cart_gen_graf.iloc[1:,4],
                             showlegend=False, 
                             text = cart_gen_graf.iloc[1:,4].apply(convertir_a_dolares),
                             textfont={"family": "Arial", "size": 9, "color": "black"},
                             textinfo='percent+label+text',
                             marker= dict(colors = ['rgb(158,202,225)',
                                                    'darkviolet', 
                                                    'crimson',
                                                    'royalblue']))])
fig.update_layout(title='Composición de la cartera', title_y=0.95)
fig.show()



# COMPOSICIÓN CARTERA REFINANCIADA Y RESTRUCTURADA


def comp_cart_nov(cartera):
    cart_nov = pd.DataFrame(columns=cartera.columns)
    cart_nov.loc['Cartera Bruta'] = cartera.loc[14] - cartera.loc[1499]
    cart_nov.loc['Cartera Refinanciada Neta'] = cartera.loc[[1409,1410,1411,1412,1413,1414,
                                                        1415,1416,1433,1434,1435,1436,
                                                        1437,1438,1439,1440,1457,1458,
                                                        1459,1460,1461,1462,1463,1464,
                                                        1475,1481,1487]].sum(axis=0)
    cart_nov.loc['Provisiones Refinanciada'] = cartera.loc[149945]*-1
    cart_nov.loc['Cartera Reestructurada Neta'] = cartera.loc[[1417,1418,1419,1420,1421,1422,
                                                        1423,1424,1441,1442,1443,1444,
                                                        1445,1446,1447,1448,1465,1466,
                                                        1467,1468,1469,1470,1471,1472,
                                                        1477,1483,1489]].sum(axis=0)
    cart_nov.loc['Provisiones Reestructurada'] = cartera.loc[149950]*-1
    cart_nov.loc['Cartera Refinanciada'] = (cart_nov.loc['Cartera Refinanciada Neta'] + 
                                            cart_nov.loc['Provisiones Refinanciada'])
    cart_nov.loc['Cartera Reestructurada'] = (cart_nov.loc['Cartera Reestructurada Neta'] + 
                                                cart_nov.loc['Provisiones Reestructurada'])
    cart_nov = cart_nov.loc[['Cartera Bruta','Cartera Refinanciada','Cartera Reestructurada']]

    return cart_nov

cartera = cartera_actual
cart_nov_act = comp_cart_nov(cartera )

cartera = cartera_dic
cart_nov_dic = comp_cart_nov(cartera )

cartera = cartera_mensual
cart_nov_mes = comp_cart_nov(cartera )

tab_datos = cart_nov_act
tab_dic = cart_nov_dic
tab_mensual = cart_nov_mes

cart_nov_fin = calcular_variaciones (tab_datos, tab_dic, tab_mensual)



#### Grafico Calidad de la Cartera ======================================>>>>>>>

fig = go.Figure(go.Funnel(
    y = cart_nov_fin.index[1:],
    x = cart_nov_fin.iloc[1:,4],
    text = cart_nov_fin.iloc[1:,5].apply(convertir_a_porcentaje).values.tolist(),
    textposition = "inside",
    textinfo = "value+text",
    textfont=dict(color='black', size=14, family='Arial'),
    opacity = 0.98, marker = {"color": ["deepskyblue", "lightsalmon"],
    "line": {"width": [4, 4], "color": ["wheat", "wheat"]}},
    connector = {"line": {"color": "royalblue", "dash": "dot", "width": 3}})
    )
fig.update_traces(texttemplate='$%{value:,.0f}<br>%{text}', 
                  hovertemplate='$%{value:,.0f}<br>%{text}')
fig.update_layout(title='Cartera Refinanciada y Reestructurada', 
                  title_y=0.95)
fig.show()




# MORORSIDAD CARTERA REFINANCIADA

def comp_cart_ref (cartera, cart_nov):
    cart_ref = pd.DataFrame(columns=cartera.columns)
    cart_ref.loc['Cartera Refinanciada'] = cart_nov.loc['Cartera Refinanciada']
    cart_ref.loc['Ref. Improductiva'] = cartera.loc[[1433,1434,1435,1436,1437,1438,
                                                            1439,1440,1457,1458,1459,1460,
                                                            1461,1462,1463,1464,1481,1487]].sum(axis=0)
    cart_ref.loc['Provisiones Ref.'] = cartera.loc[149945]*-1 #######
    cart_ref.loc['Morosidad Ref.'] = (cart_ref.loc['Ref. Improductiva'] / cart_ref.loc['Cartera Refinanciada']) ######
    cart_ref.loc['Cobertura Ref.'] = (cart_ref.loc['Provisiones Ref.'] / cart_ref.loc['Ref. Improductiva']) ########
    cart_ref = cart_ref.fillna(0)
    return cart_ref

cartera = cartera_actual
cart_nov = cart_nov_act
cart_ref_actual = comp_cart_ref(cartera, cart_nov)

cartera = cartera_dic
cart_nov = cart_nov_dic
cart_ref_dic = comp_cart_ref(cartera, cart_nov)

cartera = cartera_mensual
cart_nov = cart_nov_mes
cart_ref_mes = comp_cart_ref(cartera, cart_nov)

tab_datos = cart_ref_actual
tab_dic = cart_ref_dic
tab_mensual = cart_ref_mes

cart_ref_fin = calcular_variaciones (tab_datos, tab_dic, tab_mensual)
cart_ref_fin = cart_ref_fin.drop('Participación', axis =1) ##############################

#### Grafico cartera refinanciada ==================================================>>>>

fig = go.Figure(go.Funnelarea(
    labels=['Mora Ref.','Cobertura'],
    values=cart_ref_fin.iloc[1:3, 4], 
    text = cart_ref_fin.iloc[3:, 4].apply(convertir_a_porcentaje).values.tolist(),
    textinfo="value+text",
    textfont={"family": "Arial", "size": 18, "color": "black"},
    marker=dict(colors=["deepskyblue", "lightsalmon"],
                pattern=dict(shape=["", "/"]))))

fig.update_layout(legend=dict(font=dict(size=9)),
                  title='Morosidad Cartera Refinanciada', 
                  title_y=0.95)
fig.update_traces(texttemplate='$%{value:,.0f}<br>%{text}', 
                  hovertemplate='$%{value:,.0f}<br>%{text}')


# MORORSIDAD CARTERA REESTRUCTURADA

def comp_cart_res (cartera, cart_nov):
    cart_res = pd.DataFrame(columns=cartera.columns)
    cart_res.loc['Cartera Reestructurada'] = cart_nov.loc['Cartera Reestructurada']
    cart_res.loc['Reest. Improductiva'] = cartera.loc[[1441,1442,1443,1444,1445,1446,
                                                            1447,1448,1465,1466,1467,1468,
                                                            1469,1470,1471,1472,1483,1489]].sum(axis=0)
    
    cart_res.loc['Provisiones Reest.'] = cartera.loc[149950]*-1 #######
    cart_res.loc['Mora Reest.'] = (cart_res.loc['Reest. Improductiva'] / cart_res.loc['Cartera Reestructurada']) ######
    cart_res.loc['Cobertura'] = (cart_res.loc['Provisiones Reest.'] / cart_res.loc['Reest. Improductiva']) ########
    cart_res = cart_res.fillna(0)
    return cart_res


cartera = cartera_actual
cart_nov = cart_nov_act
cart_res_actual = comp_cart_res(cartera, cart_nov)

cartera = cartera_dic
cart_nov = cart_nov_dic
cart_res_dic = comp_cart_res(cartera, cart_nov)

cartera = cartera_mensual
cart_nov = cart_nov_mes
cart_res_mes = comp_cart_res(cartera, cart_nov)

tab_datos = cart_res_actual
tab_dic = cart_res_dic
tab_mensual = cart_res_mes

cart_res_fin = calcular_variaciones (tab_datos, tab_dic, tab_mensual)
cart_res_fin = cart_res_fin.drop('Participación', axis =1) ##############################

#### Grafico cartera refinanciada ==================================================>>>>

fig = go.Figure(go.Funnelarea(
    labels=['Mora Reest.','Cobertura'],
    values=cart_res_fin.iloc[1:3, 4], 
    text = cart_res_fin.iloc[3:, 4].apply(convertir_a_porcentaje).values.tolist(),
    textinfo="value+text",
    textfont={"family": "Arial", "size": 18, "color": "black"},
    marker=dict(colors=["deepskyblue", "lightsalmon"],
                pattern=dict(shape=["", "/"]))))

fig.update_layout(legend=dict(font=dict(size=9)),
                  title='Morosidad Cartera Reestructurada', 
                  title_y=0.95)
fig.update_traces(texttemplate='$%{value:,.0f}<br>%{text}', 
                  hovertemplate='$%{value:,.0f}<br>%{text}')


# COMPOSICÓN DE LA CARTERA POR TIPO DE CRÉDITO

def comp_cart_tipo(cartera):
    cart_tipo = pd.DataFrame(columns=cartera.columns)

    cart_tipo.loc['Comercial Prioritario'] = cartera.loc[[1401,1409,1417,1425,1433,1441,
                                                        1449,1457,1465,149905]].sum(axis=0)
    cart_tipo.loc['P Comercial Prioritario'] = cartera.loc[149905]*-1
    cart_tipo.loc['Imp Comercial Prioritario'] = cartera.loc[[1425,1433,1441,1449,1457,1465]].sum(axis=0)

    cart_tipo.loc['Comercial Ordinario'] = cartera.loc[[1406,1414,1422,1430,1438,1446,
                                                        1454,1462,1470,149930]].sum(axis=0)
    cart_tipo.loc['P Comercial Ordinario'] = cartera.loc[149930]*-1
    cart_tipo.loc['Imp Comercial Ordinario'] = cartera.loc[[1430,1438,1446,1454,1462,1470]].sum(axis=0)

    cart_tipo.loc['Productivo'] = cartera.loc[[1405,1413,1421,1429,1437,1445,1453,
                                                        1461,1469,149925]].sum(axis=0)
    cart_tipo.loc['P Productivo'] = cartera.loc[149925]*-1
    cart_tipo.loc['Imp Productivo'] = cartera.loc[[1429,1437,1445,1453,1461,1469]].sum(axis=0)

    cart_tipo.loc['Microcrédito'] = cartera.loc[[1404,1412,1420,1428,1436,1444,
                                                1452,1460,1468,149920]].sum(axis=0)
    cart_tipo.loc['P Microcrédito'] = cartera.loc[149920]*-1
    cart_tipo.loc['Imp Microcrédito'] = cartera.loc[[1428,1436,1444,1452,1460,1468]].sum(axis=0)

    cart_tipo.loc['Consumo Prioritario'] = cartera.loc[[1410,1418,1434,1442,1458,
                                                        1466,1402,1426,1450,149910]].sum(axis=0)
    cart_tipo.loc['P Consumo Prioritario'] = cartera.loc[149910]*-1
    cart_tipo.loc['Imp Consumo Prioritario'] = cartera.loc[[1426,1434,1442,1450,1458,1466]].sum(axis=0)

    cart_tipo.loc['Consumo Ordinario'] = cartera.loc[[1415,1423,1431,1439,1447,1455,
                                                    1463,1471,1407,149935]].sum(axis=0)
    cart_tipo.loc['P Consumo Ordinario'] = cartera.loc[149935]*-1
    cart_tipo.loc['Imp Consumo Ordinario'] = cartera.loc[[1431,1439,1447,1455,1463,1471]].sum(axis=0)

    cart_tipo.loc['Vivienda'] = cartera.loc[[1408,1416,1424,1432,1440,
                                                    1448,1456,1464,1472,149940]].sum(axis=0)
    cart_tipo.loc['P Vivienda'] = cartera.loc[149940]*-1
    cart_tipo.loc['Imp Vivienda'] = cartera.loc[[1432,1440,1448,1456,1464,1472]].sum(axis=0)

    cart_tipo.loc['Inmobiliario'] = cartera.loc[[1411,1419,1435,1443,1459,1467,
                                                1403,1427,1451,149915]].sum(axis=0)
    cart_tipo.loc['P Inmobiliario'] = cartera.loc[149915]*-1
    cart_tipo.loc['Imp Inmobiliario'] = cartera.loc[[1427,1435,1443,1451,1459,1467]].sum(axis=0)

    cart_tipo.loc['Educativo'] = cartera.loc[[1475,1477,1479,1481,1483,1485,
                                            1487,1489,1473,149955]].sum(axis=0)
    cart_tipo.loc['P Educativo'] = cartera.loc[149955]*-1
    cart_tipo.loc['Imp Educativo'] = cartera.loc[[1479,1481,1483,1485,1487,1489]].sum(axis=0)

    return cart_tipo

cartera = cartera_actual
cart_tipo_actual = comp_cart_tipo(cartera)

cartera = cartera_dic
cart_tipo_dic = comp_cart_tipo(cartera)

cartera = cartera_mensual
cart_tipo_mes = comp_cart_tipo(cartera)

#========================> COMPOSICIÓN CARTERA BRUTA

def comp_cart_bruta(cart_tipo):
    cart_bruta = pd.DataFrame(columns=cart_tipo.columns)
    cart_bruta.loc['Crédito Productivo'] = cart_tipo.loc[['Comercial Prioritario',
                                                        'Comercial Ordinario',
                                                        'Productivo',
                                                        'P Comercial Prioritario',
                                                        'P Comercial Ordinario',
                                                        'P Productivo']].sum(axis=0)
    cart_bruta.loc['Microcrédito'] = cart_tipo.loc[['Microcrédito',
                                                    'P Microcrédito']].sum(axis=0)                    
    cart_bruta.loc['Crédito Consumo'] = cart_tipo.loc[['Consumo Prioritario',
                                                    'Consumo Ordinario',
                                                    'P Consumo Prioritario',
                                                    'P Consumo Ordinario']].sum(axis=0)
    cart_bruta.loc['Crédito Vivienda'] = cart_tipo.loc[['Vivienda',
                                                        'P Vivienda']].sum(axis=0)
    cart_bruta.loc['Crédito Inmobiliario'] = cart_tipo.loc[['Inmobiliario',
                                                            'P Inmobiliario']].sum(axis=0)
    cart_bruta.loc['Crédito Educativo'] = cart_tipo.loc[['Educativo',
                                                        'P Educativo']].sum(axis=0)
    return cart_bruta

cart_tipo = cart_tipo_actual
cart_bruta_actual = comp_cart_bruta(cart_tipo)

cart_tipo = cart_tipo_dic
cart_bruta_dic = comp_cart_bruta(cart_tipo)

cart_tipo = cart_tipo_mes
cart_bruta_mes = comp_cart_bruta(cart_tipo)

tab_datos = cart_bruta_actual
tab_dic = cart_bruta_dic
tab_mensual = cart_bruta_mes

cart_bruta_fin = calcular_variaciones (tab_datos, tab_dic, tab_mensual)


#========================================================================>>>>>>>>
# Gráfico de composición de la cartera

# GRAFICO COMPOSICIÓN DE LA  CARTERA ================================>>>>>

fig = go.Figure(data=[go.Pie(labels=cart_bruta_fin.index[0:], 
                             values=cart_bruta_fin.iloc[0:,4],
                             showlegend=False,
                             text = cart_bruta_fin.iloc[0:,4].apply(convertir_a_dolares),
                             textinfo='text+percent+label',
                             textfont_size=8,
                             marker= dict(colors = ['moccasin',
                                                    'orange',
                                                    'lightviolet',
                                                    'indianred',
                                                    'brown',
                                                    'lightslategray']))])
fig.update_layout(title='Cartera por Tipo de Crédito', title_y=0.95)
fig.show()


# MOROSIDAD POR TIPO DE CRÉDITO

def mora_tipo(cart_tipo,cart_bruta):
    mora_tipo = pd.DataFrame(columns=cart_tipo.columns)
    mora_tipo.loc['Crédito Productivo'] = (cart_tipo.loc[['Imp Comercial Prioritario',
                                                        'Imp Comercial Ordinario',
                                                        'Imp Productivo']].sum(axis=0)) / cart_bruta.loc['Crédito Productivo']
    mora_tipo.loc['Microcrédito'] = cart_tipo.loc['Imp Microcrédito'] / cart_bruta.loc['Microcrédito']
    mora_tipo.loc['Crédito Consumo'] = (cart_tipo.loc[['Imp Consumo Prioritario',
                                                    'Imp Consumo Ordinario']].sum(axis=0)) / cart_bruta.loc['Crédito Consumo']
    mora_tipo.loc['Crédito Vivienda'] = cart_tipo.loc['Imp Vivienda'] / cart_bruta.loc['Crédito Vivienda']
    mora_tipo.loc['Crédito Inmobiliario'] = cart_tipo.loc['Imp Inmobiliario'] / cart_bruta.loc['Crédito Inmobiliario']
    mora_tipo.loc['Crédito Educativo'] = cart_tipo.loc['Imp Educativo'] / cart_bruta.loc['Crédito Educativo']
    return mora_tipo

cart_tipo = cart_tipo_actual
cart_bruta = cart_bruta_actual
mora_tipo_actual = mora_tipo(cart_tipo, cart_bruta)

cart_tipo = cart_tipo_dic
cart_bruta = cart_bruta_dic
mora_tipo_dic = mora_tipo(cart_tipo, cart_bruta)

cart_tipo = cart_tipo_mes
cart_bruta = cart_bruta_mes
mora_tipo_mes = mora_tipo(cart_tipo, cart_bruta)

tab_datos = mora_tipo_actual
tab_dic = mora_tipo_dic
tab_mensual = mora_tipo_mes

mora_tipo_fin = calcular_variaciones (tab_datos, tab_dic, tab_mensual)
mora_tipo_fin = mora_tipo_fin.drop('Participación', axis =1)
mora_tipo_fin['Promedio'] = mora_tipo_fin.iloc[:, 0:5].mean(axis=1)
mora_tipo_fin.insert(5, 'Promedio',mora_tipo_fin.pop('Promedio'))



# Gráfico de Morosidad por tipo de crédito ========================================================================>>>
graf_tipo_fin = mora_tipo_fin.loc[(mora_tipo_fin != 0).any(axis=1)].dropna()
graf_tipo_fin = graf_tipo_fin.iloc[0:,0:5].T

fig = go.Figure()
for column in graf_tipo_fin.columns:
    fig.add_trace(go.Scatter(
        x=graf_tipo_fin.index,
        y=graf_tipo_fin[column],
        mode='lines+markers+text',
        name=column,
        text=graf_tipo_fin[column].apply(convertir_a_porcentaje).values.tolist(),
        textposition='top center',
        textfont={"family": "Arial", "size": 8, "color": "black"}
    ))
fig.update_layout(
    title='Morosidad - Evolución anual ',
    yaxis_title='Índice de Morosidad',
    legend=dict(orientation='h', x = 0.25, y=-0.07),
    margin=dict(b=50))
fig.show()



# COBERTURA POR TIPO DE CRÉDITO


def cobertura_tipo (cart_tipo):
    if cart_tipo.shape[1] >= 36: ############=============> cambiar
        cob_tipo = pd.DataFrame(columns=cart_tipo.columns)
    else:
        cob_tipo = pd.DataFrame(columns=cart_tipo.columns[0:6])
    cob_tipo.loc['Crédito Productivo'] = ((cart_tipo.loc[['P Comercial Prioritario',
                                                        'P Comercial Ordinario',
                                                        'P Productivo']].sum(axis=0)) / 
                                        (cart_tipo.loc[['Imp Comercial Prioritario',
                                                                        'Imp Comercial Ordinario',
                                                                        'Imp Productivo']].sum(axis=0)))
    cob_tipo.loc['Microcrédito'] = cart_tipo.loc['P Microcrédito'] / cart_tipo.loc['Imp Microcrédito']
    cob_tipo.loc['Crédito Consumo'] = ((cart_tipo.loc[['P Consumo Prioritario',
                                                    'P Consumo Ordinario']].sum(axis=0)) /
                                    (cart_tipo.loc[['Imp Consumo Prioritario',
                                                    'Imp Consumo Ordinario' ]].sum(axis=0)))
    cob_tipo.loc['Crédito Vivienda'] = cart_tipo.loc['P Vivienda'] / cart_tipo.loc['Imp Vivienda']
    cob_tipo.loc['Crédito Inmobiliario'] = cart_tipo.loc['P Inmobiliario'] / cart_tipo.loc['Imp Inmobiliario'] 
    cob_tipo.loc['Crédito Educativo'] = cart_tipo.loc['P Educativo'] / cart_tipo.loc['Imp Educativo']
    return cob_tipo

cart_tipo = cart_tipo_actual
cob_tipo_actual = cobertura_tipo(cart_tipo)

cart_tipo = cart_tipo_dic
cob_tipo_dic = cobertura_tipo(cart_tipo)

cart_tipo = cart_tipo_mes
cob_tipo_mes = cobertura_tipo(cart_tipo)

tab_datos = cob_tipo_actual
tab_dic = cob_tipo_dic
tab_mensual = cob_tipo_mes

cob_final = calcular_variaciones (tab_datos, tab_dic, tab_mensual)
cob_final = cob_final.drop('Participación', axis =1)
cob_final['Promedio'] = cob_final.iloc[:, 0:5].mean(axis=1)
cob_final.insert(5, 'Promedio',cob_final.pop('Promedio'))


# Gráfico de Morosidad por tipo de crédito ========================================================================>>>

graf_cob_final = cob_final.loc[graf_tipo_fin.columns]
graf_cob_final = graf_cob_final.iloc[0:,0:5].T

fig = go.Figure()
for column in graf_cob_final.columns:
    fig.add_trace(go.Scatter(
        x=graf_cob_final.index,
        y=graf_cob_final[column],
        mode='lines+markers+text',
        name=column,
        text=graf_cob_final[column].apply(convertir_a_porcentaje).values.tolist(),
        textposition='top center',
        textfont={"family": "Arial", "size": 8, "color": "black"}
    ))
fig.update_layout(
    title='Cobertura - Evolución anual ',
    yaxis_title='Índice de Cobertura',
    legend=dict(orientation='h', x = 0.25, y=-0.07),
    margin=dict(b=50))
fig.show()


###################################################################
# ESTADO DE SITUACIÓN

def tabla_financiera ( balance_actual, balance_dic, balance_mensual):
        
    desc_cuent = balance_actual['DESCRIPCION_CUENTA']
    balance_actual = balance_actual.iloc[:,0:6]
    balance_actual = balance_actual.set_index('CUENTA')

    balance_dic = balance_dic.T
    balance_mensual = balance_mensual.T

    tab_datos = balance_actual
    tab_dic = balance_dic
    tab_mensual = balance_mensual

    tab_final = calcular_variaciones (tab_datos, tab_dic, tab_mensual)
    tab_final = tab_final.drop('Participación', axis =1)
    tab_final['Participación'] = tab_final.iloc[:, 4] / tab_final.iloc[0,4]
    tab_final.insert(5, 'Participación',tab_final.pop('Participación'))

    desc_cuent.reset_index(drop=True, inplace=True)
    tab_final.reset_index(drop=True, inplace=True)
    tab_final = pd.concat([desc_cuent, tab_final], axis=1)
    tab_final = tab_final.set_index('DESCRIPCION_CUENTA')
    return tab_final

#ACTIVOS

cuentas = [1,11,12,13,14,1499,15,16,17,18,19]
dfs1 = cuenta_balance(ruc, fecha, cuentas, bal_total)
balance_actual = dfs1.get('bal_total3')
balance_dic = dfs1.get('bal_dic')
balance_mensual = dfs1.get('bal_mensual')
act_fin = tabla_financiera ( balance_actual, balance_dic, balance_mensual)

#PASIVOS

cuentas = [2,21,2101,2102,2103,2104,2105,23,24,25,26,27,28,29]
dfs1 = cuenta_balance(ruc, fecha, cuentas, bal_total)
balance_actual = dfs1.get('bal_total3')
balance_dic = dfs1.get('bal_dic')
balance_mensual = dfs1.get('bal_mensual')
pas_fin = tabla_financiera ( balance_actual, balance_dic, balance_mensual)


#PATRIMONIO

cuentas = [3,31,33,34,35,36]
dfs1 = cuenta_balance(ruc, fecha, cuentas, bal_total)
balance_actual = dfs1.get('bal_total3')
balance_dic = dfs1.get('bal_dic')
balance_mensual = dfs1.get('bal_mensual')
pat_fin = tabla_financiera ( balance_actual, balance_dic, balance_mensual)

balance_final = pd.concat([act_fin, pas_fin, pat_fin], axis=0)

########################
# ESTADO DE RESULTADOS

#INGRESOS

cuentas = [5,51,52,53,54,55,56,59]
dfs1 = cuenta_balance(ruc, fecha, cuentas, bal_total)
balance_actual = dfs1.get('bal_total3')
balance_dic = dfs1.get('bal_dic')
balance_mensual = dfs1.get('bal_mensual')
ing_fin = tabla_financiera ( balance_actual, balance_dic, balance_mensual)

# GASTOS

cuentas = [4,41,42,43,44,45,46,47,48]
dfs1 = cuenta_balance(ruc, fecha, cuentas, bal_total)
balance_actual = dfs1.get('bal_total3')
balance_dic = dfs1.get('bal_dic')
balance_mensual = dfs1.get('bal_mensual')
gas_fin = tabla_financiera ( balance_actual, balance_dic, balance_mensual)

res_final = pd.concat([ing_fin, gas_fin], axis=0)
res_final.loc['RESULTADO DEL EJERCICIO'] =res_final.loc['INGRESOS'] - res_final.loc['GASTOS']


#calcular resultado del ejercicio 5-4

# ESTRUCTURA DE LOS ACTIVOS 

# ACTIVOS PRODUCTIVOS

bal_total = bal_total
periodos = 'anuales'#=============> modificar
cuentas = [1103,12,13,15,1901,190205,190210,
           190215,190220,190240,190280,190286]
dfs1 = cuenta_balance(ruc, fecha, cuentas, periodos, bal_total)

balance_actual = dfs1.get('bal_total3')
balance_dic = dfs1.get('bal_dic')
balance_mensual = dfs1.get('bal_mensual')

act_prod = tabla_financiera ( balance_actual, balance_dic, balance_mensual)
act_prod.loc['CARTERA POR VENCER']= cart_gen_fin.loc['Cartera por Vencer']
p_vencer = act_prod.loc['CARTERA POR VENCER']
act_prod = act_prod.drop(index='CARTERA POR VENCER')
act_prod = pd.concat([act_prod.iloc[:3], p_vencer.to_frame().transpose(), act_prod.iloc[3:]])
tot_act_prod = act_prod.cumsum()
act_prod.loc['ACTIVO PRODUCTIVO'] = tot_act_prod.iloc[-1]
a_prod = act_prod.loc['ACTIVO PRODUCTIVO']
act_prod = act_prod.drop(index='ACTIVO PRODUCTIVO')
act_prod = pd.concat([act_prod.iloc[:0], a_prod.to_frame().transpose(), act_prod.iloc[0:]])
otros_act = act_prod.iloc[6:,]
otros_act = otros_act.cumsum()
act_prod = act_prod.iloc[0:6,]
act_prod.loc['OTROS ACTIVOS PRODUCTIVOS'] = otros_act.iloc[-1]

act_prod = act_prod.drop('Participación', axis =1)
act_prod['Participación'] = act_prod.iloc[:, 4] / act_prod.iloc[0,4]
act_prod.insert(5, 'Participación',act_prod.pop('Participación'))

#====  Gráfico activos improductivos

fig = go.Figure(data=[go.Pie(labels=act_prod.index[1:], 
                             values=act_prod.iloc[1:,4],
                             showlegend=False,
                             text = act_prod.iloc[1:,4].apply(convertir_a_dolares),
                             textinfo='text+percent+label',
                             textfont_size=8,
                             marker= dict(colors = ['moccasin',
                                                    'orange',
                                                    'lightviolet',
                                                    'indianred',
                                                    'brown',
                                                    'lightslategray']))])
fig.update_layout(title='Composición Activo Productivo', 
                  title_y=0.98,
                  margin=dict(b=4, t=4, l=4, r=4))
fig.show()



# ACTIVOS IMPRODUCTIVOS


cuentas = [11,1103,1499,15,16,17,18,19]
dfs1 = cuenta_balance(ruc, fecha, cuentas, periodos, bal_total)

balance_actual = dfs1.get('bal_total3')
balance_dic = dfs1.get('bal_dic')
balance_mensual = dfs1.get('bal_mensual')

act_improd = tabla_financiera ( balance_actual, balance_dic, balance_mensual)
act_improd.loc['(PROVISIONES PARA CREDITOS INCOBRABLES)'] = act_improd.loc['(PROVISIONES PARA CREDITOS INCOBRABLES)'] * -1
act_improd.loc['(FONDOS DISPONIBLES - BANCOS)'] = act_improd.loc['FONDOS DISPONIBLES'] - act_improd.loc['BANCOS Y OTRAS INSTITUCIONES FINANCIERAS']
act_improd.loc['CARTERA EN RIESGO'] = cart_gen_fin.loc['Cartera No Devenga'] + cart_gen_fin.loc['Cartera Vencida'] 
act_improd.loc['OTROS ACTIVOS IMPRODUCTIVOS'] = act_improd.loc['OTROS ACTIVOS'] - act_prod.loc['OTROS ACTIVOS PRODUCTIVOS']

act_improd = act_improd.loc[['(FONDOS DISPONIBLES - BANCOS)',
                        'CARTERA EN RIESGO',
                        '(PROVISIONES PARA CREDITOS INCOBRABLES)',
                        'DEUDORES POR ACEPTACION',
                        'CUENTAS POR COBRAR',
                        'BIENES REALIZABLES, ADJUDICADOS POR PAGO, DE ARRENDAMIENTO MERCANTIL Y NO UTILIZADOS POR LA INSTITUCION',
                        'PROPIEDADES Y EQUIPO',
                        'OTROS ACTIVOS IMPRODUCTIVOS']]

tot_act_improd = act_improd.cumsum() ############### ojo cambiar 
act_improd.loc['ACTIVO IMPRODUCTIVO'] = tot_act_improd.iloc[-1]
a_improd = act_improd.loc['ACTIVO IMPRODUCTIVO']
act_improd = act_improd.drop(index='ACTIVO IMPRODUCTIVO')
act_improd = pd.concat([act_improd.iloc[:0], a_improd.to_frame().transpose(), act_improd.iloc[0:]])

act_improd = act_improd.drop('Participación', axis =1)
act_improd['Participación'] = act_improd.iloc[:, 4] / act_improd.iloc[0,4]
act_improd.insert(5, 'Participación',act_improd.pop('Participación'))

act_improd.reset_index(inplace=True) #############
act_improd['index'] = act_improd['index'].str.split().apply(lambda x: ' '.join(x[:3]))
act_improd.set_index(act_improd['index'], inplace=True)

estr_activo = pd.concat([act_prod, act_improd], axis=0)

#====  Gráfico activos improductivos

fig = go.Figure(data=[go.Pie(labels=act_improd.index[1:], 
                             values=act_improd.iloc[1:,4],
                             showlegend=False,
                             text = act_improd.iloc[1:,4].apply(convertir_a_dolares),
                             textinfo='text+percent+label',
                             textfont_size=8,
                             marker= dict(colors = ['moccasin',
                                                    'orange',
                                                    'lightviolet',
                                                    'indianred',
                                                    'brown',
                                                    'lightslategray']))])
fig.update_layout(title='Composición Activo Improductivo', 
                  title_y=0.98,
                  margin=dict(b=4, t=4, l=4, r=4))
fig.show()


### tabla de porcentajes de participación
estr_act_tot = estr_activo.loc[['ACTIVO PRODUCTIVO','ACTIVO IMPRODUCTIVO']]
estr_act_tot = estr_act_tot.iloc[:,:5]
estr_act_tot.loc['TOTAL ACTIVO'] = estr_act_tot.loc['ACTIVO PRODUCTIVO']+ estr_act_tot.loc['ACTIVO IMPRODUCTIVO']
estr_act_tot.loc['ACTIVO PRODUCTIVO (%)'] = estr_act_tot.loc['ACTIVO PRODUCTIVO'] / estr_act_tot.loc['TOTAL ACTIVO']
estr_act_tot.loc['ACTIVO IMPRODUCTIVO (%)'] = estr_act_tot.loc['ACTIVO IMPRODUCTIVO'] / estr_act_tot.loc['TOTAL ACTIVO']



etiq = estr_act_tot.columns[0:5]
act_pr = pd.Series(estr_act_tot.iloc[3, 0:5])
#act = act.str.replace('[$,]', '', regex=True).astype(float)
act_imp = pd.Series(estr_act_tot.iloc[4, 0:5])


# Razón de activos y pasivos

fig = go.Figure()
fig.add_trace(go.Bar(
    x=etiq,
    y=act_pr,
    name='Activos Productivos',
    marker_color='indianred',
    #text=act,
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=etiq,
    y=act_imp,
    name='Activos Improductivos',
    marker_color='rgb(55, 83, 109)',
    #text=pas,
    textposition='auto'
))

fig.update_layout(barmode='stack', xaxis_tickangle=-45, 
                  title="Composición de Activos",
                  xaxis_tickfont_size = 9,
                  legend=dict(orientation='h', x=0.3, y=-0.2),
                  height=400, 
                  width=500
                      )
fig.update_traces(textangle=0, 
                  texttemplate='%{y:.0%}')



###### grafico series

######## GRÁFICO MORA VS COBERTURA

periodos = 'grafico'#=============> modificar
cuentas = [14, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420,
           1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441,
           1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462,
           1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1475, 1477, 1479, 1481, 1483, 1485, 1487, 1489, 1499,
           149905,149910,149915,149920,149925,149930,149935,149940,149945,149950,149955]

dfs1 = cuenta_balance(ruc, fecha, cuentas, periodos, bal_total)
cartera = dfs1.get('bal_total3')
cartera_dic = dfs1.get('bal_dic')
cartera_mensual = dfs1.get('bal_mensual')


if cartera.shape[1] >= 36:
        cartera_actual = cartera.iloc[:, :-1]
else: 
        cartera_actual = cartera.iloc[:,0:6]
cartera_actual = cartera_actual.set_index('CUENTA')
cartera_dic = cartera_dic.T
cartera_mensual = cartera_mensual.T


cartera = cartera_actual
cart_gen_actual = comp_gen_cart(cartera)

cartera = cartera_dic
cart_gen_dic = comp_gen_cart(cartera)

cartera = cartera_mensual
cart_gen_mes = comp_gen_cart(cartera)

tab_datos = cart_gen_actual
tab_dic = cart_gen_dic
tab_mensual = cart_gen_mes

cart_gen_fin = calcular_variaciones (tab_datos, tab_dic, tab_mensual)
cart_gen_fin = cart_gen_fin.drop('Participación', axis =1)



graf_cart_gen_fin = cart_gen_fin.iloc[7:9,0:-10].T

fig = go.Figure()

column_principal = "Cobertura" 
fig.add_trace(go.Scatter(
    x=graf_cart_gen_fin.index,
    y=graf_cart_gen_fin[column_principal],
    mode='lines+markers+text',
    name=column_principal,
    textposition='top center',
    textfont={"family": "Arial", "size": 8, "color": "black"}
))


column_secundario = "Morosidad Cartera"
fig.add_trace(go.Scatter(
    x=graf_cart_gen_fin.index,
    y=graf_cart_gen_fin[column_secundario],
    mode='lines+markers+text',
    name=column_secundario,
    textposition='top center',
    textfont={"family": "Arial", "size": 8, "color": "black"},
    yaxis='y2'
))

fig.update_layout(
    title='Evolución Morosidad Vs. Cobertura',
    yaxis_title=f'{column_principal}',
    yaxis2=dict(title=f'{column_secundario}', overlaying='y', side='right'),
    legend=dict(orientation='h', x=0.3, y=-0.3),
    margin=dict(b=50),
    showlegend=True 
)
fig.show()


###### GRÁFICO REFINANCIAMIENTO

cartera = cartera_actual
cart_nov_act = comp_cart_nov(cartera )

cartera = cartera_dic
cart_nov_dic = comp_cart_nov(cartera )

cartera = cartera_mensual
cart_nov_mes = comp_cart_nov(cartera )

tab_datos = cart_nov_act
tab_dic = cart_nov_dic
tab_mensual = cart_nov_mes

cart_nov_fin = calcular_variaciones (tab_datos, tab_dic, tab_mensual)
cart_nov_fin.loc['Refinanciamiento'] = (cart_nov_fin.loc['Cartera Refinanciada'] / 
                                        cart_nov_fin.loc['Cartera Bruta'])*100
cart_nov_fin.loc['Reestructuración'] = (cart_nov_fin.loc['Cartera Reestructurada'] / 
                                        cart_nov_fin.loc['Cartera Bruta'])*100

graf_cart_nov_fin = cart_nov_fin.iloc[3:5,0:-11].T


fig = go.Figure()
for column in graf_cart_nov_fin.columns:
    fig.add_trace(go.Scatter(
        x=graf_cart_nov_fin.index,
        y=graf_cart_nov_fin[column],
        mode='lines+markers+text',
        name=column,
        textposition='top center',
        textfont={"family": "Arial", "size": 8, "color": "black"}
    ))
fig.update_layout(
    title='Evolución Refinanciamiento Vs. Reestructuración',
    legend=dict(orientation='h', x = 0.1, y=-0.3),
    margin=dict(b=50))
fig.show()




###### GRÁFICO MOROSIDAD CARTERA REFINANCIADA

cartera = cartera_actual
cart_nov = cart_nov_act
cart_ref_actual = comp_cart_ref(cartera, cart_nov)

cartera = cartera_dic
cart_nov = cart_nov_dic
cart_ref_dic = comp_cart_ref(cartera, cart_nov)

cartera = cartera_mensual
cart_nov = cart_nov_mes
cart_ref_mes = comp_cart_ref(cartera, cart_nov)

tab_datos = cart_ref_actual
tab_dic = cart_ref_dic
tab_mensual = cart_ref_mes

cart_ref_fin = calcular_variaciones (tab_datos, tab_dic, tab_mensual)
cart_ref_fin = cart_ref_fin.drop('Participación', axis =1)


graf_cart_ref_fin = cart_ref_fin.iloc[3:5,0:-10].T

fig = go.Figure()

column_principal = "Cobertura Ref." 
fig.add_trace(go.Scatter(
    x=graf_cart_ref_fin.index,
    y=graf_cart_ref_fin[column_principal],
    mode='lines+markers+text',
    name=column_principal,
    textposition='top center',
    textfont={"family": "Arial", "size": 8, "color": "black"}
))


column_secundario = "Morosidad Ref."
fig.add_trace(go.Scatter(
    x=graf_cart_ref_fin.index,
    y=graf_cart_ref_fin[column_secundario],
    mode='lines+markers+text',
    name=column_secundario,
    textposition='top center',
    textfont={"family": "Arial", "size": 8, "color": "black"},
    yaxis='y2'
))

fig.update_layout(
    title='Morosidad Vs. Cobertura Cartera Refinanciada',
    yaxis_title=f'{column_principal}',
    yaxis2=dict(title=f'{column_secundario}', overlaying='y', side='right'),
    legend=dict(orientation='h', x=0.1, y=-0.3),
    margin=dict(b=50),
    showlegend=True 
)
fig.show()


###### GRÁFICO MOROSIDAD CARTERA REESTRUCTURADA

cartera = cartera_actual
cart_nov = cart_nov_act
cart_res_actual = comp_cart_res(cartera, cart_nov)

cartera = cartera_dic
cart_nov = cart_nov_dic
cart_res_dic = comp_cart_res(cartera, cart_nov)

cartera = cartera_mensual
cart_nov = cart_nov_mes
cart_res_mes = comp_cart_res(cartera, cart_nov)

tab_datos = cart_res_actual
tab_dic = cart_res_dic
tab_mensual = cart_res_mes

cart_res_fin = calcular_variaciones (tab_datos, tab_dic, tab_mensual)
cart_res_fin = cart_res_fin.drop('Participación', axis =1) 

graf_cart_res_fin = cart_res_fin.iloc[3:5,0:-10].T

fig = go.Figure()

column_principal = "Cobertura" 
fig.add_trace(go.Scatter(
    x=graf_cart_res_fin.index,
    y=graf_cart_res_fin[column_principal],
    mode='lines+markers+text',
    name=column_principal,
    textposition='top center',
    textfont={"family": "Arial", "size": 8, "color": "black"}
))


column_secundario = "Mora Reest."
fig.add_trace(go.Scatter(
    x=graf_cart_res_fin.index,
    y=graf_cart_res_fin[column_secundario],
    mode='lines+markers+text',
    name=column_secundario,
    textposition='top center',
    textfont={"family": "Arial", "size": 8, "color": "black"},
    yaxis='y2'
))

fig.update_layout(
    title='Morosidad Vs. Cobertura Cartera Reestructurada',
    yaxis_title=f'{column_principal}',
    yaxis2=dict(title=f'{column_secundario}', overlaying='y', side='right'),
    legend=dict(orientation='h', x=0.1, y=-0.3),
    margin=dict(b=50),
    showlegend=True 
)
fig.show()




############# CARTERA POR TIPO DE CRÉDITO

cartera = cartera_actual
cart_tipo_actual = comp_cart_tipo(cartera)

cartera = cartera_dic
cart_tipo_dic = comp_cart_tipo(cartera)

cartera = cartera_mensual
cart_tipo_mes = comp_cart_tipo(cartera)

###
cart_tipo = cart_tipo_actual
cart_bruta_actual = comp_cart_bruta(cart_tipo)

cart_tipo = cart_tipo_dic
cart_bruta_dic = comp_cart_bruta(cart_tipo)

cart_tipo = cart_tipo_mes
cart_bruta_mes = comp_cart_bruta(cart_tipo)

tab_datos = cart_bruta_actual
tab_dic = cart_bruta_dic
tab_mensual = cart_bruta_mes

cart_bruta_fin = calcular_variaciones (tab_datos, tab_dic, tab_mensual)

graf_cart_bruta_fin = cart_bruta_fin.iloc[:,0:-11].T


####

fig = go.Figure()

for column in graf_cart_bruta_fin.columns:
    fig.add_trace(go.Bar(x=graf_cart_bruta_fin.index, y=graf_cart_bruta_fin[column], name=column))

fig.update_layout(barmode='stack', title='Evolución Cartera por tipo de crédito')

# Mostrar el gráfico
fig.show()