# Importe de librerias
import numpy as np
import cv2
import math
import pandas as pd

from pdiFun import *
from skimage import filters, measure
import matplotlib.pyplot as plt

azul_min = 0.043873
azul_max = 0.37589
verde_min = 0.064716
verde_max = 0.756334
rojo_min = 0.125059
rojo_max = 1.104814
area_min = 0.003216
area_max = 0.269094
ratio_min = 0.259904
ratio_max = 0.999375
textura_min = 0.036803
textura_max = 0.35895
perimetro_min = 0.0
perimetro_max = 2627


def normalize(value, max, min):
    if min <= value < max:
        value = (value - min) / (max - min)
    elif value < min:
        value = min
    elif value > max:
        value = max
    return value


def fun_normalize(p_azul, p_verde, p_rojo, p_area, axis_ratio, porcentaje_textura, perimetro):
    p_azul = normalize(p_azul, azul_max, azul_min)
    p_verde = normalize(p_verde, verde_max, verde_min)
    p_rojo = normalize(p_rojo, rojo_max, rojo_min)
    p_area = normalize(p_area, area_max, area_min)
    axis_ratio = normalize(axis_ratio, ratio_max, ratio_min)
    porcentaje_textura = normalize(porcentaje_textura, textura_max, textura_min)
    perimetro = normalize(perimetro,perimetro_max,perimetro_min)
    return p_azul, p_verde, p_rojo, p_area, axis_ratio, porcentaje_textura, perimetro


def extraer_caracteristicas(img):
    dataf = pd.DataFrame(columns=['AZUL', 'Verde', 'Rojo', 'Area', 'Ratio', 'Textura','Perimetro'])
    #dataf = pd.DataFrame(columns=['azul', 'verde', 'rojo', 'area', 'ratio', 'textura','perimetro'])

    # # #Filtrado medio
    imgMedF = cv2.medianBlur(img, 55)

    gris = grayFun(imgMedF)  # Escala de grises
    imgBin = binFun(gris)  # binarizada 0. 255

    B, G, R = cv2.split(img)  # Separar Capas
    BB = B.copy()
    BB[imgBin == 0] = 0

    """Texturas (Bordes)"""
    # Canny Edge Detection
    edges = cv2.Canny(image=B, threshold1=20, threshold2=20)  # Canny Edge Detection

    # Canny Edge Detection para el perimetro
    borde = cv2.Canny(image=R, threshold1=100, threshold2=200)  # Canny Edge Detection

    """Tamaño imagen  Marco total de visón"""
    W = B.shape[1]
    H = B.shape[0]
    areaFoto = W * H

    """Tamaño Fruta"""
    Area = (np.sum(imgBin))  # Area de fruta en sumas de 255 por pixel
    areaPixel = Area / 255  # Area de fruta en sumas de 1 por pixel
    pArea = areaPixel / areaFoto  # Area de la fruta respecto al tamaño de la foto

    """Color"""
    azul = np.sum(B)
    verde = np.sum(G)
    rojo = np.sum(R)

    # porcentaje de color respecto a la escala de  0 - 255
    pAzul = azul / Area
    pVerde = verde / Area
    pRojo = rojo / Area

    """Eliminar basuras internas del perimetro"""
    borde = bwareaopen(borde, pArea * 5000)

    """Valor del perimetro en pixeles"""
    perimetro = (borde.sum()) / 255

    """Region props skimage"""
    label_img = measure.label(imgBin)
    regions = measure.regionprops(label_img)
    props = measure.regionprops_table(label_img, properties=('centroid',
                                                             'orientation',
                                                             'major_axis_length',
                                                             'minor_axis_length'))
    props1 = props.copy()
    ## Graficas
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray)
    ax.imshow(borde, cmap=plt.cm.gray)
    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)

        ax.axis((0, 1280, 720, 0))
        plt.show()
    ## End Graficas

    dt = pd.DataFrame(props1).head()
    majorAxis = dt.iloc[0, 3]
    minorAxis = dt.iloc[0, 4]
    textura = edges.sum() / Area  # Pondera el total de texturas en el area de la fruta

    # Relacion de ejes
    axisRatio = minorAxis / majorAxis

    # *********************************************************************

    """Guardar Caracteristicas"""
    pAzul, pVerde, pRojo, pArea, axisRatio, textura, perimetro = fun_normalize(pAzul, pVerde, pRojo, pArea, axisRatio, textura, perimetro)
    dataf.loc[0] = [pAzul, pVerde, pRojo, pArea, axisRatio, textura, perimetro]

    # *********************************************************************
    """Guardar Caracteristicas"""
    #print(f'Axis Ratio: {axisRatio}, Azul: {pAzul}, Rojo: {pRojo}, Verde: {pVerde}, Area: {pArea}, Ratio: {axisRatio}, Textura: {porcentaje_textura}')

    #pAzul, pVerde, pRojo, pArea, axisRatio, porcentaje_textura = fun_normalize(pAzul, pVerde, pRojo, pArea, axisRatio, porcentaje_textura)
    #dataf.loc[0] = [pAzul, pVerde, pRojo, pArea, axisRatio, porcentaje_textura]
    # Estandarizando
    #scaler = preprocessing.StandardScaler()
    #scaler = scaler.fit(dataf[dataf.columns])
    #dataf[dataf.columns] = scaler.transform(dataf[dataf.columns])
    print(f'Axis Ratio: {axisRatio}, Azul: {pAzul}, Rojo: {pRojo}, Verde: {pVerde}, Area: {pArea}, Ratio: {axisRatio}'
          f', Textura: {textura}, Perimetro: {perimetro}')
    return dataf
