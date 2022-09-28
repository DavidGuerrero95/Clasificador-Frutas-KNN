"""
--------------------------------------------------------------------------
------- CLASIFIACIÓN DE FRUTAS -----------------------------------------
------- Coceptos básicos de PDI ------------------------------------------
------- Por: leonardo Fuentes Bohórquez  leoardo.fuentes@udea.edu.co (1) -
--------     David Santiago Guerrero Mertinez  davids.guerrero@udea.edu.co (2)
--------     Estudiantes de ingeniería electrónica UdeA  ----------------
--------     CC 1101076907 (1), CC 1085319765 (2) ------------------------
------- Curso Básico de Procesamiento de Imágenes y Visión Artificial-----
------- 11 Septiembre de 2022 -------------------------------------------------
--------------------------------------------------------------------------
"""

'''
--------------------------------------------------------------------------
--1. Importar librerias y modulos necesarios -----------------------------
--------------------------------------------------------------------------
'''
import math
import pandas as pd
from pdiFun import *
from skimage import filters, measure
import matplotlib.pyplot as plt

'''
--------------------------------------------------------------------------
--2. Inicializar valores iniciales para la normalizacion del dataset -----
--------------------------------------------------------------------------
'''
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

'''
--------------------------------------------------------------------------
--3. Funcion que normaliza un valor --------------------------------------
--------------------------------------------------------------------------
'''


def normalize(value, max, min):
    if min <= value < max:
        value = (value - min) / (max - min)
    elif value < min:
        value = min
    elif value > max:
        value = max
    return value


'''
--------------------------------------------------------------------------
--3. Funcion que normaliza todos los valores -----------------------------
--------------------------------------------------------------------------
'''


def fun_normalize(p_azul, p_verde, p_rojo, p_area, axis_ratio, porcentaje_textura, perimetro):
    p_azul = normalize(p_azul, azul_max, azul_min)
    p_verde = normalize(p_verde, verde_max, verde_min)
    p_rojo = normalize(p_rojo, rojo_max, rojo_min)
    p_area = normalize(p_area, area_max, area_min)
    axis_ratio = normalize(axis_ratio, ratio_max, ratio_min)
    porcentaje_textura = normalize(porcentaje_textura, textura_max, textura_min)
    perimetro = normalize(perimetro, perimetro_max, perimetro_min)
    return p_azul, p_verde, p_rojo, p_area, axis_ratio, porcentaje_textura, perimetro


'''
--------------------------------------------------------------------------
--4. Funcion extrae las caracteristicas en un dataset --------------------
--------------------------------------------------------------------------
'''


def extraer_caracteristicas(img):
    # Inicializacion del dataframe
    dataf = pd.DataFrame(columns=['AZUL', 'Verde', 'Rojo', 'Area', 'Ratio', 'Textura', 'Perimetro'])
    # dataf = pd.DataFrame(columns=['azul', 'verde', 'rojo', 'area', 'ratio', 'textura','perimetro'])

    img_med_f = cv2.medianBlur(img, 55)  # Filtrado medio
    # cv2.imshow('Filtrado medio', img_med_f)
    gris = grayFun(img_med_f)  # Escala de grises
    img_bin = binFun(gris)  # binarizada 0 - 255
    b, g, r = cv2.split(img)  # Separar
    # cv2.imshow('blue', b)
    # cv2.imshow('gren', g)
    # cv2.imshow('red', r)

    bb = b.copy()
    bb[img_bin == 0] = 0

    '''
    --5. Texturas (Bordes) ---------------------------------------------------
    '''
    edges = cv2.Canny(image=b, threshold1=20, threshold2=20)  # Canny Edge Detection
    # cv2.imshow('Textura', edges)

    # Canny Edge Detection para el perimetro
    borde = cv2.Canny(image=r, threshold1=100, threshold2=200)  # Canny Edge Detection
    # cv2.imshow('borde', borde)
    '''
    --6. Tamaño imagen  Marco total de visón ---------------------------------
    '''
    w = b.shape[1]
    h = b.shape[0]
    area_foto = w * h

    '''
    --7. Tamaño fruta --------------------------------------------------------
    '''
    area = (np.sum(img_bin))  # Area de fruta en sumas de 255 por pixel
    area_pixel = area / 255  # Area de fruta en sumas de 1 por pixel
    p_area = area_pixel / area_foto  # Area de la fruta respecto al tamaño de la foto

    '''
    --8. Color ---------------------------------------------------------------
    '''
    azul = np.sum(b)
    verde = np.sum(g)
    rojo = np.sum(r)

    '''
    --9. porcentaje de color respecto a la escala de  0 - 255 ----------------
    '''
    p_azul = azul / area
    p_verde = verde / area
    p_rojo = rojo / area

    '''
    --10. Eliminar basuras internas del perimetro ----------------------------
    '''
    borde = bwareaopen(borde, p_area * 5000)

    '''
    --11. Valor del perimetro en pixeles -------------------------------------
    '''

    perimetro = (borde.sum()) / 255

    '''
    --12. Region props skimage -------------------------------------
    '''
    label_img = measure.label(img_bin)
    regions = measure.regionprops(label_img)
    props = measure.regionprops_table(label_img, properties=('centroid',
                                                             'orientation',
                                                             'major_axis_length',
                                                             'minor_axis_length'))
    props1 = props.copy()
    # Graficas
    # fig, ax = plt.subplots()
    # ax.imshow(img, cmap=plt.cm.gray)
    # ax.imshow(borde, cmap=plt.cm.gray)
    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

        # ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        # ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        # ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ##ax.plot(bx, by, '-b', linewidth=2.5)

        ##ax.axis((0, 1280, 720, 0))
        ##plt.show()
    # End Graficas

    dt = pd.DataFrame(props1).head()
    major_axis = dt.iloc[0, 3]
    minor_axis = dt.iloc[0, 4]
    textura = edges.sum() / area  # Pondera el total de texturas en el area de la fruta

    # Relacion de ejes
    axis_ratio = minor_axis / major_axis

    '''
    --13. Normalizar Caracteristicas -------------------------------------
    '''
    p_azul, p_verde, p_rojo, p_area, axis_ratio, textura, perimetro = fun_normalize(p_azul, p_verde, p_rojo, p_area,
                                                                                    axis_ratio,
                                                                                    textura, perimetro)
    '''
    --14. Almacema Caracteristicas -------------------------------------
    '''
    dataf.loc[0] = [p_azul, p_verde, p_rojo, p_area, axis_ratio, textura, perimetro]

    print(
        f'Axis Ratio: {axis_ratio}, Azul: {p_azul}, Rojo: {p_rojo}, Verde: {p_verde}, Area: {p_area}, Ratio: {axis_ratio}'
        f', Textura: {textura}, Perimetro: {perimetro}')  # Imprime caracteristicas
    print(dataf)

    return dataf, edges, borde  # retorna dataset con las caracteristicas extraidas
