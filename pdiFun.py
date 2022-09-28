"""
--------------------------------------------------------------------------
------- CLASIFIACIÓN DE FRUTAS -----------------------------------------
------- Coceptos básicos de PDI ------------------------------------------
------- Por: leonardo Fuentes Bohórquez  leoardo.fuentes@udea.edu.co (1) -
-------------David Santiago Guerrero Mertinez  davids.guerrero@udea.edu.co (2)
-------------Estudiantes de ingeniería electrónica UdeA  ----------------
-------------CC 1101076907 (1), CC 1085319765 (2) ------------------------
------- Curso Básico de Procesamiento de Imágenes y Visión Artificial-----
------- 11 Septiembre de 2022 -------------------------------------------------
--------------------------------------------------------------------------
"""
import numpy as np
import cv2
import copy
from scipy import ndimage as ndi
from skimage.feature import match_template

""" 
*******************************************************************************
--------- 1.Funciones ---------------------------------------------------------
*******************************************************************************
"""

"""Fun algoritmo bwareaopen tomado de 
#       https://stackoverflow.com/questions/2348365/matlab-bwareaopen-equivalent-function-in-opencv
"""


def bwareaopen(img, min_size, connectivity=8):
    """Remove small objects from binary image (approximation of 
    bwareaopen in Matlab for 2D images).
    
    Args:
        img: a binary image (dtype=uint8) to remove small objects from
        min_size: minimum size (in pixels) for an object to remain in the image
        connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).
    
    Returns:
        the binary image with small objects removed
    """

    # Find all connected components (called here "labels")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=connectivity)

    # check size of all connected components (area in pixels)
    for i in range(num_labels):
        label_size = stats[i, cv2.CC_STAT_AREA]

        # remove connected components smaller than min_size
        if label_size < min_size:
            img[labels == i] = 0

    return img


"""Función para pasar a escala de Grises"""


def grayFun(a):
    b = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    return b


"""Función para Binarizar con Othsu"""


def binFun(a):
    ret, binary = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


"""Funcion para Aplicar Erosion"""


def erodeFun(kernel, img, i):
    erode = cv2.erode(img, kernel, iterations=i)
    return erode


"Funcion clase en dataframe para frutas"


def clase_fruta(index):
    if (index >= 0 and index <= 15):
        clase = 'Banano'
    elif (index > 15 and index <= 26):
        clase = 'Tomate'
    elif (index > 26 and index <= 37):
        clase = 'Manzana'
    elif (index > 37 and index <= 48):
        clase = 'Pera'
    elif (index > 48 and index <= 68):
        clase = 'Limon'
    elif (index > 68 and index <= 89):
        clase = 'Uchuva'
    elif (index > 89 and index <= 109):
        clase = 'Uva'
    elif (index > 109):
        clase = 'Fresa'
    return clase


def translate(lista):
    fruta = ''
    for i in lista:
        if i == 0:
            fruta = 'aguacate'
        if i == 1:
            fruta = 'banano'
        if i == 2:
            fruta = 'fresa'
        if i == 3:
            fruta = 'limon'
        if i == 4:
            fruta = 'mango'
        if i == 5:
            fruta = 'manzana'
        if i == 6:
            fruta = 'pera'
        if i == 7:
            fruta = 'tomate'
        if i == 8:
            fruta = 'uchuva'
        if i == 9:
            fruta = 'uva'
    return fruta
