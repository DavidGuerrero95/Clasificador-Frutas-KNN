"""
--------------------------------------------------------------------------
------- CLASIFIACIÓN DE FRUTAS -----------------------------------------
------- Coceptos básicos de PDI ------------------------------------------
------- Por: leonardo Fuentes Bohórquez  leoardo.fuentes@udea.edu.co (1) -
--------     David Santiago Guerrero Mertinez  davids.guerrero@udea.edu.co (2)
--------     Estudiantes de ingeniería electrónica UdeA  ----------------
--------     CC 1101076907 (1), CC 1085319765 (2) ------------------------
------- Curso Básico de Procesamiento de Imágenes y Visión Artificial-----
------- 11 Septiembre de 2022 --------------------------------------------
--------------------------------------------------------------------------
"""


'''
--------------------------------------------------------------------------
--1. Importar modulos necesarios ---------------------------------------
--------------------------------------------------------------------------
'''
from pdiFun import *

'''
--------------------------------------------------------------------------
--2. Funcion para extraer la fruta sin el fondo --------------------------
--------------------------------------------------------------------------
'''


def extraer_frutas(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convertir al espacio HSV
    h, s, v = cv2.split(hsv)  # Dividir las capas
    ss = s.copy()  # Copia de la capa de saturación
    cv2.imshow('capa', ss)
    kernel = np.ones((5, 5), np.uint8)  # Kernel

    img_med_f = cv2.medianBlur(ss, 55)  # Filtrado medio

    img_bin2 = binFun(img_med_f)  # Binarizado

    mask2 = bwareaopen(img_bin2, 4000)  # Aplicar mascara
    mask_t = erodeFun(kernel, mask2, 2)  # Suaviza el contorno

    img[mask_t == 0] = 0  # Eliminar fondo de la imagen
    cv2.imshow('recorte', img)
    return img  # retorna la imagen sin fondo y suavizada
