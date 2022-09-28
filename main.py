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
--1. Importar librerias y modulos necesarios -----------------------------
--------------------------------------------------------------------------
'''
import pickle
import imutils
from ExtraeCaracteristicas import *
from ExtraeFrutas import *
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
'''
--------------------------------------------------------------------------
--2. Lectura de la cámara ------------------------------------------------
--------------------------------------------------------------------------
'''
cam = cv2.VideoCapture(0)

'''
--------------------------------------------------------------------------
--3. Elegir lectura a resolusión de 720 HD -------------------------------
--------------------------------------------------------------------------
'''


def tomar_foto(img):
    global image
    # Leer la imagen de entrada y la redimensionamos
    image = img.copy()
    image = imutils.resize(image, height=380)
    # Para visualizar la imagen de entrada en la GUI
    imageToShow = imutils.resize(image, width=180)
    im = Image.fromarray(imageToShow)
    img = ImageTk.PhotoImage(image=im)
    lblInputImage.configure(image=img)
    lblInputImage.image = img
    # Label IMAGEN DE ENTRADA
    lblInfo1 = Label(root, text="IMAGEN DE ENTRADA:")
    lblInfo1.grid(column=0, row=1, padx=5, pady=5)
    # Al momento que leemos la imagen de entrada, vaciamos
    # la iamgen de salida y se limpia la selección de los
    # radiobutton
    lblOutputImage.image = ""
    selected.set(0)


def make_720p():
    cam.set(3, 1290)  # Ancho en pixeles
    cam.set(4, 720)  # Alto  en pixeles


make_720p()

'''
--------------------------------------------------------------------------
--4. Cargar modelo KNN ya entrenado para la clasificación ----------------
--------------------------------------------------------------------------
'''
model = pickle.load(open('model_KNN_21_text_perime.pkl', 'rb'))
'''
--------------------------------------------------------------------------
--5. Inicializar ciclo para la ejecución continua del programa -----------
--------------------------------------------------------------------------
'''
while True:
    # Capturar imagen
    ret, img = cam.read()

    # Aplicar filtro suavizado Gaussiano
    img = cv2.GaussianBlur(img, (7, 7), 0)
    cv2.imshow('cam', img)
    # Salir presionando ESC
    if cv2.waitKey(1) == 27:
        break

    '''
    --------------------------------------------------------------------------
    --6. Presionar C para capturar imagen y clasificarla ---------------------
    --------------------------------------------------------------------------
    '''
    if cv2.waitKey(0) == 99:
        foto = img.copy()  # Sacar copia de la imagen
        foto = extraer_frutas(foto)  # Extrae foto binarizada
        foto2 = foto.copy()  # Sacar copia de la imagen
        caracteristicas = extraer_caracteristicas(
            foto2)  # Extraer Dataset Con las caracteristicas listo para clasificar
        fruit_num = model.predict(caracteristicas)  # Clasificacion de la fruta
        print(f'La fruta es: {translate(fruit_num)}')  # Que fruta fue clasificada

'''
--------------------------------------------------------------------------
--7. Destruir todas las ventanas -----------------------------------------
--------------------------------------------------------------------------
'''
cv2.destroyAllWindows()
