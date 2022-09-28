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


def tomar_foto():
  global image  ## Imagenes para mostrar en la interfaz
  global contorno
  global textura
  ret, img2 = cam.read()
  # Leer la imagen de entrada y la redimensionamos
  image = img2.copy()
  image = imutils.resize(image, height=380)
  # Para visualizar la imagen de entrada en la GUI
  imageToShow= imutils.resize(image, width=180)
  imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGR2RGB)
  im = Image.fromarray(imageToShow )
  img = ImageTk.PhotoImage(image=im)
  lblInputImage.configure(image=img)
  lblInputImage.image = img
  # Label IMAGEN DE ENTRADA
  lblInfo1 = Label(root, text="IMAGEN DE ENTRADA:",font='bold')
  lblInfo1.grid(column=0, row=0, padx=5, pady=5)


  img3 = cv2.GaussianBlur(img2, (7, 7), 0)  ## Filtro Gaussiano
  foto = img3.copy()  # Sacar copia de la imagen
  foto = extraer_frutas(foto)  # Extrae foto binarizada
  foto2 = foto.copy()  # Sacar copia de la imagen
  caracteristicas,textura,contorno = extraer_caracteristicas(foto2)  # Extraer Dataset Con las caracteristicas listo para clasificar
  #cv2.imshow('recorte', edges)
  imageToShowOutput = cv2.cvtColor(contorno, cv2.COLOR_BGR2RGB)
  # Para visualizar la imagen en lblOutputImage en la GUI
  contorno = imutils.resize(imageToShowOutput, height=100)
  im2 = Image.fromarray(contorno)
  img2 = ImageTk.PhotoImage(image=im2)
  lblOutputImage.configure(image=img2)
  lblOutputImage.image = img2

  imageToShowOutput = cv2.cvtColor(textura, cv2.COLOR_BGR2RGB)
  # Para visualizar la imagen en lblOutputImage en la GUI
  textura = imutils.resize(imageToShowOutput, height=100)
  im3 = Image.fromarray(textura)
  img3 = ImageTk.PhotoImage(image=im3)
  lblOutputImage2.configure(image=img3)
  lblOutputImage2.image = img3


  #Binarizada
  imageToShowOutput = cv2.cvtColor(foto, cv2.COLOR_BGR2RGB)
  # Para visualizar la imagen en lblOutputImage en la GUI
  textura = imutils.resize(imageToShowOutput, height=100)
  im4 = Image.fromarray(textura)
  img4 = ImageTk.PhotoImage(image=im4)
  lblOutputImage3.configure(image=img4)
  lblOutputImage3.image = img4



  # Label resultados
  lblInfo3 = Label(root, text="IMAGENES DEL PROCESO IMPLEMENTADO", font="bold")
  lblInfo3.grid(column=1, row=0, padx=5, pady=5,columnspan=3)


  #label textura
  lblInfo3 = Label(root, text="Contorno")
  lblInfo3.grid(column=1, row=1, padx=5, pady=5)

  # label Contorno
  lblInfo3 = Label(root, text="Textura")
  lblInfo3.grid(column=2, row=1, padx=5, pady=5)

  # label Binarizada
  lblInfo3 = Label(root, text="Selección")
  lblInfo3.grid(column=1, row=3, padx=5, pady=5)

  fruit_num = model.predict(caracteristicas)  # Clasificacion de la fruta
  print(f'La fruta es: {translate(fruit_num)}')  # Que fruta fue clasificada

  Resultado.configure(text=f'La fruta es {translate(fruit_num)}')






root = Tk()

root.title('Detector de Frutas')
#Creamos el marco
root.resizable(1,1)
root.geometry('550x400')


#Boton
boton1 = Button(root, text="Tomar Foto",command=tomar_foto).grid(row=4, column=0)

lblInputImage = Label(root)
lblInputImage.grid(column=0, row=2)
# Label donde se presentará la imagen de salida
lblOutputImage = Label(root)
lblOutputImage.grid(column=1, row=2)
#imagen de la textura
lblOutputImage2 = Label(root)
lblOutputImage2.grid(column=2, row=2)

#imagen de la binarizada
lblOutputImage3 = Label(root)
lblOutputImage3.grid(column=1, row=4)

Resultado = Label(root, text=f'¿Que fruta es?', font= 'bold')
Resultado.grid(row=3, column=0)
root.mainloop()
