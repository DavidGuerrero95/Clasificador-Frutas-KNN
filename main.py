import pickle
from ExtraeFrutas import *
from ExtraeCaracteristicas import *

# Lectura de la cámara:
cam = cv2.VideoCapture(0)


# Eigir lectura a resolusión de 720 HD
def make_720p():
    cam.set(3, 1290)  # Ancho en pixeles
    cam.set(4, 720)  # Alto  en pixeles


make_720p()
model = pickle.load(open('model_KNN_21_text_perime.pkl', 'rb'))

while True:
    ret, img = cam.read()
    img = cv2.GaussianBlur(img, (7, 7), 0)

    cv2.imshow('cam', img)

    if cv2.waitKey(1) == 27:  # Salida con Esc
        break

    if cv2.waitKey(0) == 99: # C para capturar
        foto = img.copy()
        foto = extraer_frutas(foto)  # Extrae foto binarizada
        # cv2.imshow('FRUTA BIEN CHIMBITA',foto)
        foto2 = foto.copy()
        caracteristicas = extraer_caracteristicas(foto2)
        fruit_num = model.predict(caracteristicas)
        print(translate(fruit_num))

cv2.destroyAllWindows()
