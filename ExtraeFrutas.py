from pdiFun import *


def extraer_frutas(img):
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV', HSV)
    H, S, V = cv2.split(HSV)
    SS = S.copy()
    cv2.imshow('H', H)
    cv2.imshow('S', S)
    cv2.imshow('V', V)
    # Kernel
    kernel = np.ones((5, 5), np.uint8)
    print(f'kernel: {kernel}')

    ##Filtrado medio
    imgMedF = cv2.medianBlur(SS, 55)
    cv2.imshow('Filtrado medio', imgMedF)

    # Binarizado
    imgBin2 = binFun(imgMedF)
    cv2.imshow('Binariza', imgBin2)

    # Aplicar mascara
    mask2 = bwareaopen(imgBin2, 4000)
    cv2.imshow('BwareaOpen', mask2)
    mask_t = erodeFun(kernel, mask2, 2)     # Suaviza el contorno
    cv2.imshow('erodeFun', mask_t)

    img[mask_t == 0] = 0
    cv2.imshow('images salida fun extraer_frutas', img)
    return img
