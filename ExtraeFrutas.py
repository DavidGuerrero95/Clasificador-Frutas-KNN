from pdiFun import *

def extraer_frutas(img):
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    SS = S.copy()

    # Kernel
    kernel = np.ones((5, 5), np.uint8)

    ##Filtrado medio
    imgMedF = cv2.medianBlur(SS, 55)

    # Binarizado
    imgBin2 = binFun(imgMedF)

    # Aplicar mascara
    mask2 = bwareaopen(imgBin2, 4000)
    mask_t = erodeFun(kernel, mask2, 2)

    img[mask_t == 0] = 0
    return img