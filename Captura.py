import cv2

classificador = cv2.CascadeClassifier("haarcascade/haarcascade-frontalface-default.xml")
camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostras = 25
id = input('Digite seu identificador: ')
largura, altura = 220, 220
print("Capturando as faces ...")

while (True):
    conectado, frameImagem = camera.read()

    imagemCinza = cv2.cvtColor(frameImagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                     scaleFactor=1.5,
                                                     minSize=(150,150))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(frameImagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
            cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
            print("[Foto " + str(amostra) + " capturada com sucesso]")
            amostra += 1

    cv2.imshow('Face', frameImagem)
    if cv2.waitKey(1) == ord('q'):
        break

    if (amostra >= numeroAmostras + 1):
        break

print("Faces capturadas com sucesso")
camera.release()
cv2.destroyAllWindows()

