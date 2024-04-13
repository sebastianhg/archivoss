import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Cargar la imagen
imagen = cv2.imread('/home/sebastian/Documentos/analisisNETCDF/datosGraficados/2022-01-01.png')

# Obtener el centro de la imagen
centro_imagen = (int(imagen.shape[1] / 2), int(imagen.shape[0] / 2))

# Obtener los objetos del centro de la imagen
objeto_centro = imagen[centro_imagen[1]-1:centro_imagen[1]+2, centro_imagen[0]-1:centro_imagen[0]+2]

# Binarizar los objetos del centro
img_gris = cv2.cvtColor(objeto_centro, cv2.COLOR_BGR2GRAY)
_, objeto_binario = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY_INV)

# Obtener los componentes conectados en el objeto binario
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(objeto_binario, connectivity=8)

# Conectar objetos si hay 4 píxeles en medio del objeto
for label in range(1, num_labels):
    # Obtener las coordenadas del objeto y su centroide
    x, y, w, h, _ = stats[label]
    cX, cY = centroids[label]

    # Verificar si hay 4 píxeles en medio del objeto
    if np.sum(objeto_binario[y+1:y+h-1, x+1:x+w-1]) >= 4*255:
        # Conectar el objeto estableciendo los píxeles en 255
        objeto_binario[y:y+h, x:x+w] = 255

# Encontrar contornos en el objeto binario
contornos, _ = cv2.findContours(objeto_binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar contornos en la imagen original
imagen_contornos = imagen.copy()
cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)

# Mostrar la imagen con los contornos
plt.imshow(cv2.cvtColor(imagen_contornos, cv2.COLOR_BGR2RGB))
plt.title('Contornos')
plt.show()

# Calcular propiedades de los objetos
propiedades_objetos = []
for i, contorno in enumerate(contornos):
    # Calcular el área del objeto
    area = cv2.contourArea(contorno)

    # Obtener el perímetro y las coordenadas del rectángulo que encierra el objeto
    perimetro = cv2.arcLength(contorno, True)
    (x, y, w, h) = cv2.boundingRect(contorno)

    # Calcular la latitud y la longitud del centroide del objeto
    latitud = -60 + (y + h / 2) * (80 / imagen.shape[0])
    longitud = -90 + (x + w / 2) * (110 / imagen.shape[1])

    # Agregar las propiedades del objeto a la lista
    objeto = {
        "id": i + 1,
        "area": area,
        "perimetro": perimetro,
        "coordenadas": (x, y, w, h),
        "centroide": (cX, cY),
        "latitud": latitud,
        "longitud": longitud
    }
    propiedades_objetos.append(objeto)

# Calcular el esqueleto para cada objeto
objetos_esqueletos = []
for contorno in contornos:
    # Obtener el objeto binario correspondiente al contorno
    objeto_binario = np.zeros(imagen.shape[:2], dtype=np.uint8)
    cv2.drawContours(objeto_binario, [contorno], -1, 255, thickness=cv2.FILLED)

    # Esqueletonizar el objeto binario
    objeto_esqueleton = ndimage.skeletonize(objeto_binario)

    # Agregar el objeto esqueleto a la lista
    objetos_esqueletos.append(objeto_esqueleton)

# Mostrar las imágenes resultantes
plt.figure(figsize=(12, 4))

# Mostrar imagen binaria con contornos
plt.subplot(131)
plt.imshow(objeto_binario, cmap='gray')
plt.title('Objeto Binario con Contornos')

# Mostrar imagen con contornos dibujados
plt.subplot(132)
plt.imshow(imagen_contornos)
plt.title('Imagen con Contornos')

# Mostrar imágenes de los esqueletos
plt.subplot(133)
for i, objeto_esqueleton in enumerate(objetos_esqueletos):
    plt.subplot(1, len(objetos_esqueletos), i + 1)
    plt.imshow(objeto_esqueleton, cmap='gray')
    plt.title(f'Esqueleto Objeto {i + 1}')

plt.tight_layout()
plt.show()
