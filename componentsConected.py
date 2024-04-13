import cv2   # Biblioteca OpenCV para procesamiento de imágenes
import numpy as np   # Biblioteca NumPy para cálculos numéricos
from scipy import ndimage   # Biblioteca SciPy para procesamiento de imágenes

# Cargar la imagen y guardarla en la variable "imagen".
# En este caso, se carga la imagen "/home/sebastian/Documentos/analisisNETCDF/archivos_netcdf/2022-01-01.png".
imagen = cv2.imread("/home/sebastian/Documentos/analisisNETCDF/datosGraficados/2022-01-01.png")

# Convertir la imagen a escala de grises y guardarla en la variable "imgGris".
imgGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar un umbral a la imagen utilizando el método de Otsu y guardar el umbral y la imagen resultante en las variables "umbral" e "imagenMetodo", respectivamente.
umbral, imagenMetodo = cv2.threshold(imgGris, 0, 255, cv2.THRESH_OTSU)

# Crear una máscara binaria a partir de la imagen en escala de grises utilizando NumPy.
# Los píxeles con valor menor que el umbral se convierten en 255 (blanco) y los píxeles con valor mayor o igual que el umbral se convierten en 0 (negro).
mascara = np.uint8((imgGris < umbral) * 255)

# Mostrar la imagen binaria resultante utilizando la función cv2.imshow().
cv2.imshow("objeto (imagen binaria)", mascara)

# Utilizar cv2.connectedComponentsWithStats() para identificar los componentes conectados en la máscara binaria.
# La función devuelve el número de componentes, una matriz de etiquetas de componentes, estadísticas para cada componente y centroides para cada componente.
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara, 4, cv2.CV_32S)

# Calcular el tamaño máximo de los componentes y guardar el resultado en la variable "valor_max_pix".
valor_max_pix = (np.max(stats[:4][1:]))/2

# Número del objeto que se desea llenar
objeto = 29

# Crear una lista de máscaras binarias para cada objeto identificado y sumarlas para obtener una máscara binaria final.
mascaraFinal = np.zeros_like(labels)  # Inicializar la máscara binaria final
for i in range(1, num_labels):
    if i == objeto:
        mascara = i == labels
        mascaraFinal = mascaraFinal + mascara

# Rellenar los agujeros en la máscara binaria final utilizando ndimage.binary_fill_holes() de la biblioteca SciPy y convertirla a un array de enteros.
mascaraFinal = ndimage.binary_fill_holes(mascaraFinal).astype(int)

# Dilatar la máscara "mascaraFinal" utilizando ndimage.binary_dilation() de la biblioteca SciPy y guardar el resultado en la variable "mascarasVecinos".
estructura = np.ones((3, 3), dtype=np.int)  # Estructura para dilatar la máscara
mascarasVecinos = ndimage.binary_dilation(mascaraFinal, structure=estructura)

# Mostrar la imagen con el objeto 29 y los huecos llenos en la máscara "mascarasVecinos".
mascaraRGB = cv2.cvtColor(mascarasVecinos, cv2.COLOR_GRAY2RGB)
imagenFinal = cv2.bitwise_and(imagen, mascaraRGB)
cv2.imshow("objeto con huecos llenos y vecinos dilatados", imagenFinal)

# Esperar a que el usuario presione una tecla para cerrar las ventanas.
cv2.waitKey(0)
cv2.destroyAllWindows()