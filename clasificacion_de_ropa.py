
import tensorflow as ts
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import random as rd
import numpy as np

datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

datos_entrenamiento, datos_pruebas = datos['train'], datos['test']

nombres_clases = metadatos.features['label'].names

def normalizar(imagenes, etiquetas ):
  imagenes = ts.cast(imagenes, ts.float32)
  imagenes /= 255
  return imagenes, etiquetas

  datos_de_entrenamiento = datos_entrenamiento.map(normalizar).cache()
  datos_de_pruebas = datos_pruebas.map(normalizar).cache()

  datos_de_entrenamiento = datos_de_entrenamiento.cache()
  datos_de_pruebas = datos_de_pruebas.cache()

modelo = ts.keras.Sequential([
    ts.keras.layers.Flatten(input_shape=(28,28,1)),
    ts.keras.layers.Dense(50, activation=ts.nn.relu),
    ts.keras.layers.Dense(50, activation=ts.nn.relu),
    ts.keras.layers.Dense(10, activation=ts.nn.softmax)
  ])

modelo.compile(
    optimizer='adam',
    loss=ts.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

num_ej_entrenamiento = metadatos.splits["train"].num_examples
num_ej_pruebas = metadatos.splits["test"].num_examples

TAMAÑO_LOTE = 32
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMAÑO_LOTE)
datos_de_pruebas = datos_pruebas.batch(TAMAÑO_LOTE)

import math
historial = modelo.fit(datos_entrenamiento,epochs=5, steps_per_epoch=math.ceil(num_ej_entrenamiento/TAMAÑO_LOTE))

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

for batch_images, batch_labels in datos_de_pruebas.take(1):
    break


batch_size = batch_images.shape[0]

index_aleatorio = rd.randrange(batch_size)


imagen_aleatoria = batch_images[index_aleatorio].numpy().squeeze()

plt.figure()
plt.imshow(imagen_aleatoria, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

