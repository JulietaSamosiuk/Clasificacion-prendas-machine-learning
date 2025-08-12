# Trabajo Práctico 2 - Laboratorio de Datos (1er Cuatrimestre 2025)

# Nombre de Grupo: Datos de Labo

# Integrantes:
# - Denisse Britez
# - Julieta Samosiuk
# - Lautaro Alvarez Bertoya

# Contenido:
# - Análisis exploratorio
# - Clasificación binaria con kNN
# - Clasificación multiclase con árboles de decisión

#%%
# === IMPORTS ===

# Importamos las bibliotecas necesarias para análisis de datos, visualización y machine learning.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics

#%%
# === CARGA DE DATOS ===

# Cargamos el dataset Fashion-MNIST desde el archivo CSV, usando el índice como identificador.
datos = pd.read_csv("Fashion-MNIST.csv", index_col=0)

#%%
# === FUNCIONES AUXILIARES ===

# Definimos una función para visualizar una grilla de imágenes del dataset.
def imprimirGrillaDeFotos(data, rows, cols, grupo, label_names):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))
    fig.suptitle("Fashion-MNIST - Grilla de imágenes", fontsize=16)
    for i, ax in enumerate(axes.flat):
        img = np.array(data.iloc[i+grupo*25, :-1]).reshape(28, 28)
        label = data.iloc[i+grupo*25, -1]
        ax.imshow(img, cmap="Reds")
        ax.set_title(f"Label: {label_names[label]}")
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

#%%
# === EJERCICIO 1: ANÁLISIS EXPLORATORIO DEL DATASET ===

# Analizamos la cantidad de imágenes disponibles.
print("Cantidad de imágenes:", datos.shape[0])

# Calculamos la cantidad de atributos por imagen (28x28 = 784) y mostramos los tipos de datos.
n_columnas = datos.shape[1]
n_atributos = n_columnas - 1
print(f"Cantidad de atributos por imagen: {n_atributos}")
print("\nTipos de atributos:")
print(datos.dtypes.value_counts())

# Revisamos cuántas clases hay en el dataset y qué etiquetas contiene.
n_clases = datos["label"].nunique()
print(f"Cantidad de clases de prendas: {n_clases}")
print(f"\nClases presentes: {sorted(datos['label'].unique())}")

# 1.a. Análisis de píxeles relevantes.

# Analizamos qué píxeles tienen mayor o menor valor promedio y variabilidad, para identificar regiones que podrían ser más útiles para clasificar las prendas.

# Calculamos el promedio y el desvío estándar de cada píxel para identificar cuáles son más activos y variables.
imagenes = datos.drop("label", axis=1)
promedio_pixeles = imagenes.mean()
desvio_pixeles = imagenes.std()

# Visualizamos los resultados.
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(promedio_pixeles.values.reshape(28, 28), cmap="gray")
axs[0].set_title("Promedio de activación por píxel")
axs[0].axis("off")

axs[1].imshow(desvio_pixeles.values.reshape(28, 28), cmap="gray")
axs[1].set_title("Variabilidad por píxel")
axs[1].axis("off")

plt.tight_layout()
plt.show()

# 1.b. Similitud entre clases.

# Observamos visualmente ejemplos de diferentes clases para identificar cuáles se parecen más entre sí. En este caso comparamos pullover, pantalón y camisa.

# Seleccionamos ejemplos de tres clases: Pullover (2), Pantalón (1) y Camisa (6) para comparar visualmente.
# Diccionario de Clases
clases_dict = {
    0: "Camiseta/Top",
    1: "Pantalón",
    2: "Pullover",
    3: "Vestido",
    4: "Abrigo",
    5: "Sandalia", 
    6: "Camisa",
    7: "Zapatilla deportiva",
    8: "Bolsa",
    9: "Botín"
}

fig, axes = plt.subplots(3, 5, figsize=(10, 5))
labels = [2, 1, 6]
for i, label in enumerate(labels):
    muestras = datos[datos["label"] == label].drop("label", axis=1).sample(5, random_state=42 + label)
    for j in range(5):
        ax = axes[i, j]
        ax.imshow(muestras.iloc[j].values.reshape(28, 28), cmap="gray")
        ax.axis("off")
        if j == 0:
            ax.set_ylabel(clases_dict[label], fontsize=12)

plt.tight_layout()
plt.show()

# 1.c. Variabilidad intra-clase para la clase 8 (Bolsa).

# Exploramos la diversidad de imágenes dentro de una misma clase para ver si hay mucha o poca variación visual.

# Observamos 10 ejemplos aleatorios de la clase 8 para evaluar si hay variabilidad entre las imágenes.
bolsas = datos[datos["label"] == 8].drop("label", axis=1)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, (_, row) in zip(axes.flatten(), bolsas.sample(10, random_state=8).iterrows()):
    ax.imshow(row.values.reshape(28, 28), cmap="gray")
    ax.axis("off")

plt.show()

#%%
# === EJERCICIO 2: CLASIFICACIÓN BINARIA CON KNN (CLASE 0 VS CLASE 8) ===

# 2.a. Construir un nuevo DataFrame que contenga solo al subconjunto de imágenes correspondientes a las clases 0 y 8.

# Filtramos el dataset original para quedarnos únicamente con las clases 0 (Camiseta/Top) y 8 (Bolsa), con el objetivo de entrenar un clasificador binario.

grupos = [0, 8]
data_df_copy = pd.read_csv("Fashion-MNIST.csv", index_col=0)
data_limpia = []

for i, row in data_df_copy.iterrows():
    if row['label'] in grupos:
        data_limpia.append(row)

grupos_0_y_8 = pd.DataFrame(data_limpia)

conteo_0_y_8 = grupos_0_y_8.groupby('label').size()
print("Cantidad de imágenes por clase seleccionada:")
for label, cantidad in conteo_0_y_8.items():
    print(f"Clase {label}: {cantidad} imágenes")

# Visualizamos ejemplos de las clases 0 y 8 para observar diferencias visuales.
imprimirGrillaDeFotos(grupos_0_y_8, rows=3, cols=4, grupo=2, label_names=clases_dict)

"""
Aclaracion: Imprimimos la grilla de imágenes nuevamente, pero con los grupos 0 y 8 únicamente,
para ver qué píxeles serían útiles para analizar en este preciso caso.

Vemos a ojo que podemos tomar algunos píxeles de las primeras filas para corroborar,
ya que los bolsos no suelen ocupar píxeles en estas filas, o pixeles de los costados
de las remeras, ya que son píxeles con informacion importante en los bolsos.

Para no hacer un analisis a ojo, decidimos realizar un scan de los píxeles mas importantes
y dejarlos en lista (Ordenados por importancia) para luego agruparlos.
"""

# 2.b. Separar los datos en conjuntos de entrenamiento y testeo.

# Dividimos el dataset en entrenamiento (70%) y testeo (30%) de forma estratificada, para evaluar el modelo de manera justa.

X = grupos_0_y_8.drop("label", axis=1)
Y = grupos_0_y_8["label"]

X_train_total, X_test_total, Y_train_total, Y_test_total = train_test_split(
    X, Y, test_size=0.3, stratify=Y
)

# 2.c. Ajustar un modelo de kNN utilizando una cantidad reducida de atributos.

# Identificamos los píxeles que mejor diferencian entre clases 0 y 8, y entrenamos un modelo de kNN utilizando solo esos atributos seleccionados.

# Separamos las clases 0 y 8 en dos datasets separados.
imagenes_0 = grupos_0_y_8[grupos_0_y_8["label"] == 0].drop("label", axis=1)
imagenes_8 = grupos_0_y_8[grupos_0_y_8["label"] == 8].drop("label", axis=1)

# Calculamos el promedio por píxel para cada clase y analizamos la diferencia entre las clases.
promedio_0 = imagenes_0.mean().values.reshape(28, 28)
promedio_8 = imagenes_8.mean().values.reshape(28, 28)

diferencia = np.abs(promedio_0 - promedio_8)

# Visualizamos los promedios y diferencias por clase.
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Mapa de promedios del grupo 0.
axs[0].imshow(promedio_0, cmap="gray")
axs[0].set_title("Promedio clase 0 (Camiseta/Top)")
axs[0].axis("off")

# Mapa de promedios del grupo 8.
axs[1].imshow(promedio_8, cmap="gray")
axs[1].set_title("Promedio clase 8 (Bolsa)")
axs[1].axis("off")

# Mapa de calor.
axs[2].imshow(diferencia, cmap="Reds")
axs[2].set_title("Diferencia de promedios por píxel")
axs[2].axis("off")

plt.tight_layout()
plt.show()

# 2.d. Comparar modelos de kNN con diferentes grupos de píxeles.

# Probamos múltiples grupos de píxeles y valores de k para evaluar el impacto en la precisión, y así seleccionar la mejor configuración del modelo.

# Seleccionamos los píxeles más distintos entre clases y generamos grupos.
# Los agrupamos en una lista ordenados de mayor importancia a menor importancia (con el "[::-1]").
diferencia_flat = diferencia.flatten()
indices_ordenados = np.argsort(diferencia_flat)[::-1]

# Elegimos las cantidades de píxeles que tendrán los grupos y la cantidad de subgrupos que se entrenarán.
cantidades_pixeles = [2, 4, 6]
cantidad_pruebas_por_cantidad_pxs = 3
mejores_pixeles = []

# Con el for generamos los grupos de píxeles (tanto de 2, como de 4 o 6 pixeles).
for cant in cantidades_pixeles:
    for i in range(cantidad_pruebas_por_cantidad_pxs):
        start = i * cant
        end = start + cant
        grupo = [f"pixel{idx}" for idx in indices_ordenados[start:end]]
        mejores_pixeles.append(grupo)

# Visualizamos los grupos de píxeles seleccionados en forma de grilla y ubicandolos en el lugar de la imagen en donde se encontrarían.
filas = 3
columnas = int(np.ceil(len(mejores_pixeles) / filas))
fig, axes = plt.subplots(nrows=filas, ncols=columnas, figsize=(4 * columnas, 8))
fig.suptitle("Visualización de píxeles seleccionados por grupo", fontsize=16)

for i, grupo in enumerate(mejores_pixeles):
    mascara = np.zeros(784)
    for px in grupo:
        idx = int(px.replace("pixel", ""))
        mascara[idx] = 1

    ax = axes.flat[i]
    ax.imshow(mascara.reshape(28, 28), cmap="Reds")
    ax.set_title(f"Grupo {i+1}")
    ax.axis("off")

plt.tight_layout()
plt.subplots_adjust(top=0.8)
plt.show()

# Evaluamos cada grupo utilizando kNN con k=10.
accuracies_px = []
labels_px = []

for i, pixeles in enumerate(mejores_pixeles):
    X_train = X_train_total[pixeles]
    X_test = X_test_total[pixeles]

    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(X_train, Y_train_total)
    Y_pred = model.predict(X_test)

    acc = metrics.accuracy_score(Y_test_total, Y_pred)
    accuracies_px.append(acc)
    labels_px.append(f"Grupo pxs {i+1} ({len(pixeles)} px)")

# Mostramos la precisión obtenida por grupo.
plt.figure(figsize=(10, 5))
bars = plt.bar(labels_px, accuracies_px, color=["aqua" if "2" in l else ("skyblue" if "4" in l else "salmon") for l in labels_px])
plt.xticks(rotation=45)
plt.ylabel("Exactitud (accuracy)")
plt.ylim(0.88, 0.94)
plt.title("Comparación de precisión por grupo de píxeles")
plt.tight_layout()
plt.show()

# Comparamos diferentes valores de k para los mejores grupos.
promedio_exactitud_pxs = []

for i in range(len(cantidades_pixeles)):
    inicio = i * cantidad_pruebas_por_cantidad_pxs
    fin = inicio + cantidad_pruebas_por_cantidad_pxs
    promedio = np.mean(accuracies_px[inicio:fin])
    promedio_exactitud_pxs.append(promedio)

# Seleccionamos el grupo con la cantidad de píxeles con mejor accuracy (2, 4 o 6).
indice_mejor_promedio_pxs = np.argmax(promedio_exactitud_pxs)
inicio = indice_mejor_promedio_pxs * cantidad_pruebas_por_cantidad_pxs
fin = inicio + cantidad_pruebas_por_cantidad_pxs
grupos_pixeles = mejores_pixeles[inicio:fin]

# Elegimos estudiar los k en el rango 2 a 15 (salteamos el 1 porque no es muy relevante para comparar y también para intentar amortizar algo el tiempo de la consulta).
valores_k = range(2, 15)
resultados_test_por_grupo = []
resultados_train_por_grupo = []

for i, pixeles in enumerate(grupos_pixeles):
    acc_test = []
    acc_train = []

    X_train = X_train_total[pixeles]
    X_test = X_test_total[pixeles]

    for k in valores_k:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, Y_train_total)
        Y_pred = model.predict(X_test)
        Y_pred_train = model.predict(X_train)

        acc_test.append(metrics.accuracy_score(Y_test_total, Y_pred))
        acc_train.append(metrics.accuracy_score(Y_train_total, Y_pred_train))

    resultados_test_por_grupo.append(acc_test)
    resultados_train_por_grupo.append(acc_train)

# Graficamos la evolución de la precisión de los datos de train para cada grupo según k. 
plt.figure(figsize=(10, 6))
for i, acc_train in enumerate(resultados_train_por_grupo):
    plt.plot(valores_k, acc_train, label=f"Grupo {i+1}")
plt.title(f"Exactitud en train según valor de k (grupo de {cantidades_pixeles[indice_mejor_promedio_pxs]} pxs)")
plt.xlabel("Cantidad de vecinos (k)")
plt.ylabel("Exactitud (accuracy)")
plt.legend()
plt.grid(True)
plt.show()

# Graficamos la evolución de la precisión de los datos de test para cada grupo según k. 
plt.figure(figsize=(10, 6))
for i, acc_test in enumerate(resultados_test_por_grupo):
    plt.plot(valores_k, acc_test, label=f"Grupo {i+1}")
plt.title(f"Exactitud en test según valor de k (grupo de {cantidades_pixeles[indice_mejor_promedio_pxs]} pxs)")
plt.xlabel("Cantidad de vecinos (k)")
plt.ylabel("Exactitud (accuracy)")
plt.legend()
plt.grid(True)
plt.show()

# Para mostrar el mejor grupo, debemos hacer un promedio para cada grupo de píxeles de los valores según k.
# Identificamos el grupo con mejor promedio entre train y test.
promedios_exactitud_por_k = []
for i in range(len(resultados_train_por_grupo)):
    promedio = (np.mean(resultados_test_por_grupo[i]) + np.mean(resultados_train_por_grupo[i])) / 2
    promedios_exactitud_por_k.append(promedio)

# Con el índice obtenido, seleccionamos el grupo de píxeles de cada grupo (train y test).
indice_mejor_promedio_pxs_segun_k = np.argmax(promedios_exactitud_por_k)
mejores_promedios_train = resultados_train_por_grupo[indice_mejor_promedio_pxs_segun_k]
mejores_promedios_test = resultados_test_por_grupo[indice_mejor_promedio_pxs_segun_k]

# Gráfico final del grupo con el mejor promedio de train y test.
plt.plot(valores_k, mejores_promedios_train, label='Train')
plt.plot(valores_k, mejores_promedios_test, label='Test')
plt.legend()
plt.title(f'Exactitud final del modelo kNN (Grupo {indice_mejor_promedio_pxs_segun_k+1} de {cantidades_pixeles[indice_mejor_promedio_pxs]} pxs)')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Exactitud (accuracy)')
plt.grid(True)
plt.show()

# Mostramos los píxeles utilizados por el mejor modelo.
indice_pixeles_utilizado = indice_mejor_promedio_pxs * cantidad_pruebas_por_cantidad_pxs + indice_mejor_promedio_pxs_segun_k
pixeles_utilizados = mejores_pixeles[indice_pixeles_utilizado]
print(f"Píxeles utilizados: {pixeles_utilizados}")

#%%
# === EJERCICIO 3: CLASIFICACIÓN MULTICLASE CON ÁRBOL DE DECISIÓN ===

# 3.a. Separar el conjunto de datos en desarrollo (dev) y validación (held-out).

# Separamos el dataset completo en un conjunto de desarrollo y otro de validación, que se usará únicamente al final para estimar la performance real del modelo.

# Separamos los datos (X) y las etiquetas (Y).
X = datos.drop("label", axis=1).values
Y = datos["label"].values

# Dividimos el dataset en conjunto de desarrollo (80%) y conjunto held-out (20%) de manera estratificada.
X_dev, X_heldout, Y_dev, Y_heldout = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# 3.b. Entrenar árboles con diferentes profundidades.

# Entrenamos árboles de decisión con profundidades crecientes para analizar su comportamiento y ver cómo varía la precisión en el conjunto de desarrollo.

dev_results = []

# Entrenamos diez modelos de árbol de decisión variando la profundidad de 1 a 10.
for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_dev, Y_dev)
    Y_pred = model.predict(X_dev)
    acc = metrics.accuracy_score(Y_dev, Y_pred)
    dev_results.append((depth, acc))
    print(f"Profundidad: {depth}, Accuracy en dev: {acc:.4f}")

# 3.c. Cross-validation para elegir la mejor profundidad.

# Utilizamos validación cruzada con 5 folds sobre el conjunto de desarrollo para encontrar la mejor profundidad del árbol de decisión.

cv_results = []

# Aplicamos validación cruzada con 5 folds para cada profundidad entre 1 y 10.
for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(model, X_dev, Y_dev, cv=5, scoring='accuracy')
    acc_promedio = np.mean(scores)
    cv_results.append((depth, acc_promedio))
    print(f"Profundidad: {depth}, Accuracy promedio CV: {acc_promedio:.4f}")

# Seleccionamos la mejor profundidad según la validación cruzada.
mejor_depth, mejor_acc = max(cv_results, key=lambda x: x[1])
print(f"\n>>> Mejor profundidad: {mejor_depth} con Accuracy promedio: {mejor_acc:.4f}")

# 3.d. Entrenar el mejor modelo y evaluar sobre el conjunto held-out.

# Entrenamos el modelo final con los mejores hiperparámetros y lo evaluamos sobre datos no vistos para estimar su rendimiento real en la tarea de clasificación multiclase.

# Entrenamos el modelo con la mejor profundidad sobre el conjunto de desarrollo completo.
final_model = DecisionTreeClassifier(max_depth=mejor_depth, random_state=42)
final_model.fit(X_dev, Y_dev)

# Predecimos las etiquetas del conjunto held-out.
Y_pred_heldout = final_model.predict(X_heldout)

# Calculamos y mostramos la exactitud final.
heldout_acc = accuracy_score(Y_heldout, Y_pred_heldout)
print(f"\nAccuracy final en held-out: {heldout_acc:.4f}")

# Mostramos la matriz de confusión para analizar los errores por clase.
matriz_confusion = confusion_matrix(Y_heldout, Y_pred_heldout)
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.title('Matriz de confusión en held-out')
plt.show()

# %%
