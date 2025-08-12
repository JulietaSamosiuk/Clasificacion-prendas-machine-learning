# Clasificación con Imágenes (Fashion-MNIST)

Este trabajo práctico se realizó en el marco de la materia Laboratorio de Datos (Licenciatura en Ciencias de Datos, UBA) – 1er Cuatrimestre 2025.

---

## Descripción

Este proyecto analiza y modela el conjunto de datos **Fashion-MNIST,** una base de 70.000 imágenes en escala de grises (28×28 píxeles) que representan prendas de vestir.  
El trabajo se divide en tres partes principales:

1. **Análisis exploratorio**  
   - Identificación de píxeles relevantes.
   - Estudio de similitudes y diferencias entre clases.
   - Análisis de variabilidad dentro de una misma clase.

2. **Clasificación binaria (kNN)**  
   - Diferenciación entre camisetas/tops (clase 0) y bolsas (clase 8).
   - Selección automática de píxeles más relevantes.
   - Comparación de configuraciones de atributos y valores de k.

3. **Clasificación multiclase (Árbol de Decisión)**  
   - Entrenamiento y evaluación con diferentes profundidades.
   - Validación cruzada para selección de hiperparámetros.
   - Evaluación final sobre conjunto held-out.

---

## Tecnologías y Librerías

- **Lenguaje:** Python
- **Librerías:**
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Instalación rápida:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Ejecución

1. Descargar el dataset Fashion-MNIST.csv y colocarlo en la carpeta raíz del proyecto.

2. Abrir el archivo TP2-Alvarez_Britez_Samosiuk.py en VSCode, Spyder o JupyterLab.

3. Ejecutar sección por sección (#%%) para reproducir análisis y modelos.

4. El script genera:
   - Visualizaciones del dataset.
   - Mapas de calor de píxeles relevantes.
   - Métricas de clasificación (accuracy, matriz de confusión).

---

## Estructura del repositorio

```plaintext
📂 Dataset/
   ├── Carpeta para el dataset `Fashion-MNIST.csv`.
📄 Enunciado.pdf
   ├── Enunciado del trabajo práctico con la descripción y objetivo del proyecto.
📄 Informe.pdf
   ├── Informe del trabajo práctico con explicación detallada del proceso y resultados.
📄 Código.py
   ├── Script principal en Python: análisis exploratorio, clasificación binaria (kNN) y multiclase (Árbol de Decisión), generación de visualizaciones.
📄 README.md
   ├── Descripción general del proyecto, instrucciones de instalación, ejecución y resultados destacados.
```
