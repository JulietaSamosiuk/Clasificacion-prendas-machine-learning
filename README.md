# Clasificación con Imágenes (Fashion-MNIST)

**Materia:** Laboratorio de Datos – 1er Cuatrimestre 2025  

---

## 📌 Descripción

Este proyecto analiza y modela el conjunto de datos **Fashion-MNIST,** una base de 70.000 imágenes en escala de grises (28×28 píxeles) que representan prendas de vestir.  
El trabajo se divide en tres partes principales:

1. **Análisis exploratorio**  
   - Identificación de píxeles relevantes.
   - Estudio de similitudes y diferencias entre clases.
   - Análisis de variabilidad dentro de una misma clase.

2. **Clasificación binaria (kNN)**  
   - Diferenciación entre camisetas/tops (clase 0) y bolsas (clase 8).
   - Selección automática de píxeles más relevantes.
   - Comparación de configuraciones de atributos y valores de *k.*

3. **Clasificación multiclase (Árbol de Decisión)**  
   - Entrenamiento y evaluación con diferentes profundidades.
   - Validación cruzada para selección de hiperparámetros.
   - Evaluación final sobre conjunto *held-out.*

---

## 🛠 Tecnologías y Librerías

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
