# Clasificador de Hojas de Caña de Azúcar

Esta aplicación web utiliza Deep Learning con una red convolucional (CNN) basada en MobileNetV2 para identificar y clasificar enfermedades en hojas de caña de azúcar a partir de imágenes cargadas por el usuario. Es una solución intuitiva que permite a agricultores, técnicos o investigadores obtener un diagnóstico rápido y confiable del estado de salud de las plantas.

---

## Características

- Clasificación automática de múltiples enfermedades comunes en hojas de caña (Healthy, Mosaic, RedRot, Rust, Yellow, entre otras).
- Visualización clara de la imagen cargada y resultado de la clasificación.
- Indicador de confianza en la predicción con mensaje de advertencia si la certeza es baja.
- Top 3 predicciones y distribución visual de probabilidades para un análisis detallado.
- Fácil de usar mediante una interfaz web desarrollada con Streamlit.

---

## Requisitos

- Python 3.8 o superior
- streamlit==1.51.0
- tensorflow==2.20.0
- tf-keras==2.20.1
- pillow==12.0.0
- numpy==2.2.6
- matplotlib==3.10.7

El archivo `requirements.txt` incluye todas las dependencias necesarias.

---

## Instalación y ejecución

1. Clonar el repositorio.
2. Crear y activar un entorno virtual de Python.
3. Instalar las dependencias:  
   `pip install -r requirements.txt`
4. Ejecutar la aplicación con:  
   `streamlit run appDisease.py`

---

## Modelo

El modelo entrenado está basado en MobileNetV2 con transfer learning. Se incluye el archivo `sugarcane_leaf_disease_model.h5` y el mapeo de clases `sugarcane_leaf_classes.json` para facilitar su uso e integración.

---

## Soporte y contacto

Para dudas o soporte técnico:

- Gabriel Santiago Cely Forero – gabriel.cely@uptc.edu.co  
- Julián Camilo Cerón Patiño – julian.ceron02@uptc.edu.co  
- Alec Fabián Corzo Salazar – alec.corzo@uptc.edu.co

También se recomienda revisar foros y documentación oficial de Streamlit y TensorFlow.

---

## Licencia

Este proyecto es para fines académicos y de investigación. Para usos comerciales consultar a los autores.

