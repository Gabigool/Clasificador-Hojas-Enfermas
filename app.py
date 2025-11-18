import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Hojas de Caña",
    page_icon="🌱",
    layout="wide"
)

# Función para cargar el modelo y clases con caché
@st.cache_resource
def load_model_and_classes():
    with open('sugarcane_leaf_classes.json', 'r') as f:
        classes = json.load(f)

    model = load_model('sugarcane_leaf_disease_model.h5')  # Carga modelo completo
    return model, classes

# Preprocesamiento de imagen para el modelo
def preprocess_image(image):
    # Forzar convertir a RGB para garantizar 3 canales
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Comprobar que tiene 3 canales
    if img_array.ndim == 2:  # Si sólo tiene altura y anchura, expandir canales
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] != 3:  # Si número de canales diferente de 3, lanza error
        raise ValueError(f"Imagen con número de canales inválido: {img_array.shape[2]}")
    
    img_array = np.expand_dims(img_array, axis=0)  # Añadir batch
    
    # Verificar el shape final
    if img_array.shape != (1, 224, 224, 3):
        raise ValueError(f"Shape inesperado después de preprocesar: {img_array.shape}")
    
    return img_array


# Cargar el modelo y las clases
model, classes = load_model_and_classes()

# Título y descripción
st.title("🌱 Clasificador de Hojas de Caña de Azúcar")
st.markdown("""
Esta aplicación utiliza **Deep Learning (CNN - MobileNetV2)** para identificar y clasificar enfermedades en hojas de caña de azúcar.

**Clases disponibles:** Healthy, Mosaic, RedRot, Rust y Yellow.
""")
st.markdown("---")

# Crear dos columnas
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Cargar Imagen")
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de hoja de caña (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)

with col2:
    st.subheader("🔍 Resultados")
    if uploaded_file is not None:
        if st.button("🌱 Clasificar Hoja", type="primary", use_container_width=True):
            with st.spinner("Analizando la imagen..."):
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img, verbose=0)
                predicted_class_idx = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][predicted_class_idx] * 100)

                # Obtener nombre de la clase (adaptado para int key)
                predicted_class = classes[str(predicted_class_idx)]

                st.success("✅ Clasificación completada")
                
                # Advertencia si la confianza es baja
                if confidence < 60:
                    st.warning("⚠️ **Advertencia:** El modelo no está seguro de esta clasificación (confianza < 60%). Los resultados pueden no ser confiables.")
                
                st.markdown(f"### 🌾 **{predicted_class}**")
                st.metric("Confianza", f"{confidence:.2f}%")

                # Estado saludable/enferma adaptado para caña
                if "healthy" in predicted_class.lower():
                    st.info("🟢 Estado: **Saludable**")
                else:
                    st.warning("🟠 Estado: **Enferma**")

                # Top 3
                st.markdown("---")
                st.markdown("#### 📊 Top 3 Predicciones")

                top3_idx = np.argsort(predictions[0])[-3:][::-1]
                for i, idx in enumerate(top3_idx):
                    class_name = classes[str(idx)]
                    prob = float(predictions[0][idx] * 100)
                    st.write(f"**{i+1}.** {class_name}: {prob:.2f}%")
                    st.progress(float(prob / 100))

                # Barra horizontal de probabilidades
                st.markdown("---")
                st.markdown("#### 📈 Distribución de Probabilidades")
                fig, ax = plt.subplots(figsize=(10, 8))
                class_names = [classes[str(i)] for i in range(len(predictions[0]))]
                probs = predictions[0] * 100

                sorted_indices = np.argsort(probs)[::-1]
                sorted_classes = [class_names[i] for i in sorted_indices]
                sorted_probs = [probs[i] for i in sorted_indices]
                bars = ax.barh(sorted_classes, sorted_probs, color='#388E3C')
                ax.set_xlabel('Probabilidad (%)')
                ax.set_title('Probabilidades de Clasificación')
                ax.grid(axis='x', alpha=0.3)
                bars[0].set_color('#81C784')
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.info("👆 Por favor, carga una imagen para comenzar")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Proyecto Final - Machine Learning | Caña de Azúcar</p>
    <p>Algoritmo: CNN (MobileNetV2) con Transfer Learning</p>
    <p>Accuracy en Test: 90.34%</p>
</div>
""", unsafe_allow_html=True)
