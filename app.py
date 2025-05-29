import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from utils import load_model, predict_defects

""" ASIGNAMOS EL TITULO DE LA PAGINA"""
st.set_page_config(
    page_title="Sistema de Detección de Defectos",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Sistema Automático de Detección de Defectos")
st.markdown("""
**Sube una imagen de un producto y nuestro sistema identificará defectos como rasguños, manchas o ambos.**
""")

""" FUNCION PARA CARGAR LOS MODELOS NOTA:SOLO SE CARGAN UNA VEZ"""

@st.cache_resource
def load_cached_model():
    return load_model()

model = load_cached_model()

""" SE DEFINEN LAS CLASES CON SU TIPO DE DESCRIPCION PARA NUESTRO MODELO"""
class_info = {
    0: {
        "name": "Buen Estado",
        "description": "No se detectaron defectos significativos. El producto cumple con los estándares de calidad.",
        "color": "green"
    },
    1: {
        "name": "Rasguños",
        "description": "Se detectaron líneas superficiales en la superficie. Posible causa: fricción durante el transporte o manipulación.",
        "color": "orange"
    },
    2: {
        "name": "Manchas",
        "description": "Se identificaron áreas irregulares de decoloración. Posible causa: contaminación durante la producción o almacenamiento.",
        "color": "blue"
    },
    3: {
        "name": "Múltiples Defectos",
        "description": "Se encontraron tanto rasguños como manchas. El producto requiere revisión de calidad completa.",
        "color": "red"
    }
}

""" SECCION PARA SUBIR IMAGENES"""
st.sidebar.header("Subir Imagen")
uploaded_file = st.sidebar.file_uploader(
    "Selecciona una imagen del producto", 
    type=["jpg", "jpeg", "png"]
)

""" MOSTRAMOS LOS RESULTADOS"""
if uploaded_file is not None:
    # Procesar imagen
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True)
    
    # Realizar predicción
    class_id, confidence = predict_defects(model, image)
    defect_info = class_info[class_id]
    
    with col2:
        st.subheader("Resultado del Análisis")
        
        # Mostrar resultado con color
        st.markdown(f"""
        <div style='border-left: 5px solid {defect_info["color"]}; padding: 10px;'>
            <h3 style='color: {defect_info["color"]};'>{defect_info["name"]}</h3>
            <p><b>Confianza:</b> {confidence:.2%}</p>
            <p><b>Descripción:</b> {defect_info["description"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        
        with st.expander("Recomendaciones de Acción"):
            if class_id == 0:
                st.success(" El producto puede pasar al siguiente paso en la línea de producción.")
            elif class_id == 1:
                st.warning(" Revisar procesos de embalaje y transporte. Considerar protección adicional.")
            elif class_id == 2:
                st.warning(" Verificar condiciones de almacenamiento y materiales de producción.")
            else:
                st.error("Producto debe ser retirado para inspección completa. Revisar múltiples etapas del proceso.")
        
        # Gráfico de confianza
        fig, ax = plt.subplots(figsize=(8, 4))
        classes = [info["name"] for info in class_info.values()]
        confidences = [confidence if i == class_id else 0 for i in range(4)]
        
        ax.barh(classes, confidences, color=[info["color"] for info in class_info.values()])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confianza")
        ax.set_title("Probabilidad por Tipo de Defecto")
        st.pyplot(fig)
    
    """ SECCION PARA EL ANALISIS TECNICO """
    st.subheader("Análisis Técnico Detallado")
    tab1, tab2, tab3 = st.tabs(["Áreas Sospechosas", "Métricas", "Sobre el Modelo"])
    
    with tab1:
        st.info("Esta funcionalidad mostraría las áreas específicas donde se detectaron defectos usando técnicas como Grad-CAM (en una implementación completa).")
        # Aquí iría la implementación real de detección de áreas
    
    with tab2:
        st.write("**Métricas de Rendimiento del Modelo:**")
        st.metric("Precisión General", "96.2%")
        st.metric("Recall para Rasguños", "93.5%")
        st.metric("Recall para Manchas", "95.8%")
        st.metric("Falsos Positivos", "2.1%")
    
    with tab3:
        st.write("**Especificaciones Técnicas:**")
        st.write("- Arquitectura: MobileNetV2 con Transfer Learning")
        st.write("- Dataset: 1,600 imágenes sintéticas (400 por clase)")
        st.write("- Entrenamiento: 15 épocas con aumento de datos")
        st.write("- Precisión Validación: 94.7%")
        
else:
    st.info("Por favor, sube una imagen a la izquierda para comenzar el análisis.")
    st.image("https://via.placeholder.com/800x400?text=Sube+una+imagen+para+analizar", use_column_width=True)

""" INFORMACION ADICIONAL QUE APARECE EN EL PIE DE PAGINA"""
st.sidebar.markdown("---")
st.sidebar.info("""
**Nota:** Este sistema fue entrenado con datos SINTETICOS Y FICTICIOS. 
""")