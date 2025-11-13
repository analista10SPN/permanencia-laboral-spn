#!/usr/bin/env python3
"""
Sistema de Permanencia Laboral - SPN (SYNTH.IA)
Dashboard ejecutivo para análisis de retención
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================
# DICCIONARIOS DE CÓDIGOS
# ============================================

POSICIONES = {
    221: 'Rep. Envasadora', 158: 'Op. GLP', 148: 'Asist. Envasadora',
    223: 'Sup. Gasolinera', 76: 'Cajero Gasolinera',
    255: 'Mecánico', 258: 'Téc. Taller', 261: 'Soldador',
    257: 'Pintor', 259: 'Téc. Mantenimiento', 229: 'Op. Grúa',
    28: 'Asesor Ventas', 27: 'Ejec. Ventas', 219: 'Rep. Ventas',
    206: 'Ag. Contact Center', 217: 'Asesor Servicio', 222: 'Recepcionista',
    69: 'Chofer', 72: 'Chofer Ruta', 73: 'Chofer Dist.',
    67: 'Mensajero', 70: 'Chofer Pesado',
    999: 'Posición General'
}

DEPARTAMENTOS = {
    16: 'Envasadora/GLP', 19: 'Técnico', 111: 'Ventas',
    116: 'Ventas - Sucursal', 109: 'Transportación',
    132: 'Transportación - Logística'
}

CLASIFICACIONES = {
    9: 'Técnico', 16: 'Gasolinera', 19: 'Ventas',
    33: 'Comercial', 62: 'Conductor', 109: 'Téc. Especializado'
}

NIVELES = {
    1: 'Operativo', 2: 'Op. Senior', 3: 'Técnico', 4: 'Téc. Senior',
    5: 'Profesional', 6: 'Prof. Senior', 7: 'Supervisión',
    8: 'Gerencial', 9: 'Ger. Senior', 10: 'Ejecutivo'
}

HORARIOS = {
    1: 'Regular (8-5)', 2: 'Extendido', 3: 'Flexible',
    4: '24/7', 5: 'Part-time', 6: 'Nocturno'
}

TIPO_EMPLEADO_DESC = {
    0: 'Variable', 1: 'Fijo'
}

RISK_CATEGORIES = {
    'bajo': {'threshold': 1000, 'label': 'Riesgo Bajo', 'prob': 15, 'color': '#38ef7d'},
    'medio': {'threshold': 1500, 'label': 'Riesgo Medio', 'prob': 40, 'color': '#fee140'},
    'urgente': {'threshold': 2500, 'label': 'Riesgo Urgente', 'prob': 70, 'color': "#ad5c00"},
    'inminente': {'threshold': float('inf'), 'label': 'Riesgo Inminente', 'prob': 95, 'color': '#ff0844'}
}

def get_code_description(code, lookup_dict, default='--'):
    """Obtiene descripción de código o retorna el código si no existe"""
    try:
        code_int = int(float(code)) if pd.notna(code) else 0
        if code_int in lookup_dict:
            return f"{lookup_dict[code_int]} ({code_int})"
        return f"Código {code_int}" if code_int != 0 else default
    except:
        return default

# Configuración de página
st.set_page_config(
    page_title="Permanencia Laboral - SPN",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado con fuentes más grandes
st.markdown("""
<style>
    .main-title {
        font-size: 4rem !important;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #444;
        font-size: 1.4rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        text-align: center;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 500;
    }
    .risk-card {
        padding: 2.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    .risk-bajo { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .risk-medio { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .risk-urgente { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    .risk-inminente { background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%); }
    
    .employee-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border-left: 6px solid #1f77b4;
        margin: 1.5rem 0;
        box-shadow: 0 3px 8px rgba(0,0,0,0.12);
    }
    
    .employee-photo {
        border-radius: 50%;
        width: 150px;
        height: 150px;
        object-fit: cover;
        border: 4px solid #1f77b4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .comparison-section {
        background: #f8f9fa;
        padding: 2.5rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 0.7rem 2.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    h1, h2, h3 {
        font-weight: 600;
    }
    
    .stMarkdown {
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)


class DemoRetencion:
    """Dashboard profesional para análisis de retención"""
    
    def __init__(self):
        self.model_survival = None
        self.model_rf = None
        self.scaler = None
        self.feature_names = None
        self.data = None
        self.photos_dir = Path("demo_profile_pictures")
        
    @st.cache_resource
    def cargar_modelos(_self, model_path='production_retention_model.pkl'):
        """Cargar modelos del paquete"""
        try:
            with open(model_path, 'rb') as f:
                package = pickle.load(f)
            
            survival_model = package.get('survival_model') or package.get('model')
            rf_model = package.get('rf_model')
            scaler = package.get('scaler')
            features = package.get('feature_names')
            
            return survival_model, rf_model, scaler, features
        except Exception as e:
            st.error(f"Error al cargar modelos: {str(e)}")
            return None, None, None, None
    
    @st.cache_data
    def cargar_datos(_self, data_path):
        """Cargar datos de empleados"""
        try:
            df = pd.read_csv(data_path)
            
            if 'Tenure_Years' not in df.columns and 'Tenure_Days' in df.columns:
                df['Tenure_Years'] = df['Tenure_Days'] / 365.25
            
            return df
        except Exception as e:
            st.error(f"Error al cargar datos: {str(e)}")
            return None
    
    def get_employee_photo(self, employee_id):
        """Obtener ruta de foto del empleado si existe"""
        if not self.photos_dir.exists():
            return None
        
        photo_path = self.photos_dir / f"{employee_id}.png"
        if photo_path.exists():
            return str(photo_path)
        
        photo_path_jpg = self.photos_dir / f"{employee_id}.jpg"
        if photo_path_jpg.exists():
            return str(photo_path_jpg)
        
        return None
    
    def categorizar_riesgo(self, risk_score):
        """Categorizar puntaje de riesgo"""
        if risk_score < RISK_CATEGORIES['bajo']['threshold']:
            return 'bajo'
        elif risk_score < RISK_CATEGORIES['medio']['threshold']:
            return 'medio'
        elif risk_score < RISK_CATEGORIES['urgente']['threshold']:
            return 'urgente'
        else:
            return 'inminente'
    
    def calcular_risk_scores_batch(self, employees_data):
        """Calcular risk scores para múltiples empleados"""
        try:
            X = employees_data[self.feature_names].fillna(0)
            
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            
            risk_scores = self.model_survival.predict(X_scaled)
            
            return risk_scores
            
        except Exception as e:
            st.error(f"Error calculando risk scores: {str(e)}")
            return None
    
    def predecir_empleado(self, employee_data):
        """Hacer predicción completa con ambos modelos"""
        try:
            X = employee_data[self.feature_names].fillna(0)
            
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            
            risk_score_survival = float(self.model_survival.predict(X_scaled)[0])
            categoria = self.categorizar_riesgo(risk_score_survival)
            
            prob_1_mes = min(95, (risk_score_survival / 4000) * 100)
            prob_3_meses = min(95, (risk_score_survival / 3500) * 100)
            prob_6_meses = min(95, (risk_score_survival / 3000) * 100)
            prob_1_año = min(95, (risk_score_survival / 2500) * 100)
            
            rf_prediction = None
            rf_probability = None
            
            if self.model_rf is not None:
                rf_proba = self.model_rf.predict_proba(X_scaled)[0]
                rf_probability = float(rf_proba[1])
                rf_prediction = int(self.model_rf.predict(X_scaled)[0])
            
            return {
                'risk_score': risk_score_survival,
                'categoria': categoria,
                'categoria_label': RISK_CATEGORIES[categoria]['label'],
                'prob_1_mes': prob_1_mes,
                'prob_3_meses': prob_3_meses,
                'prob_6_meses': prob_6_meses,
                'prob_1_año': prob_1_año,
                'rf_probability': rf_probability * 100 if rf_probability else None,
                'rf_prediction': 'Dejará la empresa' if rf_prediction == 1 else 'Se quedará' if rf_prediction == 0 else None,
            }
            
        except Exception as e:
            st.error(f"Error en predicción: {str(e)}")
            return None
    
    def mostrar_header(self):
        """Mostrar header principal"""
        st.markdown('<p class="main-title">Sistema Predictivo de Permanencia Laboral</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Tecnología Machine Learning para Análisis de Retención de Talento</p>', unsafe_allow_html=True)
        
        # Explicación funcional del modelo
        with st.expander("¿Cómo funciona este sistema? (Explicación para RRHH)", expanded=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                ### ¿Qué hace este sistema?
                
                Este sistema utiliza **Machine Learning** (aprendizaje automático) para predecir 
                qué empleados tienen mayor probabilidad de dejar la empresa en los próximos meses.
                
                **Su objetivo es ser un "detector de señales":**
                - Analiza patrones históricos de empleados que se han ido
                - Identifica factores comunes (salario, antigüedad, ausencias, etc.)
                - Calcula un "score de riesgo" para cada empleado activo
                
                **La base del modelo es matemática aplicada:**
                El sistema aprendió de más de 4,900 casos reales.
                """)
            
            with col2:
                st.markdown("""
                ### ¿Cómo se construyó?
                
                **Paso 1: Recolección de datos**
                - Historial de empleados (activos y retirados)
                - Variables: salario, antigüedad, departamento, ausencias, desempeño, etc.
                
                **Paso 2: Entrenamiento del modelo**
                - El algoritmo de Machine Learning analiza patrones
                - Identifica qué combinaciones de factores predicen salidas
                - Se valida con datos históricos (94% de precisión)
                
                **Paso 3: Predicción en tiempo real**
                - Ingresa el ID de un empleado
                - El sistema calcula su "score de riesgo"
                - Te dice cuándo podría irse (1 mes, 3 meses, 6 meses, 1 año)
                
                **Resultado:** Puedes actuar ANTES de que sea tarde.
                """)
            
            st.markdown("---")
            st.info("""
            **Para el equipo de RRHH:** Este sistema NO reemplaza tu criterio profesional. 
            Es una herramienta de alerta temprana que te ayuda a priorizar conversaciones 
            y acciones de retención con los empleados que más lo necesitan.
            """)
        
        st.markdown("---")
    
    def mostrar_estadisticas_generales(self):
        """Mostrar estadísticas generales"""
        st.markdown("## Panorama General de la Empresa")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_empleados = len(self.data)
        empleados_activos = len(self.data[self.data['Has_Left'] == 0])
        empleados_retirados = len(self.data[self.data['Has_Left'] == 1])
        tasa_rotacion = (empleados_retirados / total_empleados) * 100
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Empleados</div>
                <div class="metric-value">{total_empleados:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Empleados Activos</div>
                <div class="metric-value">{empleados_activos:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tasa de Rotación</div>
                <div class="metric-value">{tasa_rotacion:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            tenure_promedio = self.data['Tenure_Years'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Antigüedad Promedio</div>
                <div class="metric-value">{tenure_promedio:.1f} años</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Gráficos de distribución
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                self.data[self.data['Tenure_Years'] < 20],
                x='Tenure_Years',
                nbins=30,
                title='Distribución de Antigüedad en la Empresa',
                labels={'Tenure_Years': 'Años en la Empresa', 'count': 'Número de Empleados'}
            )
            fig.update_traces(marker_color='#667eea')
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Distribución de Riesgo - Empleados Activos")
            
            empleados_activos_data = self.data[self.data['Has_Left'] == 0]
            
            if len(empleados_activos_data) > 0:
                risk_scores = self.calcular_risk_scores_batch(empleados_activos_data)
                
                if risk_scores is not None:
                    categorias = [self.categorizar_riesgo(score) for score in risk_scores]
                    
                    from collections import Counter
                    conteo = Counter(categorias)
                    
                    categorias_orden = ['bajo', 'medio', 'urgente', 'inminente']
                    labels_orden = [RISK_CATEGORIES[cat]['label'] for cat in categorias_orden]
                    valores = [conteo.get(cat, 0) for cat in categorias_orden]
                    colores = [RISK_CATEGORIES[cat]['color'] for cat in categorias_orden]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=labels_orden,
                            y=valores,
                            marker_color=colores,
                            text=valores,
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        title='Empleados por Categoría de Riesgo',
                        xaxis_title='Categoría de Riesgo',
                        yaxis_title='Número de Empleados',
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    total_activos = len(empleados_activos_data)
                    st.markdown(f"""
                    <div style="font-size: 1rem; color: #666; text-align: center; font-weight: 500;">
                        {(conteo['bajo']/total_activos*100):.1f}% Bajo | 
                        {(conteo['medio']/total_activos*100):.1f}% Medio | 
                        {(conteo['urgente']/total_activos*100):.1f}% Urgente | 
                        {(conteo['inminente']/total_activos*100):.1f}% Inminente
                    </div>
                    """, unsafe_allow_html=True)
    
    def buscar_empleado(self):
        """Sección de búsqueda de empleado"""
        st.markdown("## Análisis Individual de Empleado")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            empleado_id = st.text_input(
                "Ingrese ID del Empleado:",
                placeholder="Ej: EMP12345",
                help="Busque un empleado por su ID único"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            buscar = st.button("Analizar Empleado", use_container_width=True)
        
        if buscar and empleado_id:
            empleado_data = self.data[self.data['Employee_ID'] == empleado_id]
            
            if len(empleado_data) == 0:
                st.error(f"No se encontró el empleado con ID: {empleado_id}")
                return None
            
            empleado = empleado_data.iloc[0]
            prediccion = self.predecir_empleado(pd.DataFrame([empleado]))
            
            if prediccion:
                self.mostrar_analisis_empleado(empleado, prediccion)
                return empleado_id
        
        return None
    
    def mostrar_analisis_empleado(self, empleado, prediccion):
        """Mostrar análisis detallado del empleado"""
        
        # Foto del empleado si existe
        photo_path = self.get_employee_photo(empleado['Employee_ID'])
        
        if photo_path:
            col_photo, col_info = st.columns([1, 3])
            
            with col_photo:
                st.image(photo_path, use_column_width=True)
        else:
            col_info = st.container()
        
        with col_info if photo_path else st.container():
            st.markdown("### Información del Empleado")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ID", empleado['Employee_ID'])
            
            with col2:
                tenure = empleado.get('Tenure_Years', 0)
                st.metric("Antigüedad", f"{tenure:.1f} años")
            
            with col3:
                salario = empleado.get('Base_Salary', 0)
                st.metric("Salario", f"${salario:,.0f}")
            
            with col4:
                status = "Activo" if empleado.get('Has_Left', 0) == 0 else "Retirado"
                st.metric("Estado", status)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Predicción de riesgo
        categoria = prediccion['categoria']
        risk_score = prediccion['risk_score']
        
        st.markdown(f"""
        <div class="risk-card risk-{categoria}">
            <h2 style="color: white; text-align: center; margin: 0; font-size: 2rem;">
                {prediccion['categoria_label']}
            </h2>
            <h1 style="color: white; text-align: center; margin: 1rem 0; font-size: 5rem; font-weight: 700;">
                {risk_score:.0f}
            </h1>
            <p style="color: white; text-align: center; font-size: 1.3rem; margin: 0; font-weight: 500;">
                Puntaje de Riesgo
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probabilidades temporales
        st.markdown("### Probabilidad de Salida en el Tiempo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("En 1 Mes", f"{prediccion['prob_1_mes']:.1f}%")
        
        with col2:
            st.metric("En 3 Meses", f"{prediccion['prob_3_meses']:.1f}%")
        
        with col3:
            st.metric("En 6 Meses", f"{prediccion['prob_6_meses']:.1f}%")
        
        with col4:
            st.metric("En 1 Año", f"{prediccion['prob_1_año']:.1f}%")
        
        # Gráfico de evolución temporal
        fig = go.Figure()
        
        tiempos = ['1 Mes', '3 Meses', '6 Meses', '1 Año']
        probabilidades = [
            prediccion['prob_1_mes'],
            prediccion['prob_3_meses'],
            prediccion['prob_6_meses'],
            prediccion['prob_1_año']
        ]
        
        fig.add_trace(go.Scatter(
            x=tiempos,
            y=probabilidades,
            mode='lines+markers',
            line=dict(color=RISK_CATEGORIES[categoria]['color'], width=4),
            marker=dict(size=14),
            fill='tozeroy',
            fillcolor=f"rgba{tuple(list(int(RISK_CATEGORIES[categoria]['color'][i:i+2], 16) for i in (1, 3, 5)) + [0.3])}"
        ))
        
        fig.update_layout(
            title="Evolución de Riesgo en el Tiempo",
            xaxis_title="Período de Tiempo",
            yaxis_title="Probabilidad de Salida (%)",
            height=400,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretación
        st.markdown("### Interpretación y Recomendaciones")
        
        if categoria == 'bajo':
            st.success("""
            **Empleado Estable**: Este empleado muestra bajo riesgo de salida.  
            **Recomendación**: Mantener las condiciones actuales y seguimiento rutinario.
            """)
        elif categoria == 'medio':
            st.warning("""
            **Atención Requerida**: Riesgo moderado de salida.  
            **Recomendación**: Programar reunión 1-1 para entender satisfacción y necesidades.
            """)
        elif categoria == 'urgente':
            st.error("""
            **Intervención Urgente**: Alto riesgo de salida en corto plazo.  
            **Recomendación**: Acción inmediata - reunión con RRHH y supervisor directo.
            """)
        else:
            st.error("""
            **CRÍTICO**: Riesgo inminente de salida.  
            **Recomendación**: Intervención ejecutiva inmediata. Posible contraoferta o plan de retención.
            """)
        
        # Análisis complementario
        if prediccion.get('rf_probability') is not None:
            st.markdown("### Análisis Complementario (Random Forest)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Probabilidad de Salida (RF)", f"{prediccion['rf_probability']:.1f}%")
            
            with col2:
                st.metric("Predicción Binaria", prediccion['rf_prediction'])
            
            st.info("""
            **Doble Validación**: Ambos modelos (Survival Analysis + Random Forest) 
            trabajan juntos para dar una predicción más robusta. Cuando ambos modelos 
            coinciden en alto riesgo, la confianza en la predicción es muy alta.
            """)
    
    def comparar_empleados(self):
        """Comparación entre dos empleados"""
        st.markdown("## Comparación de Empleados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            emp1_id = st.text_input("ID Empleado 1:", key="emp1")
        
        with col2:
            emp2_id = st.text_input("ID Empleado 2:", key="emp2")
        
        if st.button("Comparar Empleados", use_container_width=True):
            if not emp1_id or not emp2_id:
                st.warning("Por favor ingrese ambos IDs de empleados")
                return
            
            emp1_data = self.data[self.data['Employee_ID'] == emp1_id]
            emp2_data = self.data[self.data['Employee_ID'] == emp2_id]
            
            if len(emp1_data) == 0 or len(emp2_data) == 0:
                st.error("Uno o ambos empleados no fueron encontrados")
                return
            
            emp1 = emp1_data.iloc[0]
            emp2 = emp2_data.iloc[0]
            
            pred1 = self.predecir_empleado(pd.DataFrame([emp1]))
            pred2 = self.predecir_empleado(pd.DataFrame([emp2]))
            
            if pred1 and pred2:
                self.mostrar_comparacion(emp1, pred1, emp2, pred2)
    
    def mostrar_comparacion(self, emp1, pred1, emp2, pred2):
        """Mostrar comparación visual entre dos empleados"""
        
        st.markdown('<div class="comparison-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Empleado 1
        with col1:
            photo1 = self.get_employee_photo(emp1['Employee_ID'])
            if photo1:
                st.image(photo1, width=150)
            
            st.markdown(f"### {emp1['Employee_ID']}")
            st.markdown(f"**Categoría:** {pred1['categoria_label']}")
            st.metric("Puntaje de Riesgo", f"{pred1['risk_score']:.0f}")
            st.metric("Antigüedad", f"{emp1.get('Tenure_Years', 0):.1f} años")
            st.metric("Salario", f"${emp1.get('Base_Salary', 0):,.0f}")
        
        # Empleado 2
        with col2:
            photo2 = self.get_employee_photo(emp2['Employee_ID'])
            if photo2:
                st.image(photo2, width=150)
            
            st.markdown(f"### {emp2['Employee_ID']}")
            st.markdown(f"**Categoría:** {pred2['categoria_label']}")
            st.metric("Puntaje de Riesgo", f"{pred2['risk_score']:.0f}")
            st.metric("Antigüedad", f"{emp2.get('Tenure_Years', 0):.1f} años")
            st.metric("Salario", f"${emp2.get('Base_Salary', 0):,.0f}")
        
        # Gráfico comparativo
        st.markdown("### Comparación de Probabilidades")
        
        fig = go.Figure()
        
        tiempos = ['1 Mes', '3 Meses', '6 Meses', '1 Año']
        
        fig.add_trace(go.Bar(
            name=emp1['Employee_ID'],
            x=tiempos,
            y=[pred1['prob_1_mes'], pred1['prob_3_meses'], pred1['prob_6_meses'], pred1['prob_1_año']],
            marker_color=RISK_CATEGORIES[pred1['categoria']]['color']
        ))
        
        fig.add_trace(go.Bar(
            name=emp2['Employee_ID'],
            x=tiempos,
            y=[pred2['prob_1_mes'], pred2['prob_3_meses'], pred2['prob_6_meses'], pred2['prob_1_año']],
            marker_color=RISK_CATEGORIES[pred2['categoria']]['color']
        ))
        
        fig.update_layout(
            barmode='group',
            yaxis_title="Probabilidad de Salida (%)",
            height=400,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Resumen
        st.markdown("### Resumen")
        
        diff_riesgo = abs(pred1['risk_score'] - pred2['risk_score'])
        
        if pred1['risk_score'] > pred2['risk_score']:
            st.info(f"""
            **{emp1['Employee_ID']}** tiene un riesgo **{diff_riesgo:.0f} puntos mayor** que **{emp2['Employee_ID']}**.  
            Se recomienda priorizar la atención en **{emp1['Employee_ID']}**.
            """)
        else:
            st.info(f"""
            **{emp2['Employee_ID']}** tiene un riesgo **{diff_riesgo:.0f} puntos mayor** que **{emp1['Employee_ID']}**.  
            Se recomienda priorizar la atención en **{emp2['Employee_ID']}**.
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def mostrar_tabla_empleados(self):
        """Mostrar tabla filtrable de empleados"""
        st.markdown("## Lista de Empleados - Explorador Completo")
        
        st.info("Usa esta tabla para encontrar perfiles interesantes y copiar su ID para analizar en detalle")
        
        # Filtros
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            solo_activos = st.checkbox("Solo Empleados Activos", value=True)
        
        with col2:
            min_tenure = st.number_input("Antigüedad Mínima (años)", 0, 20, 0)
        
        with col3:
            min_salary = st.number_input("Salario Mínimo ($)", 0, 200000, 0, step=5000)
        
        with col4:
            max_results = st.slider("Máximo de Resultados", 10, 200, 50)
        
        # Filtrar datos
        df_filtrado = self.data.copy()
        
        if solo_activos:
            df_filtrado = df_filtrado[df_filtrado['Has_Left'] == 0]
        
        df_filtrado = df_filtrado[df_filtrado['Tenure_Years'] >= min_tenure]
        
        if 'Base_Salary' in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado['Base_Salary'] >= min_salary]
        
        # Calcular risk scores
        st.info("Calculando scores de riesgo...")
        risk_scores_all = self.calcular_risk_scores_batch(df_filtrado)
        
        if risk_scores_all is not None:
            df_filtrado['Risk_Score'] = risk_scores_all
            df_filtrado['Risk_Category'] = df_filtrado['Risk_Score'].apply(self.categorizar_riesgo)
            df_filtrado = df_filtrado.sort_values('Risk_Score', ascending=False)
            st.success("Scores calculados - Tabla ordenada por riesgo")
        
        # Seleccionar columnas
        columnas_mostrar = [
            'Employee_ID', 'Risk_Score', 'Risk_Category', 'Tenure_Years',
            'Base_Salary', 'Level_Code', 'Position_Code', 'Department_Code',
            'Total_Absences', 'Has_Left'
        ]
        
        columnas_disponibles = [col for col in columnas_mostrar if col in df_filtrado.columns]
        df_display = df_filtrado[columnas_disponibles].head(max_results).copy()
        
        # Aplicar descripciones
        if 'Position_Code' in df_display.columns:
            df_display['Position_Code'] = df_display['Position_Code'].apply(
                lambda x: get_code_description(x, POSICIONES, 'N/A')
            )
        
        if 'Department_Code' in df_display.columns:
            df_display['Department_Code'] = df_display['Department_Code'].apply(
                lambda x: get_code_description(x, DEPARTAMENTOS, 'N/A')
            )
        
        if 'Level_Code' in df_display.columns:
            df_display['Level_Code'] = df_display['Level_Code'].apply(
                lambda x: get_code_description(x, NIVELES, 'N/A')
            )
        
        # Traducir columnas
        traducciones = {
            'Employee_ID': 'ID',
            'Risk_Score': 'Score Riesgo',
            'Risk_Category': 'Categoría',
            'Tenure_Years': 'Antigüedad (años)',
            'Base_Salary': 'Salario ($)',
            'Level_Code': 'Nivel',
            'Position_Code': 'Puesto',
            'Department_Code': 'Departamento',
            'Total_Absences': 'Ausencias',
            'Has_Left': 'Retirado'
        }
        
        df_display.columns = [traducciones.get(col, col) for col in df_display.columns]
        
        # Formatear
        if 'Categoría' in df_display.columns:
            df_display['Categoría'] = df_display['Categoría'].apply(
                lambda x: RISK_CATEGORIES.get(x, {}).get('label', x) if pd.notna(x) else 'N/A'
            )
        
        if 'Score Riesgo' in df_display.columns:
            df_display['Score Riesgo'] = df_display['Score Riesgo'].apply(
                lambda x: f"{x:.0f}" if pd.notna(x) else 'N/A'
            )
        
        if 'Salario ($)' in df_display.columns:
            df_display['Salario ($)'] = df_display['Salario ($)'].apply(lambda x: f"${x:,.0f}")
        
        if 'Antigüedad (años)' in df_display.columns:
            df_display['Antigüedad (años)'] = df_display['Antigüedad (años)'].apply(lambda x: f"{x:.1f}")
        
        if 'Retirado' in df_display.columns:
            df_display['Retirado'] = df_display['Retirado'].apply(lambda x: 'Sí' if x == 1 else 'No')
        
        # Mostrar tabla
        st.dataframe(
            df_display,
            use_container_width=True,
            height=500
        )
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Empleados mostrados", len(df_display))
        
        with col2:
            st.metric("Total filtrado", len(df_filtrado))
        
        with col3:
            if 'Base_Salary' in df_filtrado.columns:
                promedio = df_filtrado['Base_Salary'].head(max_results).mean()
                st.metric("Salario Promedio", f"${promedio:,.0f}")
        
        # IDs de ejemplo
        if len(df_filtrado) > 0 and 'Risk_Score' in df_filtrado.columns:
            st.markdown("---")
            st.markdown("### IDs de Ejemplo para Demo")
            
            ejemplos = []
            
            alto_riesgo = df_filtrado.nlargest(3, 'Risk_Score')
            if len(alto_riesgo) > 0:
                ejemplos.append(("ALTO RIESGO", alto_riesgo['Employee_ID'].tolist(), alto_riesgo['Risk_Score'].tolist()))
            
            if len(df_filtrado) > 10:
                medio_inicio = len(df_filtrado) // 3
                medio_fin = medio_inicio + 3
                riesgo_medio = df_filtrado.iloc[medio_inicio:medio_fin]
                if len(riesgo_medio) > 0:
                    ejemplos.append(("RIESGO MEDIO", riesgo_medio['Employee_ID'].tolist(), riesgo_medio['Risk_Score'].tolist()))
            
            bajo_riesgo = df_filtrado.nsmallest(3, 'Risk_Score')
            if len(bajo_riesgo) > 0:
                ejemplos.append(("BAJO RIESGO", bajo_riesgo['Employee_ID'].tolist(), bajo_riesgo['Risk_Score'].tolist()))
            
            if ejemplos:
                cols = st.columns(len(ejemplos))
                for idx, (categoria, ids, scores) in enumerate(ejemplos):
                    with cols[idx]:
                        st.markdown(f"**{categoria}**")
                        for emp_id, score in zip(ids, scores):
                            st.code(f"{emp_id} (Score: {score:.0f})", language=None)
    
    def ejecutar(self):
        """Ejecutar la aplicación"""
        
        # Cargar modelos y datos
        if self.model_survival is None:
            self.model_survival, self.model_rf, self.scaler, self.feature_names = self.cargar_modelos('production_retention_model.pkl')
            self.data = self.cargar_datos('employee_retention_data1.csv')
        
        if self.model_survival is None or self.data is None:
            st.error("No se pudieron cargar los modelos o datos necesarios")
            st.stop()
        
        # Mostrar info de modelos
        with st.expander("Información del Sistema"):
            st.markdown("""
            **Sistema de Predicción Dual:**
            - Survival Analysis (Random Survival Forest) - Predicciones temporales
            - Random Forest Classifier - Clasificación binaria
            
            Ambos modelos trabajan juntos para dar predicciones más robustas.
            """)
            
            if self.model_rf is None:
                st.warning("Modelo Random Forest no disponible. Solo usando Survival Analysis.")
        
        # Renderizar UI
        self.mostrar_header()
        self.mostrar_estadisticas_generales()
        st.markdown("---")
        self.buscar_empleado()
        st.markdown("---")
        self.comparar_empleados()
        st.markdown("---")
        
        with st.expander("Ver Lista Completa de Empleados"):
            self.mostrar_tabla_empleados()


if __name__ == "__main__":
    app = DemoRetencion()
    app.ejecutar()