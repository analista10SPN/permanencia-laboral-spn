"""
🗂️ DICCIONARIO DE CÓDIGOS - EATON HR DATABASE
Tabla de leyendas para todos los códigos usados en el sistema
"""

# ============================================
# CÓDIGOS DE CAMPOS PRINCIPALES
# ============================================

# GÉNERO (Genero)
GENERO = {
    '1': 'Masculino',
    '2': 'Femenino',
    '-1': 'Otro/Desconocido'
}

# ESTADO CIVIL (Estado_Civil)
ESTADO_CIVIL = {
    '1': 'Soltero/a',
    '2': 'Casado/a',
    '5': 'Unión Libre',
    '6': 'No Definido',
    '-1': 'Desconocido'
}

# NIVEL ACADÉMICO (Nivel_Academico)
NIVEL_ACADEMICO = {
    '16': 'Primaria Incompleta',
    '1': 'Primaria Completa',
    '13': 'Secundaria Incompleta',
    '15': 'Secundaria Completa',
    '18': 'Técnico',
    '19': 'Universitaria Incompleta',
    '7': 'Grado Universitario',
    '10': 'Posgrado',
    '8': 'Maestría',
    '17': 'Doctorado',
    '0': 'Desconocido'
}

# TIPO DE EMPLEADO (Tipo_Empleado)
TIPO_EMPLEADO = {
    '1': 'FIJO (Salaried)',
    '6': 'FIJO + VARIABLE (Mixed)',
    '0': 'Desconocido'
}

# MODALIDAD DE TRABAJO (Modalidad_De_Trabajo)
MODALIDAD_TRABAJO = {
    '1': 'Presencial',
    '2': 'Remoto',
    '3': 'Híbrido',
    '0': 'Desconocido'
}

# ESTATUS DEL EMPLEADO (Estatus)
ESTATUS = {
    'A': 'Activo',
    'I': 'Inactivo',
    'C': 'Cancelado',
    'S': 'Suspendido'
}

# ============================================
# JERARQUÍA DE POSICIONES
# ============================================

# JERARQUÍA POR REFERENCIA (Position_Hierarchy_Level)
JERARQUIA_POSICION = {
    9: 'Ejecutivo (Presidente, Vicepresidente)',
    8: 'Director',
    7: 'Gerente',
    6: 'Subgerente',
    5: 'Supervisor/Coordinador',
    4: 'Jefe/Encargado',
    3: 'Profesional (Especialista, Analista, Ejecutivo)',
    2: 'Técnico/Representante/Asesor',
    1: 'Operacional (Asistente, Auxiliar, Operador, Cajero, Vendedor)',
    0: 'Desconocido'
}

# ============================================
# CÓDIGOS DE NIVELES
# ============================================

# NIVEL SALARIAL (Nivel_Salarial)
# Estos son rangos jerárquicos de la empresa
NIVEL_SALARIAL = {
    1: 'Nivel 1 - Operativo',
    2: 'Nivel 2 - Operativo Senior',
    3: 'Nivel 3 - Técnico',
    4: 'Nivel 4 - Técnico Senior',
    5: 'Nivel 5 - Profesional',
    6: 'Nivel 6 - Profesional Senior',
    7: 'Nivel 7 - Supervisión',
    8: 'Nivel 8 - Gerencial',
    9: 'Nivel 9 - Gerencial Senior',
    10: 'Nivel 10 - Ejecutivo',
    0: 'No Asignado'
}

# ============================================
# ROLES ESPECÍFICOS POR INDUSTRIA
# ============================================

# ROLES DE GASOLINERA (Is_Gas_Station_Role)
GAS_STATION_ROLES = {
    'positions': [221, 158, 148, 223, 76],
    'departments': [16],
    'classifications': [16],
    'description': 'Roles relacionados con envasadora, GLP, operaciones de gasolinera'
}

# ROLES TÉCNICOS (Is_Technical_Role)
TECHNICAL_ROLES = {
    'positions': [255, 258, 261, 257, 259, 229, 208, 213, 53],
    'departments': [19],
    'classifications': [9, 109],
    'description': 'Roles de taller, mecánico, técnico, mantenimiento, soldador, pintor'
}

# ROLES DE VENTAS (Is_Sales_Role)
SALES_ROLES = {
    'positions': [28, 27, 219, 206, 217, 222, 246],
    'departments': [111, 116],
    'classifications': [19, 33],
    'description': 'Roles de ventas, asesor, servicio al cliente, contact center'
}

# ROLES DE CONDUCTOR (Is_Driver_Role)
DRIVER_ROLES = {
    'positions': [69, 72, 73, 67, 70, 1092, 201],
    'departments': [109, 132],
    'classifications': [62],
    'description': 'Roles de chofer, transportación, mensajero'
}

# ============================================
# CÓDIGOS DE TURNOS Y HORARIOS
# ============================================

# TURNOS (Turno) - Ejemplos comunes
TURNOS = {
    1: 'Turno Diurno',
    2: 'Turno Nocturno',
    3: 'Turno Mixto',
    4: 'Turno Rotativo',
    0: 'No Definido'
}

# HORARIOS (Codigo_Horario) - Ejemplos comunes
HORARIOS = {
    1: 'Horario Regular (8am-5pm)',
    2: 'Horario Extendido',
    3: 'Horario Flexible',
    4: 'Horario 24/7',
    0: 'No Definido'
}

# ============================================
# MOTIVOS DE SALIDA
# ============================================

# MOTIVOS DE CANCELACIÓN (Motivo_Cancelacion)
MOTIVOS_SALIDA = {
    'Renuncia': 'Renuncia Voluntaria',
    'Desahucio': 'Desahucio',
    'Despido': 'Despido',
    'Fallecimiento': 'Fallecimiento',
    'Jubilación': 'Jubilación',
    'Fin de Contrato': 'Finalización de Contrato'
}

# ============================================
# TIPOS DE ACCIONES PERSONALES
# ============================================

# RAZONES DE ACCIÓN (Razon_accion en Accion_Personal)
TIPOS_ACCION = {
    'Nuevo Ingreso': 'Nuevo Ingreso',
    'Promoción': 'Promoción',
    'Traslado': 'Traslado',
    'Transferencia': 'Transferencia',
    'Reestructuración': 'Reestructuración',
    'Reajuste salarial': 'Reajuste Salarial',
    'Aumento': 'Aumento de Salario',
    'Ajuste por inflación': 'Ajuste por Inflación',
    'Completivo': 'Pago Completivo'
}

# ============================================
# CÓDIGOS DE AUSENCIAS
# ============================================

# BASE DE CÁLCULO (base_calculo)
BASE_CALCULO = {
    'Dias': 'Días',
    'Horas': 'Horas'
}

# MOTIVOS DE AUSENCIA (Motivo_Ausencia) - Comunes
MOTIVOS_AUSENCIA = {
    1: 'Enfermedad',
    2: 'Permiso Personal',
    3: 'Vacaciones',
    4: 'Licencia Médica',
    5: 'Maternidad/Paternidad',
    6: 'Duelo',
    7: 'Asuntos Legales',
    8: 'Otros'
}

# ============================================
# BANDERAS DE RIESGO (RISK FLAGS)
# ============================================

RISK_FLAGS = {
    'High_Absence_Flag': {
        0: 'Ausencias Normales (≤10)',
        1: 'Ausencias Altas (>10)'
    },
    'Career_Stagnation_Flag': {
        0: 'Carrera en Progreso',
        1: 'Estancamiento Profesional (3+ años sin promoción)'
    },
    'Long_Time_No_Raise_Flag': {
        0: 'Aumentos Recientes',
        1: 'Sin Aumento en 2+ Años'
    },
    'Below_Market_Salary_Flag': {
        0: 'Salario Competitivo',
        1: 'Salario Bajo (< 80% del promedio del puesto)'
    },
    'High_Deduction_Flag': {
        0: 'Deducciones Normales',
        1: 'Deducciones Altas (>30% del ingreso)'
    }
}

# ============================================
# DEPARTAMENTOS (Ejemplos de ID_Departamento)
# ============================================

# Basado en los datos del SQL
DEPARTAMENTOS_CONOCIDOS = {
    16: 'Envasadora/GLP',
    19: 'Departamento Técnico',
    111: 'Ventas',
    116: 'Ventas - Sucursal',
    109: 'Transportación',
    132: 'Transportación - Logística'
}

# ============================================
# CLASIFICACIONES (Ejemplos)
# ============================================

CLASIFICACIONES_CONOCIDAS = {
    9: 'Técnico',
    16: 'Gasolinera',
    19: 'Ventas',
    33: 'Comercial',
    62: 'Conductor',
    109: 'Técnico Especializado'
}

# ============================================
# POSICIONES MÁS COMUNES (Position_Code)
# ============================================

POSICIONES_CONOCIDAS = {
    # Gasolinera
    221: 'Representante Envasadora',
    158: 'Operador GLP',
    148: 'Asistente Envasadora',
    223: 'Supervisor Gasolinera',
    76: 'Cajero Gasolinera',
    
    # Técnico
    255: 'Mecánico',
    258: 'Técnico Taller',
    261: 'Soldador',
    257: 'Pintor',
    259: 'Técnico Mantenimiento',
    229: 'Operador Grúa',
    208: 'Auxiliar Técnico',
    213: 'Técnico Especializado',
    53: 'Técnico General',
    
    # Ventas
    28: 'Asesor Ventas',
    27: 'Ejecutivo Ventas',
    219: 'Representante Ventas',
    206: 'Agente Contact Center',
    217: 'Asesor Servicio Cliente',
    222: 'Recepcionista',
    246: 'Vendedor',
    
    # Conductor
    69: 'Chofer',
    72: 'Chofer Ruta',
    73: 'Chofer Distribución',
    67: 'Mensajero',
    70: 'Chofer Pesado',
    1092: 'Chofer Especial',
    201: 'Conductor',
    
    # Genérico
    999: 'Posición General/Múltiple'
}

# ============================================
# FUNCIONES DE AYUDA
# ============================================

def get_description(code, lookup_dict, default='Desconocido'):
    """
    Obtiene la descripción de un código
    
    Args:
        code: El código a buscar
        lookup_dict: El diccionario de lookup
        default: Valor por defecto si no se encuentra
    
    Returns:
        str: Descripción del código
    """
    return lookup_dict.get(str(code), default)


def get_education_level_name(code):
    """Obtiene el nivel educativo por código"""
    return get_description(code, NIVEL_ACADEMICO)


def get_gender_name(code):
    """Obtiene el género por código"""
    return get_description(code, GENERO)


def get_marital_status_name(code):
    """Obtiene el estado civil por código"""
    return get_description(code, ESTADO_CIVIL)


def get_employee_type_name(code):
    """Obtiene el tipo de empleado por código"""
    return get_description(code, TIPO_EMPLEADO)


def get_work_modality_name(code):
    """Obtiene la modalidad de trabajo por código"""
    return get_description(code, MODALIDAD_TRABAJO)


def get_position_hierarchy_name(level):
    """Obtiene la jerarquía de posición por nivel"""
    return get_description(level, JERARQUIA_POSICION)


def get_position_name(code):
    """Obtiene el nombre de la posición por código"""
    return get_description(code, POSICIONES_CONOCIDAS)


def get_department_name(code):
    """Obtiene el nombre del departamento por código"""
    return get_description(code, DEPARTAMENTOS_CONOCIDOS, f'Departamento {code}')


def get_classification_name(code):
    """Obtiene el nombre de la clasificación por código"""
    return get_description(code, CLASIFICACIONES_CONOCIDAS, f'Clasificación {code}')


def get_salary_level_name(level):
    """Obtiene el nombre del nivel salarial"""
    return get_description(level, NIVEL_SALARIAL)


# ============================================
# MAPEO COMPLETO PARA CSV
# ============================================

def create_legend_dataframe():
    """
    Crea un DataFrame de pandas con todas las leyendas
    Para usar en la UI o exportar
    """
    import pandas as pd
    
    legends = []
    
    # Género
    for code, desc in GENERO.items():
        legends.append({
            'Campo': 'Gender_Male',
            'Código': code,
            'Descripción': desc,
            'Categoría': 'Demográfico'
        })
    
    # Estado Civil
    for code, desc in ESTADO_CIVIL.items():
        legends.append({
            'Campo': 'Has_Family_Responsibility',
            'Código': code,
            'Descripción': desc,
            'Categoría': 'Demográfico'
        })
    
    # Nivel Académico
    for code, desc in NIVEL_ACADEMICO.items():
        legends.append({
            'Campo': 'Education_Level_Ordinal',
            'Código': code,
            'Descripción': desc,
            'Categoría': 'Educación'
        })
    
    # Tipo Empleado
    for code, desc in TIPO_EMPLEADO.items():
        legends.append({
            'Campo': 'Is_Fixed_Employee / Is_Variable_Employee',
            'Código': code,
            'Descripción': desc,
            'Categoría': 'Empleo'
        })
    
    # Modalidad Trabajo
    for code, desc in MODALIDAD_TRABAJO.items():
        legends.append({
            'Campo': 'Is_In_Person_Work / Is_Remote_Work / Is_Hybrid_Work',
            'Código': code,
            'Descripción': desc,
            'Categoría': 'Modalidad'
        })
    
    # Jerarquía Posición
    for level, desc in JERARQUIA_POSICION.items():
        legends.append({
            'Campo': 'Position_Hierarchy_Level',
            'Código': str(level),
            'Descripción': desc,
            'Categoría': 'Jerarquía'
        })
    
    # Posiciones Conocidas
    for code, desc in POSICIONES_CONOCIDAS.items():
        legends.append({
            'Campo': 'Position_Code',
            'Código': str(code),
            'Descripción': desc,
            'Categoría': 'Posición'
        })
    
    # Departamentos
    for code, desc in DEPARTAMENTOS_CONOCIDOS.items():
        legends.append({
            'Campo': 'Department_Code',
            'Código': str(code),
            'Descripción': desc,
            'Categoría': 'Departamento'
        })
    
    # Clasificaciones
    for code, desc in CLASIFICACIONES_CONOCIDAS.items():
        legends.append({
            'Campo': 'Classification_Code',
            'Código': str(code),
            'Descripción': desc,
            'Categoría': 'Clasificación'
        })
    
    return pd.DataFrame(legends)


# ============================================
# USO EN STREAMLIT
# ============================================

def show_legend_in_streamlit():
    """
    Muestra la leyenda en Streamlit
    Para agregar al demo
    """
    import streamlit as st
    
    st.markdown("## 📖 Leyenda de Códigos")
    
    tabs = st.tabs([
        "Demográficos", 
        "Posiciones", 
        "Departamentos",
        "Jerarquía",
        "Banderas de Riesgo"
    ])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Género")
            for code, desc in GENERO.items():
                st.text(f"{code}: {desc}")
            
            st.markdown("### Estado Civil")
            for code, desc in ESTADO_CIVIL.items():
                st.text(f"{code}: {desc}")
        
        with col2:
            st.markdown("### Nivel Académico")
            for code, desc in NIVEL_ACADEMICO.items():
                st.text(f"{code}: {desc}")
    
    with tabs[1]:
        st.markdown("### Posiciones Principales")
        df = create_legend_dataframe()
        df_pos = df[df['Categoría'] == 'Posición']
        st.dataframe(df_pos, use_container_width=True)
    
    with tabs[2]:
        st.markdown("### Departamentos")
        for code, desc in DEPARTAMENTOS_CONOCIDOS.items():
            st.text(f"{code}: {desc}")
        
        st.markdown("### Clasificaciones")
        for code, desc in CLASIFICACIONES_CONOCIDAS.items():
            st.text(f"{code}: {desc}")
    
    with tabs[3]:
        st.markdown("### Jerarquía de Posiciones")
        for level in sorted(JERARQUIA_POSICION.keys(), reverse=True):
            st.text(f"Nivel {level}: {JERARQUIA_POSICION[level]}")
    
    with tabs[4]:
        st.markdown("### Banderas de Riesgo")
        for flag, values in RISK_FLAGS.items():
            st.markdown(f"**{flag}:**")
            for code, desc in values.items():
                st.text(f"  {code}: {desc}")


if __name__ == "__main__":
    # Test
    print("=== DICCIONARIO DE CÓDIGOS EATON HR ===\n")
    
    print("Género:", get_gender_name('1'))
    print("Estado Civil:", get_marital_status_name('2'))
    print("Educación:", get_education_level_name('7'))
    print("Posición:", get_position_name(221))
    print("Departamento:", get_department_name(16))
    print("Jerarquía:", get_position_hierarchy_name(7))
    
    print("\nCreando DataFrame de leyendas...")
    df = create_legend_dataframe()
    print(f"Total de entradas: {len(df)}")
    print(df.head(10))