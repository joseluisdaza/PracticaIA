# Predicción de Precios de Bienes Raices en Australia - Regresión Avanzada

## Índice

- [Índice](#índice)
- [Introducción](#introducción)
  - [Métodos Utilizados](#métodos-utilizados)
  - [Tecnologías](#tecnologías)
- [Descarga y Configuración](#descarga-y-configuración)
  - [Requisitos Previos](#requisitos-previos)
  - [Cómo Ejecutar](#cómo-ejecutar)
- [Declaración del Problema](#declaración-del-problema)
  - [Objetivo Comercial](#objetivo-comercial)
  - [Preparación de Datos:](#preparación-de-datos)
  - [Construcción y Evaluación del Modelo](#construcción-y-evaluación-del-modelo)
  - [Conclusiones](#conclusiones)
    - [Regresión Ridge](#regresión-ridge)
    - [Regresión Lasso](#regresión-lasso)
    - [Regresión ElasticNet](#regresión-lasso)
    - [Las Variables Más Significativas Son:](#las-variables-más-significativas-son)

## Introducción

Construir un modelo de Machine Learning avanzado para predecir precios de propiedades inmobiliarias en Australia utilizando PyCaret, aplicando todas las fases de un proyecto profesional de Ciencia de Datos.

### Métodos Utilizados

1. **Análisis Exploratorio de Datos (EDA)**
   - Evaluación de valores faltantes
   - Identificación de registros duplicados
   - Detección de valores atípicos (outliers) usando IQR
   - Análisis de distribuciones de variables
   - Matriz de correlación

2. **Preparación y Limpieza de Datos**
   - Imputación de valores faltantes (mediana para numéricas, moda para categóricas)
   - Eliminación de registros duplicados
   - Tratamiento de outliers usando IQR
   - One-Hot Encoding para variables categóricas

3. **Feature Engineering**
   - Creación de características derivadas (polinomios)
   - Variables de interacción
   - Transformaciones logarítmicas
   - Cálculo de edad de la propiedad

4. **Modelado Automático con PyCaret**
   - Comparación de múltiples algoritmos de regresión
   - Selección automática de características
   - Tunizado de hiperparámetros
   - Validación cruzada (5 folds)

5. **Algoritmos Evaluados**
   - Regresión Lineal (LR)
   - Ridge Regression
   - Lasso Regression
   - Elastic Net
   - Random Forest (RF)
   - Gradient Boosting (GBR)
   - XGBoost

### Tecnologías

1. Manipulación y Análisis de Datos
   - pandas
   - numpy

2. Machine Learning y Modelos
   - scikit-learn
   - xgboost
   - lightgbm

3. Automatización de ML
   - pycaret

4. Visualización
   - matplotlib
   - seaborn
   - plotly

5. Estadísticas
   - scipy
   - statsmodels

6. Procesamiento e Imputación
   - imbalanced-learn

7. Validación y Evaluación
   - shap

8. Utilitarios
   - python-dotenv
   - tqdm

9. Jupyter Notebook
   - jupyter
   - jupyterlab
   - ipywidgets

10. Otros
    - requests

## Descarga y Configuración

### Requisitos Previos

Este proyecto necesita que Anaconda esté instalado en la computadora.

Para más detalles sobre la instalación, visite: https://docs.anaconda.com/anaconda/install/index.html

### Cómo Ejecutar

Puede descargar el código fuente clonando este repositorio usando Git:

1. Abra su aplicación Terminal favorita (Unix, Linux o Macos), como Terminal, Comando, Consola, iTerm2, etc.

2. Clone el repositorio

```
git clone https://github.com/joseluisdaza/PracticaIA.git
```

3. Abra el archivo notebook ** regression_pycaret.ipynb** en Anaconda o en VSCode con el plugin de Anaconda configurado.

```
jupyter notebook regression_pycaret.ipynb
```

## Declaración del Problema

El análisis de datos de precios de propiedades en Australia tomando en cuenta diferentes características cómo el tipo de clasificación del terreno, superficie, ubicación eta

### Objetivo Comercial

Desarrollar un modelo predictivo de Machine Learning que permita estimar con precisión los precios de propiedades inmobiliarias en Australia. Este modelo es útil para:

- **Valuación Automática**: Proporcionar estimaciones rápidas y objetivas de precios de propiedades
- **Análisis de Inversiones**: Ayudar a inversores a identificar oportunidades de compra/venta
- **Apoyo a Decisiones**: Facilitar a agentes inmobiliarios información basada en datos para negociaciones
- **Predicción de Mercado**: Analizar tendencias de precios basadas en características de las propiedades

---

### Preparación de Datos:

#### 1. Carga del Dataset

- Se carga el dataset de propiedades australianas o Boston Housing como respaldo
- Normalización de nombres de columnas a mayúsculas
- Eliminación de columnas no predictivas (como ID)

#### 2. Análisis de Calidad de Datos

- Identificación de valores faltantes por columna
- Cálculo de porcentaje de datos faltantes
- Detección de registros duplicados
- Análisis de distribuciones de variables

#### 3. Detección de Valores Atípicos (Outliers)

- Método IQR (Rango Intercuartílico)
- Identificación de outliers: valores < Q1 - 1.5×IQR o > Q3 + 1.5×IQR
- Reporte de outliers por variable

#### 4. Limpieza de Datos

- **Imputación de valores faltantes:**
  - Variables numéricas: mediana
  - Variables categóricas: moda (valor más frecuente)
- **Eliminación de duplicados:** mantiene primera ocurrencia
- **Tratamiento de outliers:** limitación a límites IQR

#### 5. Codificación de Variables Categóricas

- One-Hot Encoding para variables categóricas
- Eliminación de una categoría para evitar multicolinealidad (drop_first=True)

#### 6. Feature Engineering

- **Características derivadas de RM (habitaciones):**
  - RM_SQUARED = RM²
  - RM_CUBED = RM³
- **Características de interacción:**
  - AGE_LSTAT_INTERACTION = AGE × LSTAT
- **Transformaciones logarítmicas:**
  - CRIM_LOG = log(1 + CRIM)
- **Características de edad de propiedad:**
  - HOUSE_AGE = YRSOLD - YEARBUILT
  - GRLIVAREA_SQUARED = GRLIVAREA²

#### 7. Análisis Exploratorio Completo

- Distribuciones univariables de variables numéricas
- Matriz de correlación con heatmap
- Correlaciones con la variable objetivo

### Construcción y Evaluación del Modelo

#### Fase 1: Preparación de Datos para Modelado

1. Separación de variables independientes (X) y dependiente (y)
2. Identificación automática de la variable objetivo
3. Información sobre características y muestras

#### Fase 2: Configuración de PyCaret

1. Creación de dataframe preparado para PyCaret
2. Normalización de características
3. Selección automática de características (método univariado)
4. División: 80% entrenamiento, 20% prueba

#### Fase 3: Comparación de Modelos

1. Entrenamiento automático de 7 algoritmos:
   - Regresión Lineal (lr)
   - Ridge (ridge)
   - Lasso (lasso)
   - Elastic Net (en)
   - Random Forest (rf)
   - Gradient Boosting (gbr)
   - XGBoost (xgboost)
2. Ordenamiento por R² Score (mejor a peor)
3. Selección automática del mejor modelo

#### Fase 4: Tunizado de Hiperparámetros

1. Optimización del mejor modelo seleccionado
2. Métrica de optimización: R² Score
3. Búsqueda en 10 iteraciones (n_iter=10)

#### Fase 5: Evaluación del Modelo

1. Métricas de desempeño en conjunto de prueba
2. Visualización de resultados

#### Fase 6: Predicción y Validación

1. Predicción en datos de prueba
2. Comparación de predicciones vs valores reales
3. Cálculo de errores por muestra

#### Fase 7: Interpretación de Resultados

1. Importancia de características (SHAP values)
2. Análisis de residuos
3. Visualización de predicciones vs valores reales
4. Distribución de errores

#### Fase 8: Exportación del Modelo

1. Guardado del modelo entrenado
2. Modelo listo para producción y predicciones futuras

### Conclusiones

### Conclusiones

El modelo desarrollado utiliza técnicas de Machine Learning automático con PyCaret para predecir precios de propiedades. Se evaluaron los siguientes algoritmos de regresión:

#### Modelos Evaluados y Seleccionados

**Mejor Modelo**: Se selecciona automáticamente el modelo con mayor R² Score de los siguientes:

- **Regresión Lineal**: Modelo base simple
- **Ridge Regression**: Regularización L2, controla multicolinealidad
- **Lasso Regression**: Regularización L1, realiza selección de características
- **Elastic Net**: Combinación de Ridge y Lasso
- **Random Forest**: Modelo de ensamble basado en árboles
- **Gradient Boosting**: Boosting secuencial de árboles
- **XGBoost**: Gradient Boosting optimizado

#### Métricas de Desempeño

El modelo entrenado se evalúa con las siguientes métricas:

| Métrica      | Descripción                                              |
| ------------ | -------------------------------------------------------- |
| **R² Score** | Coeficiente de determinación (% de varianza explicada)   |
| **RMSE**     | Error cuadrático medio (desviación estándar de errores)  |
| **MAE**      | Error absoluto medio (error promedio en unidades reales) |
| **MAPE**     | Error porcentual absoluto medio (error relativo en %)    |

#### Resultados Esperados

- **R² > 0.85**: Desempeño excelente
- **0.70 < R² ≤ 0.85**: Desempeño bueno
- **0.50 < R² ≤ 0.70**: Desempeño aceptable
- **R² ≤ 0.50**: Requiere mejora

#### Variables Más Significativas

El análisis identifica las características más importantes para predecir el precio:

- Características de ubicación y zona
- Tamaño y superficie del terreno
- Características estructurales de la propiedad
- Edad y condición de la propiedad
- Proximidad a servicios e infraestructura
