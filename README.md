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

...

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

Proveer una herramienta que enbase al entrenamiento de modelos de AI, pueda

---

### Preparación de Datos:

1. Limpieza de Datos y Análisis de Datos Faltantes.
2. Análisis y Tratamiento de Valores Atípicos.
3. Derivación de Columnas Categóricas.
4. Análisis Univariable.
5. Análisis Bivariable.
6. Análisis Multivariable.

### Construcción y Evaluación del Modelo

1. División de datos de entrenamiento y prueba.
2. Escalado de Características - StandardScaler.
3. Ingeniería y Selección de Características usando RFE y el Factor de Inflación de Varianza.
4. Regresión Lineal usando PyCaret.
5. Modelos de Regularización Ridge, Lasso y ElasticNet.
6. Análisis de Residuos.
7. Evaluación y Valoración del Modelo.
8. Predicción.
9. Conclusión y Análisis Final.

### Conclusiones

### Conclusions

R2_Score for Ridge regresion....
R2_Score for Lasso regresion....
R2_Score for ElasticNet regresion....

#### Ridge Regression (Segun PyCaret)

- **Optimal Lambda Value:** ##
- **R2 Score Train:** 0.##
- **R2 Test Score:** 0.##
- **RMSE Test:** 0.##

#### Lasso Regression (Segun PyCaret)

- **Optimal Lambda Value:** 0.####
- **R2 Score Train:** 0.##
- **R2 Test Score:** 0.##
- **RMSE Test:** 0.##

#### ElasticNet Regression (Segun PyCaret)

- **Optimal Lambda Value:** 0.####
- **R2 Score Train:** 0.##
- **R2 Test Score:** 0.##
- **RMSE Test:** 0.##

#### Las Variables Más Significativas Son:

- ...
- ...
- ...
- ...
