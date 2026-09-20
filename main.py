import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Configuración de la página
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Ocupación - PKLot",
    page_icon="🅿️",
    layout="wide",
)

N_MUESTRA = 200  # misma muestra usada en el notebook original
SEED = 42


# ----------------------------------------------------------------------------
# Extracción de características (misma lógica que en el notebook)
# ----------------------------------------------------------------------------
def extraer_features(img: Image.Image, points=None) -> dict:
    """Extrae brightness, std_intensity y edge_mean de una imagen (o de un
    recorte de la imagen definido por 'points', coordenadas normalizadas 0-1)."""
    img = img.convert("RGB")
    w, h = img.size

    if points:
        xs = [p[0] * w for p in points]
        ys = [p[1] * h for p in points]
        box = (min(xs), min(ys), max(xs), max(ys))
        recorte = img.crop(box)
    else:
        recorte = img

    recorte = recorte.resize((40, 40))
    gris = recorte.convert("L")
    arr = np.array(gris, dtype=float)
    bordes = np.array(gris.filter(ImageFilter.FIND_EDGES), dtype=float)

    return {
        "brightness": arr.mean(),
        "std_intensity": arr.std(),
        "edge_mean": bordes.mean(),
    }


# ----------------------------------------------------------------------------
# Descarga de datos + entrenamiento del modelo (se cachea: solo corre una vez)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Descargando muestra del dataset PKLot y entrenando el modelo (puede tardar 1-2 min la primera vez)...")
def entrenar_modelo(n_muestra: int = N_MUESTRA, seed: int = SEED):
    path_samples = hf_hub_download("Voxel51/PKLot", "samples.json", repo_type="dataset")
    with open(path_samples) as f:
        samples = json.load(f)
    lista_samples = samples["samples"] if isinstance(samples, dict) else samples

    rows_img = []
    for s in lista_samples:
        rows_img.append(
            {
                "filepath": s["filepath"],
                "source": s.get("source"),
                "weather": (s.get("weather") or {}).get("label"),
                "parking_spaces": (s.get("parking_spaces") or {}).get("polylines", []),
            }
        )
    df_img = pd.DataFrame(rows_img)

    muestra_imgs = df_img.sample(n=n_muestra, random_state=seed).reset_index(drop=True)

    filas = []
    for _, row in muestra_imgs.iterrows():
        try:
            ruta_local = hf_hub_download("Voxel51/PKLot", row["filepath"], repo_type="dataset")
            img = Image.open(ruta_local)
        except Exception:
            continue

        for p in row["parking_spaces"]:
            status = p.get("occupancy_status")
            if status not in ("occupied", "not occupied"):
                continue
            points = p.get("points", [[]])[0]
            if not points or len(points) < 3:
                continue
            try:
                feats = extraer_features(img, points)
            except Exception:
                continue
            feats["occupied"] = 1 if status == "occupied" else 0
            feats["weather"] = row["weather"]
            feats["source"] = row["source"]
            filas.append(feats)

    df_features = pd.DataFrame(filas)

    X = df_features[["brightness", "std_intensity", "edge_mean"]]
    y = df_features["occupied"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    reporte = classification_report(
        y_test, y_pred, target_names=["Vacío", "Ocupado"], output_dict=True
    )

    return modelo, acc, cm, reporte, df_features


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🅿️ Predicción de Ocupación de Estacionamientos")
st.caption("Proyecto Final BI · Dataset PKLot (Voxel51) · Regresión Logística")

tab_predecir, tab_modelo, tab_proyecto = st.tabs(
    ["🔮 Predecir", "📊 Modelo y Resultados", "ℹ️ Sobre el proyecto"]
)

modelo, acc, cm, reporte, df_features = entrenar_modelo()

# --- Tab 1: Predecir ---------------------------------------------------------
with tab_predecir:
    st.subheader("Sube una foto de un cajón de estacionamiento")
    st.write(
        "Idealmente una imagen ya recortada mostrando solo un cajón "
        "(el modelo fue entrenado con recortes de 40x40 px)."
    )

    archivo = st.file_uploader("Imagen (jpg/png)", type=["jpg", "jpeg", "png"])

    if archivo is not None:
        img = Image.open(archivo)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(img, caption="Imagen subida", use_container_width=True)

        with col2:
            feats = extraer_features(img)
            X_nuevo = pd.DataFrame([feats])[["brightness", "std_intensity", "edge_mean"]]
            pred = modelo.predict(X_nuevo)[0]
            proba = modelo.predict_proba(X_nuevo)[0]

            if pred == 1:
                st.error(f"### Predicción: OCUPADO 🚗  ({proba[1]*100:.1f}% confianza)")
            else:
                st.success(f"### Predicción: VACÍO ✅  ({proba[0]*100:.1f}% confianza)")

            st.write("**Características extraídas:**")
            st.dataframe(
                pd.DataFrame([feats]).round(2),
                hide_index=True,
                use_container_width=True,
            )

# --- Tab 2: Modelo y resultados ---------------------------------------------
with tab_modelo:
    st.subheader("Desempeño del modelo")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Accuracy (test)", f"{acc*100:.1f}%")
        st.write("**Reporte de clasificación:**")
        st.dataframe(pd.DataFrame(reporte).transpose().round(3), use_container_width=True)

    with c2:
        st.write("**Matriz de confusión:**")
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Vacío", "Ocupado"]
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig)

    st.subheader("Coeficientes del modelo")
    coefs = pd.Series(
        modelo.coef_[0], index=["brightness", "std_intensity", "edge_mean"]
    ).sort_values()
    st.bar_chart(coefs)
    st.caption(
        "Coeficiente negativo en 'brightness' → entre más brillante el cajón, "
        "menos probable que esté ocupado (el pavimento vacío refleja más luz)."
    )

# --- Tab 3: Sobre el proyecto -------------------------------------------------
with tab_proyecto:
    st.subheader("Contexto del proyecto")
    st.markdown(
        """
Trabajamos con el dataset **PKLot**, que contiene miles de imágenes de tres
estacionamientos universitarios, cada una con sus cajones individuales
etiquetados como ocupado o vacío. El objetivo es construir un modelo de
clasificación supervisada que, a partir de características simples de una
foto, prediga si un cajón está ocupado o vacío.

**Propuesta de negocio:** en vez de instalar un sensor físico por cajón
(caro y complicado de mantener), se propone un sistema basado en cámaras y
un modelo simple de clasificación de imágenes que detecte la disponibilidad
de estacionamiento a una fracción del costo, habilitando aplicaciones de
disponibilidad en tiempo real y mejor gestión del espacio.

**Características usadas por el modelo:**
- `brightness`: brillo promedio del recorte (el suelo vacío suele ser más claro).
- `std_intensity`: variación de intensidad (un coche agrega más variación/textura).
- `edge_mean`: intensidad promedio de bordes detectados (un coche tiene más bordes que pavimento liso).

**Modelo:** Regresión Logística, elegida por ser simple, interpretable y
adecuada para clasificación binaria.
"""
    )
    st.info(
        f"Modelo entrenado en esta sesión con una muestra de {N_MUESTRA} imágenes "
        f"del dataset, resultando en un accuracy de {acc*100:.1f}% sobre el conjunto de prueba."
    )
