import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageFilter

# ----------------------------------------------------------------------------
# Configuración de la página
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard - Ocupación de Estacionamientos (PKLot)",
    page_icon="🅿️",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 12px 10px 8px 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Extracción de características (para imágenes que suba el usuario en vivo)
# ----------------------------------------------------------------------------
def extraer_features(img: Image.Image, points=None) -> dict:
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
# Carga del modelo y los datos ya procesados (rápido: nada se descarga aquí)
# ----------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo.pkl")


@st.cache_data
def cargar_datos():
    return pd.read_csv("df_features.csv")


try:
    modelo = cargar_modelo()
    df_features = cargar_datos()
except FileNotFoundError:
    st.error(
        "Faltan los archivos **modelo.pkl** y/o **df_features.csv** en la raíz del repo. "
        "Corre `entrenar_y_exportar.py` en Colab y sube esos dos archivos junto a main.py."
    )
    st.stop()

df_test = df_features[df_features["conjunto"] == "test"].copy()
y_test = df_test["occupied"].values
y_proba_test = df_test["proba_ocupado"].values
weather_test = df_test["weather"]

# ----------------------------------------------------------------------------
# Sidebar: filtros globales (afectan Resumen y Explorar Datos)
# ----------------------------------------------------------------------------
st.sidebar.header("🎛️ Filtros del dashboard")

climas_disponibles = sorted(df_features["weather"].dropna().unique())
fuentes_disponibles = sorted(df_features["source"].dropna().unique())

climas_sel = st.sidebar.multiselect("Clima", climas_disponibles, default=climas_disponibles)
fuentes_sel = st.sidebar.multiselect(
    "Estacionamiento", fuentes_disponibles, default=fuentes_disponibles
)

df_filtrado = df_features[
    df_features["weather"].isin(climas_sel) & df_features["source"].isin(fuentes_sel)
]

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Modelo entrenado sobre una muestra del dataset PKLot "
    f"({len(df_features)} cajones analizados en total)."
)

# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
st.title("🅿️ Dashboard: Ocupación de Estacionamientos")
st.caption("Proyecto Final BI · Dataset PKLot (Voxel51) · Regresión Logística")

tab_resumen, tab_explorar, tab_predecir, tab_modelo, tab_proyecto = st.tabs(
    ["📈 Resumen", "🔎 Explorar Datos", "🔮 Predecir", "🤖 Modelo", "ℹ️ Sobre el proyecto"]
)

# ----------------------------------------------------------------------------
# Tab: Resumen
# ----------------------------------------------------------------------------
with tab_resumen:
    if df_filtrado.empty:
        st.warning("No hay datos para los filtros seleccionados. Ajusta los filtros en la barra lateral.")
    else:
        acc_global = (y_proba_test >= 0.5).astype(int)
        acc_global = (acc_global == y_test).mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cajones analizados", f"{len(df_filtrado):,}")
        c2.metric("Tasa de ocupación", f"{df_filtrado['occupied'].mean()*100:.1f}%")
        c3.metric("Estacionamientos", df_filtrado["source"].nunique())
        c4.metric("Accuracy del modelo", f"{acc_global*100:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            ocupacion_source = df_filtrado.groupby("source")["occupied"].mean().reset_index()
            fig = px.bar(
                ocupacion_source,
                x="source",
                y="occupied",
                labels={"source": "Estacionamiento", "occupied": "Tasa de ocupación"},
                title="Ocupación promedio por estacionamiento",
                color="source",
            )
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            dist_clima = df_filtrado["weather"].value_counts().reset_index()
            dist_clima.columns = ["weather", "count"]
            fig2 = px.pie(
                dist_clima, names="weather", values="count",
                title="Distribución de cajones por clima", hole=0.4,
            )
            st.plotly_chart(fig2, use_container_width=True)

        ocupacion_clima = df_filtrado.groupby("weather")["occupied"].mean().reset_index()
        fig3 = px.bar(
            ocupacion_clima,
            x="weather",
            y="occupied",
            labels={"weather": "Clima", "occupied": "Tasa de ocupación"},
            title="Ocupación promedio por clima",
            color="weather",
        )
        fig3.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab: Explorar Datos
# ----------------------------------------------------------------------------
with tab_explorar:
    st.subheader("Explora las características extraídas de cada cajón")

    if df_filtrado.empty:
        st.warning("No hay datos para los filtros seleccionados.")
    else:
        col_a, col_b = st.columns(2)
        feature_x = col_a.selectbox("Eje X", ["brightness", "std_intensity", "edge_mean"], index=0)
        feature_y = col_b.selectbox("Eje Y", ["brightness", "std_intensity", "edge_mean"], index=2)

        fig_scatter = px.scatter(
            df_filtrado,
            x=feature_x,
            y=feature_y,
            color=df_filtrado["occupied"].map({0: "Vacío", 1: "Ocupado"}),
            labels={"color": "Estado real"},
            opacity=0.7,
            title=f"{feature_x} vs {feature_y}",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        feature_hist = st.selectbox(
            "Variable para el histograma", ["brightness", "std_intensity", "edge_mean"], index=0
        )
        fig_hist = px.histogram(
            df_filtrado,
            x=feature_hist,
            color=df_filtrado["occupied"].map({0: "Vacío", 1: "Ocupado"}),
            barmode="overlay",
            nbins=30,
            labels={"color": "Estado real"},
            title=f"Distribución de {feature_hist}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        if st.checkbox("Mostrar tabla de datos filtrados"):
            st.dataframe(
                df_filtrado[
                    ["source", "weather", "brightness", "std_intensity", "edge_mean", "occupied", "pred", "correcto"]
                ].rename(columns={"occupied": "real", "pred": "predicho"}),
                use_container_width=True,
            )

# ----------------------------------------------------------------------------
# Tab: Predecir
# ----------------------------------------------------------------------------
with tab_predecir:
    st.subheader("Prueba el modelo")
    modo = st.radio(
        "¿Cómo quieres probar el modelo?",
        ["Subir una imagen", "Ajustar valores manualmente", "Ejemplo aleatorio del dataset"],
        horizontal=True,
    )

    feats = None

    if modo == "Subir una imagen":
        st.caption("Idealmente una imagen ya recortada mostrando solo un cajón.")
        archivo = st.file_uploader("Imagen (jpg/png)", type=["jpg", "jpeg", "png"])
        if archivo is not None:
            img = Image.open(archivo)
            st.image(img, caption="Imagen subida", width=250)
            feats = extraer_features(img)

    elif modo == "Ajustar valores manualmente":
        st.caption("Simula las características de un cajón moviendo los sliders:")
        b_min, b_max = float(df_features["brightness"].min()), float(df_features["brightness"].max())
        s_min, s_max = float(df_features["std_intensity"].min()), float(df_features["std_intensity"].max())
        e_min, e_max = float(df_features["edge_mean"].min()), float(df_features["edge_mean"].max())

        brightness = st.slider("Brightness", b_min, b_max, float(df_features["brightness"].mean()))
        std_intensity = st.slider("Std Intensity", s_min, s_max, float(df_features["std_intensity"].mean()))
        edge_mean = st.slider("Edge Mean", e_min, e_max, float(df_features["edge_mean"].mean()))
        feats = {"brightness": brightness, "std_intensity": std_intensity, "edge_mean": edge_mean}

    else:  # Ejemplo aleatorio del dataset
        if st.button("🎲 Elegir un cajón al azar"):
            st.session_state["ejemplo"] = df_filtrado.sample(1).iloc[0]

        if "ejemplo" in st.session_state:
            fila = st.session_state["ejemplo"]
            st.write(
                f"**Estacionamiento:** {fila['source']} · **Clima:** {fila['weather']} · "
                f"**Estado real:** {'Ocupado' if fila['occupied'] == 1 else 'Vacío'}"
            )
            feats = {
                "brightness": fila["brightness"],
                "std_intensity": fila["std_intensity"],
                "edge_mean": fila["edge_mean"],
            }
        else:
            st.info("Presiona el botón para elegir un cajón real del dataset.")

    if feats is not None:
        X_nuevo = pd.DataFrame([feats])[["brightness", "std_intensity", "edge_mean"]]
        pred = modelo.predict(X_nuevo)[0]
        proba = modelo.predict_proba(X_nuevo)[0]

        col1, col2 = st.columns([1, 2])
        with col1:
            if pred == 1:
                st.error("### 🚗 OCUPADO")
            else:
                st.success("### ✅ VACÍO")
        with col2:
            st.write("Probabilidad de estar ocupado:")
            st.progress(float(proba[1]))
            st.caption(f"{proba[1]*100:.1f}%")

        st.write("**Características usadas:**")
        st.dataframe(pd.DataFrame([feats]).round(2), hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab: Modelo
# ----------------------------------------------------------------------------
with tab_modelo:
    st.subheader("Desempeño del modelo")

    threshold = st.slider(
        "Umbral de decisión (probabilidad mínima para clasificar como 'Ocupado')",
        0.0, 1.0, 0.5, 0.01,
    )
    y_pred_umbral = (y_proba_test >= threshold).astype(int)
    acc = (y_pred_umbral == y_test).mean()

    # Matriz de confusión manual (sin depender de sklearn.metrics aquí)
    vp = int(((y_pred_umbral == 1) & (y_test == 1)).sum())
    vn = int(((y_pred_umbral == 0) & (y_test == 0)).sum())
    fp = int(((y_pred_umbral == 1) & (y_test == 0)).sum())
    fn = int(((y_pred_umbral == 0) & (y_test == 1)).sum())
    cm = np.array([[vn, fp], [fn, vp]])

    precision_ocupado = vp / (vp + fp) if (vp + fp) > 0 else 0
    recall_ocupado = vp / (vp + fn) if (vp + fn) > 0 else 0

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy (test)", f"{acc*100:.1f}%")
        st.metric("Precision (Ocupado)", f"{precision_ocupado*100:.1f}%")
        st.metric("Recall (Ocupado)", f"{recall_ocupado*100:.1f}%")

    with c2:
        st.write("**Matriz de confusión:**")
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Vacío", "Ocupado"],
            y=["Vacío", "Ocupado"],
            labels=dict(x="Predicho", y="Real", color="Cajones"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Coeficientes del modelo")
    coefs = pd.Series(
        modelo.coef_[0], index=["brightness", "std_intensity", "edge_mean"]
    ).sort_values()
    fig_coef = px.bar(
        coefs, orientation="h",
        labels={"index": "Característica", "value": "Coeficiente"},
        title="Peso de cada característica en la predicción",
    )
    fig_coef.update_layout(showlegend=False)
    st.plotly_chart(fig_coef, use_container_width=True)
    st.caption(
        "Coeficiente negativo en 'brightness' → entre más brillante el cajón, "
        "menos probable que esté ocupado (el pavimento vacío refleja más luz)."
    )

    st.subheader("Accuracy por condición climática (con el umbral seleccionado)")
    df_test_clima = pd.DataFrame(
        {"weather": weather_test.values, "real": y_test, "pred": y_pred_umbral}
    )
    acc_clima = (
        df_test_clima.groupby("weather")
        .apply(lambda g: (g["real"] == g["pred"]).mean(), include_groups=False)
        .reset_index()
    )
    acc_clima.columns = ["weather", "accuracy"]
    fig_clima = px.bar(
        acc_clima, x="weather", y="accuracy", title="Accuracy por clima", color="weather"
    )
    fig_clima.update_layout(showlegend=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig_clima, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab: Sobre el proyecto
# ----------------------------------------------------------------------------
with tab_proyecto:
    st.subheader("Contexto del proyecto")
    st.markdown(
        """
Trabajamos con el dataset **PKLot**, que contiene miles de imágenes de tres
estacionamientos universitarios, cada una con sus cajones individuales
etiquetados como ocupado o vacío. El objetivo es construir un modelo de
clasificación supervisada que, a partir de características simples de una
foto, prediga si un cajón está ocupado o vacío.

**Características usadas por el modelo:**
- `brightness`: brillo promedio del recorte (el suelo vacío suele ser más claro).
- `std_intensity`: variación de intensidad (un coche agrega más variación/textura).
- `edge_mean`: intensidad promedio de bordes detectados (un coche tiene más bordes que pavimento liso).

**Modelo:** Regresión Logística, elegida por ser simple, interpretable y
adecuada para clasificación binaria.
"""
    )

    with st.expander("💡 Propuesta de negocio"):
        st.write(
            "En vez de instalar un sensor físico por cajón (caro y complicado de "
            "mantener), se propone un sistema basado en cámaras y un modelo simple "
            "de clasificación de imágenes que detecte la disponibilidad de "
            "estacionamiento a una fracción del costo, habilitando aplicaciones de "
            "disponibilidad en tiempo real y mejor gestión del espacio."
        )

    with st.expander("⚠️ Limitaciones detectadas"):
        st.write(
            "El modelo falla un poco más con falsos negativos (dice que un cajón "
            "está vacío cuando en realidad está ocupado), lo cual es justo el error "
            "que más nos interesa evitar para el negocio. También, contra lo "
            "esperado, el modelo tiene más errores con clima soleado que con "
            "lluvia, probablemente porque el sol genera sombras que confunden la "
            "característica de brillo."
        )

    st.info(f"Modelo entrenado sobre una muestra de {len(df_features)} cajones del dataset PKLot.")
