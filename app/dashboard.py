import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

# Descargar stopwords
nltk.download('stopwords')
spanish_stopwords = set(stopwords.words('spanish'))

# Configuración de la página
st.set_page_config(page_title="Dashboard Bavaria", layout="wide", initial_sidebar_state="expanded")

# Sidebar para navegación
st.sidebar.title("Navegación")
page = st.sidebar.selectbox(
    "Selecciona una página:",
    ["Análisis General", "Nube de Palabras y N-Gramas", "Análisis de Sentimientos"]
)

# Cargar la imagen del logotipo
image_path = "data/logo_bavaria.png"
try:
    logo = Image.open(image_path)
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(logo, use_container_width=True)
    with col2:
        st.title("Dashboard Bavaria")
        st.write("Bienvenido al sistema de análisis de datos.")
        st.write("Última actualización: 20 de noviembre de 2024")
        st.write("Status del sistema: 🟢 En línea")
except FileNotFoundError:
    st.error(f"No se encontró la imagen en la ruta {image_path}. Por favor, verifica la ubicación del archivo.")

# Función de limpieza de texto
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^\w\s]", "", text)  # Remover puntuación
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r"\d+", "", text)  # Remover números
    text = text.strip()
    
    # Remover stopwords en español
    words = text.split()
    filtered_words = [word for word in words if word not in spanish_stopwords]
    return " ".join(filtered_words)

# Generar n-gramas
def generate_ngrams(corpus, n):
    corpus = corpus[corpus.str.strip() != ""].dropna()
    if corpus.empty:
        st.warning("El corpus está vacío después de la limpieza. No se pueden generar n-gramas.")
        return []
    stop_words_list = list(spanish_stopwords)
    try:
        vec = CountVectorizer(ngram_range=(n, n), stop_words=stop_words_list).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:10]
    except ValueError:
        st.warning("No se pudieron generar n-gramas. El corpus podría contener solo stopwords o estar vacío.")
        return []

# Cargar datos
@st.cache_data
def load_data():
    try:
        comments = pd.read_csv("data/sentimientos.csv", delimiter=',')
        publications = pd.read_csv("data/posts_senti.csv", delimiter=',')
        publications = publications.fillna(0)

        comments['date'] = pd.to_datetime(comments['date'], errors='coerce')
        publications['date'] = pd.to_datetime(publications['date'], errors='coerce')

        return comments, publications
    except Exception as e:
        st.error(f"Error al cargar los archivos: {e}")
        st.stop()

# Función para aplicar filtros
def apply_filters_v2(publications_df, comments_df, source=None, date_range=None, topic=None):
    # Combinar publicaciones con comentarios para obtener tópicos
    merged_data = pd.merge(
        comments_df,
        publications_df[['link', 'Nombre', 'source']],
        left_on="Link",
        right_on="link",
        how="inner"
    )

    # Filtrar por fuente
    if source and source != "Todas":
        merged_data = merged_data[merged_data["source_x"] == source]

    # Filtrar por rango de fechas
    if date_range and len(date_range) == 2 and not pd.isna(date_range[0]) and not pd.isna(date_range[1]):
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        merged_data = merged_data[
            (merged_data["date_X"] >= start_date) &
            (merged_data["date_X"] <= end_date)
        ]

    # Filtrar por tópico
    if topic and topic != "Todos":
        merged_data = merged_data[merged_data["Nombre"] == topic]

    return merged_data

# Cargar los datos
comments_df, publications_df = load_data()

# Página: Análisis General
if page == "Análisis General":
    st.title("Análisis General de Datos")

    # Filtros
    st.sidebar.header("Filtros")
    selected_source = st.sidebar.selectbox("Seleccionar fuente:", ["Todas"] + list(publications_df["source"].unique()))
    selected_date_range = st.sidebar.date_input("Seleccionar rango de fechas:", [])
    selected_topic = st.sidebar.selectbox("Seleccionar tópico:", ["Todos"] + list(publications_df["Nombre"].unique()))

    # Aplicar filtros
    filtered_data = apply_filters_v2(
        publications_df,
        comments_df,
        source=selected_source,
        date_range=selected_date_range,
        topic=selected_topic
    )

    # Verificar si hay datos después de los filtros
    if filtered_data.empty:
        st.warning("No hay datos disponibles para los filtros seleccionados.")
    else:
        # Gráfico: Distribución de Comentarios por Día de la Semana
        st.header("Distribución de Comentarios por Día de la Semana")
        filtered_data["day_of_week"] = filtered_data["date"].dt.day_name()
        comments_by_day = filtered_data["day_of_week"].value_counts()
        fig_day = px.bar(
            comments_by_day,
            x=comments_by_day.index,
            y=comments_by_day.values,
            title="Distribución de Comentarios por Día de la Semana",
            labels={"x": "Día de la Semana", "y": "Número de Comentarios"},
            color_discrete_sequence=["#FFA500"]
        )
        st.plotly_chart(fig_day, use_container_width=True)

        # Gráfico: Cantidad de Comentarios por Mes
        st.header("Cantidad de Comentarios por Mes")
        filtered_data["month"] = filtered_data["date"].dt.to_period("M")
        comments_by_month = filtered_data["month"].value_counts().sort_index()
        fig_month = px.line(
            comments_by_month,
            x=comments_by_month.index.astype(str),
            y=comments_by_month.values,
            title="Cantidad de Comentarios por Mes",
            labels={"x": "Mes", "y": "Número de Comentarios"},
            markers=True,
            color_discrete_sequence=["#0000FF"]
        )
        st.plotly_chart(fig_month, use_container_width=True)

        # Gráfico: Usuarios Más Activos
        st.header("Top 10 Usuarios Más Activos")
        top_users = filtered_data["author"].value_counts().head(10)
        fig_users = px.bar(
            top_users,
            x=top_users.values,
            y=top_users.index,
            title="Top 10 Usuarios Más Activos",
            labels={"x": "Número de Comentarios", "y": "Usuario"},
            orientation="h",
            color_discrete_sequence=["#228B22"]
        )
        st.plotly_chart(fig_users, use_container_width=True)

        # Gráfico: Cantidad de Comentarios por Publicación
        st.header("Cantidad de Comentarios por Publicación")
        comments_per_post = filtered_data.groupby("link").size().reset_index(name="Número de Comentarios")
        fig_comments_per_post = px.bar(
            comments_per_post.nlargest(10, "Número de Comentarios"),
            x="link",
            y="Número de Comentarios",
            title="Top 10 Publicaciones con Más Comentarios",
            labels={"link": "Publicación (Link)", "Número de Comentarios": "Cantidad"},
            color_discrete_sequence=["#FF6347"]
        )
        st.plotly_chart(fig_comments_per_post, use_container_width=True)

        # Gráfico: Promedio de Sentimiento por Publicación
        st.header("Promedio de Sentimiento por Publicación")
        sentiment_avg_per_post = filtered_data.groupby("link")["sentimiento_bert"].value_counts(normalize=True).unstack(fill_value=0)
        sentiment_avg_per_post = sentiment_avg_per_post.rename(columns={"positivo": "Positivo", "negativo": "Negativo", "neutro": "Neutro"})
        fig_sentiment_avg = px.bar(
            sentiment_avg_per_post.head(10),
            barmode="group",
            title="Distribución Promedio de Sentimientos por Publicación",
            labels={"value": "Proporción", "link": "Publicación (Link)"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_sentiment_avg, use_container_width=True)

# Continuar con Página 2 y 3...
# Página 2: Nube de Palabras y N-Gramas
elif page == "Nube de Palabras y N-Gramas":
    st.title("Análisis de Nube de Palabras y N-Gramas")

    # Selección de tópico para filtrar los comentarios
    selected_topic = st.selectbox("Selecciona un tópico:", publications_df["Nombre"].unique())
    topic_links = publications_df[publications_df["Nombre"] == selected_topic]["link"]
    filtered_comments = comments_df[comments_df["Link"].isin(topic_links)]["cleaned_comment"].dropna()
    corpus = filtered_comments.apply(clean_text)

    # Ruta de la imagen para la máscara
    mask_image_path = "data/cervezax3.png"

    try:
        # Generar la nube de palabras
        mask_image = np.array(Image.open(mask_image_path))
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            mask=mask_image,
            contour_color="black",
            contour_width=1
        ).generate(" ".join(corpus))

        # Mostrar la nube de palabras
        st.header("Nube de Palabras")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    except FileNotFoundError:
        st.error(f"No se encontró la imagen en la ruta {mask_image_path}. Por favor verifica la ubicación del archivo.")

    # Generar y mostrar n-gramas
    st.header("N-Gramas Más Frecuentes")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Bigramas")
        bigrams = generate_ngrams(corpus, 2)
        st.write(pd.DataFrame(bigrams, columns=["Bigrama", "Frecuencia"]))
    with col2:
        st.subheader("Trigramas")
        trigrams = generate_ngrams(corpus, 3)
        st.write(pd.DataFrame(trigrams, columns=["Trigrama", "Frecuencia"]))
    with col3:
        st.subheader("Cuadrigramas")
        fourgrams = generate_ngrams(corpus, 4)
        st.write(pd.DataFrame(fourgrams, columns=["Cuadrigrama", "Frecuencia"]))

# Página 3: Análisis de Sentimientos
elif page == "Análisis de Sentimientos":
    st.title("Análisis de Sentimientos")

    # Unir comentarios con publicaciones para obtener información de tópicos
    merged_data = pd.merge(
        comments_df,
        publications_df,
        left_on="Link",
        right_on="link",
        how="inner"
    )

    # Gráfico: Distribución de Sentimientos por Fuente
    st.header("Distribución de Sentimientos por Fuente")
    sentiment_by_source = merged_data.groupby(["source_x", "sentimiento_bert"]).size().reset_index(name="Conteo")
    fig_sentiment_source = px.bar(
        sentiment_by_source,
        x="source_x",
        y="Conteo",
        color="sentimiento_bert",
        barmode="group",
        title="Distribución de Sentimientos por Fuente",
        labels={"source_x": "Fuente", "Conteo": "Cantidad", "sentimiento_bert": "Sentimiento"},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_sentiment_source, use_container_width=True)

    # Gráfico: Distribución General de Sentimientos
    st.header("Distribución General de Sentimientos")
    general_sentiment = merged_data["sentimiento_bert"].value_counts().reset_index()
    general_sentiment.columns = ["Sentimiento", "Conteo"]
    fig_general_sentiment = px.bar(
        general_sentiment,
        x="Sentimiento",
        y="Conteo",
        title="Distribución General de Sentimientos",
        color="Sentimiento",
        text="Conteo",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_general_sentiment, use_container_width=True)

    # Gráfico: Sentimientos por Tópico
    st.header("Distribución de Sentimientos por Tópico")
    sentiment_by_topic = merged_data.groupby(["Nombre", "sentimiento_bert"]).size().reset_index(name="Conteo")
    fig_sentiment_topic = px.bar(
        sentiment_by_topic,
        x="Nombre",
        y="Conteo",
        color="sentimiento_bert",
        barmode="group",
        title="Distribución de Sentimientos por Tópico",
        labels={"Nombre": "Tópico", "Conteo": "Cantidad", "sentimiento_bert": "Sentimiento"},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_sentiment_topic, use_container_width=True)

    # Tabla: Publicaciones y Cantidad de Sentimientos
    
    # Generar la tabla interactiva
    st.header("Tabla de Publicaciones con Sentimientos Asociados")
    publication_sentiments = merged_data.groupby(["link", "sentimiento_bert"]).size().unstack(fill_value=0).reset_index()
    publication_sentiments.columns.name = None  # Eliminar el nombre del índice
    publication_sentiments = publication_sentiments.rename(columns={
        "link": "Publicación",
        "positivo": "Positivo",
        "negativo": "Negativo",
        "neutro": "Neutro"
    })

    # Crear opciones de configuración para AgGrid
    gb = GridOptionsBuilder.from_dataframe(publication_sentiments)
    gb.configure_pagination(paginationAutoPageSize=True)  # Habilitar paginación
    gb.configure_default_column(editable=False, groupable=True)  # Hacer columnas no editables y agrupables
    gb.configure_column("Publicación", sortable=True, filter=True)  # Habilitar orden y filtro en "Publicación"
    gb.configure_column("Positivo", sortable=True, filter=True)  # Habilitar orden y filtro en "Positivo"
    gb.configure_column("Negativo", sortable=True, filter=True)  # Habilitar orden y filtro en "Negativo"
    gb.configure_column("Neutro", sortable=True, filter=True)  # Habilitar orden y filtro en "Neutro"

    grid_options = gb.build()

    # Mostrar la tabla interactiva
    AgGrid(
        publication_sentiments,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        theme="streamlit",  # Estilo visual
        height=400,
        fit_columns_on_grid_load=True
    )

