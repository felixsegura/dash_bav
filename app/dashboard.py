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
nltk.download('stopwords')

# Cargar stopwords en español
spanish_stopwords = set(stopwords.words('spanish'))

# Configuración de la página
st.set_page_config(page_title="Dashboard Bavaria", layout="wide", initial_sidebar_state="expanded")

# Sidebar para navegación
st.sidebar.title("Navegación")
page = st.sidebar.selectbox(
    "Selecciona una página:",
    ["Análisis General", "Nube de Palabras y N-Gramas", "Análisis de Sentimientos"]
)

# Función de limpieza de texto
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^\w\s]", "", text)  # Remover puntuación
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r"\d+", "", text)  # Remover números
    text = text.strip()
    return text

# Generar n-gramas
def generate_ngrams(corpus, n):
    # Limpiar corpus y verificar si contiene texto válido
    corpus = corpus[corpus.str.strip() != ""].dropna()
    
    if corpus.empty:
        st.warning("El corpus está vacío después de la limpieza. No se pueden generar n-gramas.")
        return []

    stop_words_list = list(spanish_stopwords)

    try:
        vec = CountVectorizer(
            ngram_range=(n, n),
            stop_words=stop_words_list  # Lista de stopwords
        ).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:10]
    except ValueError:
        st.warning(f"No se pudieron generar n-gramas. El corpus podría contener solo stopwords o estar vacío.")
        return []

# Cargar datos
@st.cache_data
def load_data():
    try:
        # Cargar comentarios y publicaciones
        comments = pd.read_csv("data/sentimientos.csv", delimiter=',')
        publications = pd.read_csv("data/posts_senti.csv", delimiter=',')
        publications=publications.fillna(0)
        
        # Asegurar formato de la columna de fecha
        comments['date'] = pd.to_datetime(comments['date'], errors='coerce')
        publications['date'] = pd.to_datetime(publications['date'], errors='coerce')
        
        return comments, publications
    except Exception as e:
        st.error(f"Error al cargar los archivos: {e}")
        st.stop()

# Cargar los datos
comments_df, publications_df = load_data()

# Página 1: Análisis General
if page == "Análisis General":
    st.title("Análisis General de Datos")

    # Filtros
    st.sidebar.header("Filtros")
    selected_source = st.sidebar.selectbox("Seleccionar fuente:", ["Todas"] + list(publications_df["source"].unique()))
    selected_date = st.sidebar.date_input("Seleccionar rango de fechas:", [])
    selected_topic = st.sidebar.selectbox("Seleccionar tópico:", ["Todos"] + list(publications_df["Nombre"].unique()))

    # Aplicar filtros a publicaciones y comentarios
    filtered_posts = publications_df.copy()
    filtered_comments = comments_df.copy()

    if selected_source != "Todas":
        filtered_posts = filtered_posts[filtered_posts["source"] == selected_source]
        filtered_comments = filtered_comments[filtered_comments["source"] == selected_source]

    if selected_date:
        filtered_posts = filtered_posts[
            (filtered_posts["date"] >= pd.to_datetime(selected_date[0])) &
            (filtered_posts["date"] <= pd.to_datetime(selected_date[1]))
        ]
        filtered_comments = filtered_comments[
            (filtered_comments["date"] >= pd.to_datetime(selected_date[0])) &
            (filtered_comments["date"] <= pd.to_datetime(selected_date[1]))
        ]

    if selected_topic != "Todos":
        filtered_posts = filtered_posts[filtered_posts["Nombre"] == selected_topic]

    # Unir publicaciones y comentarios por el link
    merged_data = pd.merge(
        filtered_comments, filtered_posts,
        left_on="Link", right_on="link",
        how="inner"
    )

    # Gráfico: Distribución de Comentarios por Día de la Semana
    st.header("Distribución de Comentarios por Día de la Semana")
    filtered_comments["day_of_week"] = filtered_comments["date"].dt.day_name()
    comments_by_day = filtered_comments["day_of_week"].value_counts()
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
    filtered_comments["month"] = filtered_comments["date"].dt.to_period("M")
    comments_by_month = filtered_comments["month"].value_counts().sort_index()
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
    top_users = filtered_comments["author"].value_counts().head(10)
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

    # Gráfico: Publicaciones y Comentarios por Fuente
    st.header("Publicaciones y Comentarios por Fuente")
    source_analysis = pd.DataFrame({
        "Tipo": ["Publicaciones", "Comentarios"],
        "Instagram": [
            len(filtered_posts[filtered_posts["source"] == "instagram"]),
            len(filtered_comments[filtered_comments["source"] == "instagram"])
        ],
        "Twitter": [
            len(filtered_posts[filtered_posts["source"] == "twitter"]),
            len(filtered_comments[filtered_comments["source"] == "twitter"])
        ]
    })
    fig_source_analysis = px.bar(
        source_analysis,
        x="Tipo",
        y=["Instagram", "Twitter"],
        title="Publicaciones y Comentarios por Fuente",
        barmode="group",
        labels={"value": "Cantidad", "variable": "Fuente"},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    st.plotly_chart(fig_source_analysis, use_container_width=True)

    # Gráfico: Cantidad de Comentarios por Publicación
    st.header("Cantidad de Comentarios por Publicación")
    comments_per_post = merged_data.groupby("link").size().reset_index(name="Número de Comentarios")
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
    sentiment_avg_per_post = merged_data.groupby("link")["sentimiento_bert"].value_counts(normalize=True).unstack(fill_value=0)
    sentiment_avg_per_post = sentiment_avg_per_post.rename(columns={"positivo": "Positivo", "negativo": "Negativo", "neutro": "Neutro"})
    fig_sentiment_avg = px.bar(
        sentiment_avg_per_post.head(10),
        barmode="group",
        title="Distribución Promedio de Sentimientos por Publicación",
        labels={"value": "Proporción", "link": "Publicación (Link)"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_sentiment_avg, use_container_width=True)

    # Tabla: Publicaciones con Más Interacciones
    st.header("Top Publicaciones con Más Interacciones")
    top_interactive_posts = comments_per_post.merge(filtered_posts, left_on="link", right_on="link").nlargest(5, "Número de Comentarios")
    st.table(top_interactive_posts[["date", "account_name", "content", "likes", "Número de Comentarios"]].rename(columns={
        "date": "Fecha",
        "account_name": "Cuenta",
        "content": "Contenido",
        "likes": "Likes",
        "Número de Comentarios": "Comentarios"
    }))



# Página 2: Nube de Palabras y N-Gramas

elif page == "Nube de Palabras y N-Gramas":
    st.title("Análisis de Nube de Palabras y N-Gramas")

    # Obtener tópicos desde publications_df
    selected_topic = st.selectbox("Selecciona un tópico:", publications_df["Nombre"].unique())
    
    # Filtrar publicaciones y comentarios por tópico seleccionado
    topic_links = publications_df[publications_df["Nombre"] == selected_topic]["link"]
    filtered_comments = comments_df[comments_df["Link"].isin(topic_links)]["cleaned_comment"].dropna()
    corpus = filtered_comments.apply(clean_text)
    # corpus=filtered_comments
    # Ruta de la imagen para la máscara
    mask_image_path = "data/cervezax3.png"
    
    try:
        # Cargar la imagen y convertirla en máscara
        mask_image = np.array(Image.open(mask_image_path))
        
        # Crear la nube de palabras con la máscara
        st.header("Nube de Palabras")
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            mask=mask_image,
            contour_color="black",
            contour_width=1
        ).generate(" ".join(corpus))
        
        # Mostrar la nube de palabras
        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    
    except FileNotFoundError:
        st.error(f"No se encontró la imagen en la ruta {mask_image_path}. Por favor verifica la ubicación del archivo.")
    
    # Mostrar N-Gramas
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

    # Distribución de sentimientos por fuente
    st.header("Distribución de Sentimientos por Fuente")
    sentiment_by_source = comments_df.groupby(["source", "sentimiento_bert"]).size().reset_index(name="Conteo")
    fig_sentiment_source = px.bar(
        sentiment_by_source,
        x="source",
        y="Conteo",
        color="sentimiento_bert",
        barmode="group",
        title="Distribución de Sentimientos por Fuente",
        labels={"source": "Fuente", "Conteo": "Cantidad", "sentimiento_bert": "Sentimiento"},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_sentiment_source, use_container_width=True)

    # Distribución general de sentimientos
    st.header("Distribución General de Sentimientos")
    general_sentiment = comments_df["sentimiento_bert"].value_counts().reset_index()
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

    # Distribución de sentimientos por tópico
    st.header("Distribución de Sentimientos por Tópico")
    # Relacionar comentarios con publicaciones para obtener los tópicos
    merged_data = pd.merge(
        comments_df, publications_df,
        left_on="Link", right_on="link",
        how="inner"
    )
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

    # Tabla con publicaciones y cantidad de sentimientos
    st.header("Tabla de Publicaciones con Sentimientos Asociados")
    publication_sentiments = merged_data.groupby(["link", "sentimiento_bert"]).size().unstack(fill_value=0).reset_index()
    publication_sentiments.columns.name = None  # Eliminar el nombre del índice
    publication_sentiments = publication_sentiments.rename(columns={
        "link": "Publicación",
        "positivo": "Positivo",
        "negativo": "Negativo",
        "neutro": "Neutro"
    })
    st.table(publication_sentiments)

