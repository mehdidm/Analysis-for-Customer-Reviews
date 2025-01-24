import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

from data_loader import AdvancedDataLoader
from feature_extractor import AdvancedFeatureExtractor
from sentiment_analyzer import SentimentAnalyzer
from text_preprocessor import AdvancedTextPreprocessor
from theme_extractor import ThemeExtractor


class SentimentAnalysisPipeline:
    def __init__(self, data_path):
        self.data_loader = AdvancedDataLoader(data_path)
        self.text_preprocessor = AdvancedTextPreprocessor()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.theme_extractor = ThemeExtractor()
        self.is_trained = False
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def run_analysis(self):
        """Entraîne le modèle et retourne les résultats"""
        if not self.is_trained:
            # Chargement et prétraitement
            self.df = self.data_loader.load_data()
            self.df['processed'] = self.df['reviewText'].apply(self.text_preprocessor.preprocess)

            # Ajouter une colonne de sentiment
            self.df['sentiment'] = self.df['overall'].apply(lambda x: 'positive' if x > 3 else 'negative')

            # Vectorisation
            self.feature_extractor.fit(self.df['processed'])
            features = self.feature_extractor.transform(self.df['processed'])

            # Extraction des thèmes
            topics = self.theme_extractor.extract_topics(self.df['processed'])

            # Split des données
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                features,
                self.df['sentiment'],
                test_size=0.2,
                random_state=42
            )

            # Entraînement
            self.sentiment_analyzer.train(self.X_train, self.y_train)

            # Calcul des métriques
            metrics = self.sentiment_analyzer.evaluate(self.X_test, self.y_test)

            self.is_trained = True

            return {
                'metrics': metrics,
                'preprocessed_data': self.df[['reviewText', 'processed', 'sentiment']],
                'feature_names': self.feature_extractor.get_feature_names(),
                'topics': topics,
                'time_trends': self.analyze_time_trends(),
                'feature_importance': self.sentiment_analyzer.get_feature_importance() if hasattr(
                    self.sentiment_analyzer, 'get_feature_importance') else None
            }

    def analyze_time_trends(self):
        """Analyser les tendances temporelles des sentiments"""
        if 'date' in self.df.columns:
            time_trend = self.df.groupby(pd.Grouper(key='date', freq='M'))['sentiment'].value_counts(
                normalize=True).unstack()
            return time_trend
        return None

    def predict_single_review(self, user_review):
        """Prédiction pour une revue unique"""
        if not self.is_trained:
            raise RuntimeError("Le modèle n'a pas été entraîné. Exécutez run_analysis() d'abord.")

        # Prétraitement
        processed_text = self.text_preprocessor.preprocess(user_review)

        # Vectorisation
        features = self.feature_extractor.transform([processed_text])

        # Prédiction
        return {
            'sentiment': self.sentiment_analyzer.predict(features)[0],
            'details': self.text_preprocessor.sentiment_pipeline(user_review[:512])[0]
        }


def create_dashboard():
    st.set_page_config(page_title="Analyse de Sentiments", layout="wide")
    st.title("📈 Tableau de Bord Interactif d'Analyse de Sentiments")

    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
        st.session_state.results = None

    uploaded_file = st.file_uploader("Téléversez vos avis clients (CSV)", type="csv")

    if uploaded_file:
        try:
            if st.session_state.pipeline is None:
                with st.spinner("Entraînement du modèle en cours..."):
                    st.session_state.pipeline = SentimentAnalysisPipeline(uploaded_file)
                    st.session_state.results = st.session_state.pipeline.run_analysis()

            if st.session_state.results:
                # Sections de l'onglet
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Performance", "Analyse des Mots", "Thèmes Détectés", "Prédiction Personnalisée"])

                with tab1:
                    # Performance du modèle
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Performance du Modèle")
                        st.write("**Précision Globale:**", st.session_state.results['metrics']['accuracy'])
                        st.write("**Précision Classe Positive:**",
                                 st.session_state.results['metrics']['positive']['precision'])
                        st.write("**Rappel Classe Négative:**",
                                 st.session_state.results['metrics']['negative']['recall'])

                    with col2:
                        st.subheader("Distribution des Sentiments")
                        fig, ax = plt.subplots()
                        ax.pie(
                            [st.session_state.results['metrics']['positive']['support'],
                             st.session_state.results['metrics']['negative']['support']],
                            labels=['Positif', 'Négatif'],
                            autopct='%1.1f%%'
                        )
                        st.pyplot(fig, clear_figure=True)

                with tab2:
                    # Nuage de mots et caractéristiques
                    col1, col2 = st.columns(2)
                    with col1:
                        positive_text = ' '.join(st.session_state.results['preprocessed_data'][
                                                     st.session_state.results['preprocessed_data'][
                                                         'sentiment'] == 'positive']['processed'])
                        wordcloud = WordCloud(width=400, height=300).generate(positive_text)
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis("off")
                        st.pyplot(plt, clear_figure=True)
                        st.caption("Mots-clés fréquents (Avis Positifs)")

                    with col2:
                        negative_text = ' '.join(st.session_state.results['preprocessed_data'][
                                                     st.session_state.results['preprocessed_data'][
                                                         'sentiment'] == 'negative']['processed'])
                        wordcloud = WordCloud(width=400, height=300).generate(negative_text)
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis("off")
                        st.pyplot(plt, clear_figure=True)
                        st.caption("Mots-clés fréquents (Avis Négatifs)")

                    # Analyse des caractéristiques
                    if st.session_state.results['feature_importance'] is not None:
                        st.subheader("🧠 Mots les Plus Influents")
                        coef_df = pd.DataFrame({
                            'word': st.session_state.results['feature_names'],
                            'importance': st.session_state.results['feature_importance']
                        }).sort_values('importance', ascending=False).head(20)

                        fig = px.bar(
                            coef_df,
                            x='importance',
                            y='word',
                            orientation='h',
                            title="Impact sur la Prédiction de Sentiment"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    st.subheader("🔍 Thèmes Détectés")
                    # Affichage des thèmes détectés
                    topics = st.session_state.results['topics']
                    for idx, topic in enumerate(topics, 1):
                        st.write(f"**Thème {idx}:** {topic}")

                with tab4:
                    st.subheader("🔮 Prédiction de Sentiment pour un Nouvel Avis")
                    user_review = st.text_area("Saisissez votre avis client")

                    if st.button("Prédire le Sentiment"):
                        if user_review:
                            prediction = st.session_state.pipeline.predict_single_review(user_review)
                            st.write("**Sentiment Prédit:** ", prediction['sentiment'])
                            st.write("**Détails:**")
                            st.write(f"- Label: {prediction['details']['label']}")
                            st.write(f"- Score: {prediction['details']['score']:.2f}")
                        else:
                            st.warning("Veuillez saisir un avis.")

        except Exception as e:
            st.error(f"Erreur: {str(e)}")


def main():
    create_dashboard()


if __name__ == "__main__":
    main()