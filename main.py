from datetime import time
import streamlit as st
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import AdvancedDataLoader
from feature_extractor import AdvancedFeatureExtractor
from sentiment_analyzer import SentimentAnalyzer
from text_preprocessor import AdvancedTextPreprocessor


class SentimentAnalysisPipeline:
    def __init__(self, data_path):
        self.data_loader = AdvancedDataLoader(data_path)
        self.text_preprocessor = AdvancedTextPreprocessor()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
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

            # Vectorisation
            self.feature_extractor.fit(self.df['processed'])
            features = self.feature_extractor.transform(self.df['processed'])

            # Split des données
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                features,
                self.df['overall'].apply(lambda x: 'positive' if x > 3 else 'negative'),
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
                'preprocessed_data': self.df[['reviewText', 'processed']],
                'feature_names': self.feature_extractor.get_feature_names()
            }

    def predict_single_review(self, user_review):
        """Prédiction pour une revue unique"""
        if not self.is_trained:
            raise RuntimeError("Le modèle n'a pas été entraîné. Exécutez run_analysis() d'abord.")

        # Prétraitement
        processed_text = self.text_preprocessor.preprocess(user_review)

        # Vectorisation
        features = self.feature_extractor.transform([processed_text])

        # Prédiction
        return self.sentiment_analyzer.predict(features)[0]

def create_dashboard():
    st.set_page_config(page_title="Analyse de Sentiments", layout="wide")
    st.title("📈 Tableau de Bord Interactif")

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
                # Section Résultats
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Performance du Modèle")
                    st.write("**Précision Globale:**", st.session_state.results['metrics']['accuracy'])
                    st.write("**Précision Classe Positive:**",
                             st.session_state.results['metrics']['positive']['precision'])
                    st.write("**Rappel Classe Négative:**", st.session_state.results['metrics']['negative']['recall'])

                with col2:
                    st.subheader("Distribution des Sentiments")
                    fig, ax = plt.subplots()
                    ax.pie(
                        [st.session_state.results['metrics']['positive']['support'],
                         st.session_state.results['metrics']['negative']['support']],
                        labels=['Positif', 'Négatif'],
                        autopct='%1.1f%%'
                    )
                    st.pyplot(fig)

                # Export des données
                st.subheader("Aperçu des Données Prétraitées")
                st.dataframe(st.session_state.results['preprocessed_data'].head())

                csv = st.session_state.results['preprocessed_data'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Exporter les données prétraitées",
                    data=csv,
                    file_name='donnees_pretraitees.csv',
                    mime='text/csv'
                )

                # Analyse personnalisée
                st.subheader("🔍 Analyse Personnalisée")
                user_review = st.text_area("Entrez un avis à analyser:", key="review_input")

                if st.button("Analyser"):
                    if user_review:
                        start_time = time.time()
                        prediction = st.session_state.pipeline.predict_single_review(user_review)
                        st.write(f"Temps d'analyse : {time.time() - start_time:.2f}s")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Avis Original**")
                            st.write(user_review)
                        with col2:
                            st.markdown("**Résultat d'Analyse**")
                            st.metric("Sentiment Prédit", prediction.upper())
                    else:
                        st.warning("Veuillez entrer un avis à analyser")

        except Exception as e:
            st.error(f"Erreur: {str(e)}")


def main():
    create_dashboard()


if __name__ == "__main__":
    main()