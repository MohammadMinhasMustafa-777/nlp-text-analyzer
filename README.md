# NLP Text Analyzer App

A Streamlit web app that analyzes text using spaCy, TextBlob, and Sentence Transformers — showing POS tags, named entities, sentiment, keywords, and similar sentences.

## Features
- Paste or type any text for instant NLP analysis
- Token count, POS tags table, and lemmas
- Named entity recognition (with labels)
- Sentiment polarity & subjectivity score
- Keyword extraction via TF-IDF
- Similar sentence pairs using cosine similarity
- Toggle analyses via sidebar checkboxes
- Built with spaCy, TextBlob, Sentence Transformers, scikit-learn, Streamlit

## Setup & Local Run
1. Install dependencies: `pip install -r requirements.txt`
2. Download spaCy model: `python -m spacy download en_core_web_sm`
3. Run: `streamlit run app.py`
4. Open: `http://localhost:8501`
5. Paste text → Analyze

## Technologies
- Backend: spaCy, TextBlob, Sentence Transformers, scikit-learn
- Frontend: Streamlit
- Deployment: Streamlit Community Cloud

## Deployment
- Deployed on Streamlit Community Cloud
  Live app: https://nlp-text-analyzer-by-minhas.streamlit.app/


## Model & Accuracy
- NER + POS: `en_core_web_sm` (spaCy)
- Sentence similarity: `paraphrase-MiniLM-L3-v3` (Sentence Transformers)
- Sentiment: TextBlob polarity (-1 to +1)

## Notes
- Best results with 3+ sentences for similarity analysis
- Short or vague input may yield limited keyword/entity results

## License
MIT License
