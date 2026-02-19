
###### ========= Streamlit App for a personalized NLP text Analyzer ======= ######
#================================================================================#
                    #===================================#

import streamlit as st
import spacy
import spacy.cli
from textblob import TextBlob                           # TextBlob is a quick mood reader — it gives a simple sentiment score (positive/negative) and subjectivity (opinion vs fact) in one line, without any training or complex setup.
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Cache spaCy model (loads only once)
@st.cache_resource
# Load spaCy model once (outside everything)
def load_spacy_model():
    spacy.cli.download("en_core_web_sm")                # Safe: skips if already present
    return spacy.load("en_core_web_sm")
nlp = load_spacy_model()

@st.cache_resource                                      # I used two cache_resource beacuse each @st.cache_resource function caches one specific thing. We need separate decorators so Streamlit caches each model independently (fast reloads for both).Two models → two caches. No sharing possible
# Load model once (outside button for simplicity)
def load_sentence_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedder = load_sentence_embedder()


# App title (always visible)
st.title("NLP Text Analyzer")

st.write("Enter you text to get the thorough analysis via NLP")

# Sidebar toggles (user can turn features on/off)
st.sidebar.title("Analysis Options")
show_pos = st.sidebar.checkbox("Show POS Tags", value=True)
show_ner = st.sidebar.checkbox("Show Named Entities", value=True)
show_sentiment = st.sidebar.checkbox("Show Sentiment", value=True)
show_keywords = st.sidebar.checkbox("Show Keywords", value=True)
show_similar = st.sidebar.checkbox("Show Similar Sentences", value=True)

# Input box for user text
user_text = st.text_area(
    label="Paste or type your text here",
    height=200,
    placeholder="Enter your text to analyze..."
)

## Step 1: Basic streamlit app template like Analyze button + logic
if st.button("Analyze"):
    if not user_text.strip():
        st.error("No text entered. Please paste something!")
    elif len(user_text) < 10:                                                     # very short if charcters (including spaces) are less than 10
        st.warning("Text too short — add more content for better analysis.")
    else:
        st.success("Analysis started! More features coming...")
        with st.spinner("Analyzing the text..."):                                 # Nice loading effect
            doc = nlp(user_text)
            

## Step 2: Token count + simple preview
            # Basic Stats (always show)
            token_count = len(doc)
            st.subheader("Basic Stats")     # for clean layout
            st.write(f"**Total Tokens:** {token_count}")

            # Showing first 20 cleaned tokens as proof
            cleaned_tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
            st.write("**First 20 cleaned tokens (lowercased):**")
            st.write(", ".join(cleaned_tokens[:20]) if cleaned_tokens else "No tokens after cleaning.")

### ===== Conditional sections from Step 3 to Step 7 based on checkboxes (for POS, NER, kewords extractor, text sentiments, similar sentences part)

## Step 3: POS Table (first 20 tokens for preview)
            if show_pos:
                if len(doc) > 0:
                    st.subheader("POS Tags (First 20 Tokens)(Parts of Speech)")
        
                    pos_df = pd.DataFrame({
                        'Text': [token.text for token in doc[:20]],
                        'Lemma': [token.lemma_ for token in doc[:20]],
                        'POS': [token.pos_ for token in doc[:20]],
                        'Tag': [token.tag_ for token in doc[:20]]
                    })
                    st.dataframe(pos_df, use_container_width=True)  # nice wide table
                else:
                    st.info("No tokens to show POS tags.")


## Step 4: Named Entity Recognition
            if show_ner:
                st.subheader("NER (Named Entity Recognition)")
                if doc.ents:
                    entity_list = defaultdict(list)
                    for entity in doc.ents:
                        entity_list[entity.label_].append(entity.text)
                    # Making a dataframe (like a table)
                    st.dataframe(pd.DataFrame(entity_list), use_container_width=True)
                else:
                    st.info("No Named Entities found.")
            

## Step 5: Sentiment Score (using TextBlob)
            if show_sentiment:
                st.subheader("Sentiment Analysis")

                blob = TextBlob(user_text)                      # TextBlob works directly on raw text strings — it does its own simple tokenization and analysis. It doesn’t understand spaCy Doc objects. So for sentiment → we give it the original user_text (raw input) — that’s what it expects. As spaCy doc is only needed for structured things (sentences, POS, entities).
                polarity = blob.sentiment.polarity              # -1 (very negative) to +1 (very positive)
                subjectivity = blob.sentiment.subjectivity      # 0 (factual) to 1 (opinionated)

                # Simple interpretation
                if polarity > 0.5:
                    mood = "Very Positive 😊"
                elif polarity > 0.1:
                    mood = "Positive 🙂"
                elif polarity < -0.5:
                    mood = "Very Negative 😔"
                elif polarity < -0.1:
                    mood = "Negative 😞"
                else:
                    mood = "Neutral 😐"

                st.write(f"**Polarity:** {polarity:.2f} → {mood}")
                st.write(f"**Subjectivity:** {subjectivity:.2f} (0 = very factual, 1 = very opinionated)")


## Step 6: Keyword Extraction (TF-IDF on single text)
            if show_keywords:
                st.subheader("Keywords (TF-IDF Top 10)")

                if user_text.strip():
                    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')   # Creates the TF-IDF "calculator". max_features=10 → we only want top 10 keywords. stop_words='english' → automatically ignores "the", "is", "and", etc.
                    tfidf_matrix = vectorizer.fit_transform([user_text])                  # fit_transform learns which words matter most in this single text. [user_text] → we wrap the text in a list because TF-IDF expects multiple documents (even if just 1). Output: a matrix (sparse table) of word importance scores.
                    feature_names = vectorizer.get_feature_names_out()                    # → Gets the list of words/phrases that were actually used (e.g., ["elon", "musk", "tesla", "lahore", ...])
                    scores = tfidf_matrix.toarray()[0]                                    # → Converts the sparse matrix to normal array. [0] → takes the row for our single document. → Now scores is a list of numbers — one score per word (higher = more important)

                # Sort & prepare data for display/chart
                    if scores.sum() > 0:                                                  # check if any scores exist (avoids empty case)
                        top_idx = scores.argsort()[-10:][::-1]                            # → argsort() → sorts scores from lowest to highest → gets their positions. [-10:] → takes last 10 (highest scores). [::-1] → reverses so highest first. → top_idx = indices of the 10 most important words

                        top_words = feature_names[top_idx]                                # → feature_names[top_idx] instantly gives you the 10 words at those indices. top_idx is a NumPy array of 10 indices (e.g., [3, 7, 1, 8, 0, 4, 9, 2, 5, 6]). feature_names[top_idx] will handle 10 indices at once. (=== info: NumPy allows fancy indexing: when you pass an array of indices to another array, it returns a new array with values at all those positions at once.=== )
                        top_scores = scores[top_idx]                                      # → scores[top_idx] gives the 10 corresponding scores.

                        # Show as list
                        st.write("Top keywords:")
                        for word, score in zip(top_words, top_scores):
                            st.write(f"- {word} (score: {score:.3f})")

                        # Show as bar chart
                        chart_data = pd.DataFrame({
                            "Keywords": top_words,
                            "Scores": top_scores
                        })
                        st.bar_chart(chart_data.set_index("Keywords"))                   # set_index("Keywords") makes "Keywords" the row index (x-axis labels) instead of numbers 0,1,2,..
                else:
                    st.info("No text to extract keywords from.")


## Step 7: Similar Sentences
            if show_similar:
                st.subheader("Similar Sentences")

                # Get sentences from spaCy doc
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]  # doc.sents is spaCy’s built-in sentence splitter — it gives clean, accurate sentences.(→ much better than splitting user_text manually). sent.text.strip() → gets the clean sentence string and removes extra spaces.

                if len(sentences) >= 2:
                    # Compute embeddings
                    embeddings = embedder.encode(sentences)

                    # Cosine similarity matrix
                    sim_matrix = cosine_similarity(embeddings)      
                    # st.write(sim_matrix)

                    # Find top similar pairs (ignore self-similarity and duplicates)
                    pairs=[]
                    for i in range(len(sentences)):                                             # Double loop for i in range(...) + for j in range(i+1, ...) → compares every sentence with every later sentence (no self-comparison, no duplicates)
                        for j in range(i+1, len(sentences)):
                            sim_matrix_score = sim_matrix[i][j]                                 # score = sim_matrix[i][j] → gets similarity between sentence i and j
                            if sim_matrix_score > 0.7:                                          # threshold for "similar enough". If score > 0.7: → only keep pairs that are quite similar
                                pairs.append( (sentences[i], sentences[j], sim_matrix_score) )    # → saves the pair + sim_matrix_score. pairs is a list of tuples that looks like this e.g. (after sorting): pairs = [("Sentence A", "Sentence B", 0.912), ("Sentence A", "Sentence C", 0.875)]. s1 = first sentence (string). s2 = second sentence (string). sim_matrix_score = similarity number (float, e.g. 0.912)

                    if pairs:
                        # Sort by sim_matrix_score descending
                        pairs.sort(key=lambda x:x[2], reverse=True)                             # → sorts by sim_matrix_score (highest first). key=lambda x: x[2] → tells Python what to sort by. lambda = a tiny, nameless function (like a quick one-line rule). x = each item in the list (each tuple). x[2] = the third element of the tuple (index 2 → the sim_matrix_score)
                        st.write("Most similar pairs:")
                        for s1, s2, sim_matrix_score in pairs[:5]:                              # → shows top 5 matches with nice formatting
                            st.write(f"**{sim_matrix_score:.3f}** -> {s1}")
                            st.write(f" vs {s2}")
                            st.markdown("----")
                    else:
                        st.info("No very similar sentences found (all pairs < 0.7 similarity).")

            
                else:
                    st.info("Need at least 2 sentences to find similarities.")


            # Placeholder for future steps (named entities, etc.)
            st.success("Analysis has been completed!")

else:
    st.info("Enter some text and click Analyze to start.")

st.markdown("---")
st.caption("Built with spaCy, TextBlob & sentence-transformers | A personal NLP tool")