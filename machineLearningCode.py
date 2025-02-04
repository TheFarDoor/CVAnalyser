import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to extract keywords using spaCy
def extract_keywords(text):
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]

# Sample job description and CV
job_description = "Looking for a social media manager with experience in content creation and team collaboration."
cv = "Recent graduate with experience in managing social media platforms and creating engaging content. Worked in team settings to develop marketing strategies."

# Extract keywords
job_keywords = extract_keywords(job_description)
cv_keywords = extract_keywords(cv)

# Combine keywords into strings
job_keywords_str = ' '.join(job_keywords)
cv_keywords_str = ' '.join(cv_keywords)

# Vectorize the keywords
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([job_keywords_str, cv_keywords_str])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"Similarity Score: {cosine_sim[0][0]}")