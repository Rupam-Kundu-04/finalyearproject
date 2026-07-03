"""
BiscuitIQ — ML Recommendation Engine
Algorithm: Content-Based Filtering using Cosine Similarity
"""

import json, os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# ── Load data ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'data.json'), 'r') as f:
    PRODUCTS = json.load(f)

# ── Feature columns ──────────────────────────────────────────────────────────
NUM_FEATURES = [
    'energy', 'fat', 'protein', 'carbs', 'sugars',
    'added_sugar', 'added_salt', 'preservatives',
    'artificial_flavours', 'artificial_colors', 'trans_fat',
    'organic', 'vegan', 'cruelty_free',
    'health_score_raw', 'eco_score_raw'
]

# ── Build feature matrix ─────────────────────────────────────────────────────
def _build_features():
    X_num = np.array(
        [[p.get(f, 0) or 0 for f in NUM_FEATURES] for p in PRODUCTS],
        dtype=float
    )
    scaler = MinMaxScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    ingredients = [p.get('ingredients', '') or '' for p in PRODUCTS]
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    X_text = tfidf.fit_transform(ingredients).toarray()

    X_combined = np.hstack([X_num_scaled * 0.6, X_text * 0.4])
    sim_matrix = cosine_similarity(X_combined)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_num_scaled)

    return sim_matrix, clusters, X_num_scaled, scaler, tfidf, X_combined

SIM_MATRIX, CLUSTERS, X_NUM_SCALED, SCALER, TFIDF, X_COMBINED = _build_features()
ID_TO_IDX = {str(p['id']): i for i, p in enumerate(PRODUCTS)}

_cluster_health = {}
for i, p in enumerate(PRODUCTS):
    c = int(CLUSTERS[i])
    if c not in _cluster_health:
        _cluster_health[c] = []
    _cluster_health[c].append(p['health_score_raw'])

_cluster_avg = {c: np.mean(v) for c, v in _cluster_health.items()}
_sorted_clusters = sorted(_cluster_avg, key=lambda c: _cluster_avg[c], reverse=True)

CLUSTER_LABELS = {}
labels = ['Healthiest', 'Healthy', 'Moderate', 'Indulgent', 'Most Indulgent']
for rank, cid in enumerate(_sorted_clusters):
    CLUSTER_LABELS[cid] = labels[rank]

CLUSTER_COLORS = {
    'Healthiest':      '#DCFCE7',
    'Healthy':         '#D1FAE5',
    'Moderate':        '#FEF9C3',
    'Indulgent':       '#FED7AA',
    'Most Indulgent':  '#FEE2E2',
}
CLUSTER_TEXT_COLORS = {
    'Healthiest':      '#166534',
    'Healthy':         '#065F46',
    'Moderate':        '#92400E',
    'Indulgent':       '#9A3412',
    'Most Indulgent':  '#991B1B',
}

# ── Public API ───────────────────────────────────────────────────────────────

def get_similar(product_id: str, top_n: int = 6) -> list:
    """Return top_n most similar products belonging to the same category."""
    idx = ID_TO_IDX.get(str(product_id))
    if idx is None:
        return []

    target_category = PRODUCTS[idx].get('category')
    scores = list(enumerate(SIM_MATRIX[idx]))
    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, score in scores:
        if i == idx:
            continue
        
        # Isolate matching categories cleanly
        if PRODUCTS[i].get('category') != target_category:
            continue
            
        p = dict(PRODUCTS[i])
        p['similarity'] = round(float(score) * 100, 1)
        p['cluster_label'] = CLUSTER_LABELS[int(CLUSTERS[i])]
        results.append(p)
        if len(results) >= top_n:
            break
    return results


def get_cluster_label(product_id: str) -> dict:
    idx = ID_TO_IDX.get(str(product_id))
    if idx is None:
        return {'label': 'Unknown', 'bg': '#F3F4F6', 'color': '#6B7280'}
    label = CLUSTER_LABELS[int(CLUSTERS[idx])]
    return {
        'label': label,
        'bg': CLUSTER_COLORS[label],
        'color': CLUSTER_TEXT_COLORS[label],
    }


def get_recommendations_for_profile(preferences: dict, top_n: int = 12) -> list:
    health_pref = preferences.get('health_preference', 'any')

    def passes_health_filter(p):
        s = p.get('health_score_raw', 0)
        if health_pref == 'healthy':
            return s >= 130
        elif health_pref == 'moderate':
            return 100 <= s < 150
        return s > 0

    def passes_dietary_filter(p):
        if preferences.get('no_preservatives') and p.get('preservatives'):
            return False
        if preferences.get('no_artificial_colors') and p.get('artificial_colors'):
            return False
        if preferences.get('no_trans_fat') and p.get('trans_fat'):
            return False
        if preferences.get('vegan') and not p.get('vegan'):
            return False
        if preferences.get('organic') and not p.get('organic'):
            return False
        return True

    def passes_nutrition_filter(p):
        if p.get('sugars', 0) > preferences.get('max_sugar', 50):
            return False
        if p.get('fat', 0) > preferences.get('max_fat', 35):
            return False
        if p.get('protein', 0) < preferences.get('min_protein', 0):
            return False
        return True

    filtered = []
    for i, p in enumerate(PRODUCTS):
        if not passes_health_filter(p):
            continue
        if not passes_dietary_filter(p):
            continue
        if not passes_nutrition_filter(p):
            continue
        pd = dict(p)
        pd['cluster_label'] = CLUSTER_LABELS[int(CLUSTERS[i])]
        filtered.append((i, pd))

    if not filtered:
        return []

    feat_idx = {f: i for i, f in enumerate(NUM_FEATURES)}
    ideal = np.zeros(len(NUM_FEATURES))

    if health_pref == 'healthy':
        ideal[feat_idx['health_score_raw']] = 170
        ideal[feat_idx['eco_score_raw']] = 90
        ideal[feat_idx['protein']] = 15
        ideal[feat_idx['sugars']] = 5
        ideal[feat_idx['fat']] = 8
    elif health_pref == 'moderate':
        ideal[feat_idx['health_score_raw']] = 130
        ideal[feat_idx['eco_score_raw']] = 90
        ideal[feat_idx['protein']] = 7
        ideal[feat_idx['sugars']] = 20
        ideal[feat_idx['fat']] = 15
    else:
        ideal[feat_idx['health_score_raw']] = 150
        ideal[feat_idx['eco_score_raw']] = 90

    ideal[feat_idx['sugars']] = preferences.get('max_sugar', 50) * 0.5
    ideal[feat_idx['fat']] = preferences.get('max_fat', 35) * 0.5
    ideal[feat_idx['protein']] = preferences.get('min_protein', 0)

    ideal_scaled = SCALER.transform([ideal])[0] * 0.6
    ideal_full = np.concatenate([ideal_scaled, np.zeros(X_COMBINED.shape[1] - len(ideal_scaled))])

    for idx, pd in filtered:
        sim = float(cosine_similarity([ideal_full], [X_COMBINED[idx]])[0][0])
        pd['similarity'] = round(sim * 100, 1)

    filtered.sort(key=lambda x: (-x[1]['health_score_raw'], -x[1]['similarity']))
    return [pd for _, pd in filtered[:top_n]]


def get_cluster_summary() -> list:
    summary = {}
    for i, p in enumerate(PRODUCTS):
        c = int(CLUSTERS[i])
        label = CLUSTER_LABELS[c]
        if label not in summary:
            summary[label] = {
                'label': label,
                'count': 0,
                'avg_health': [],
                'avg_protein': [],
                'bg': CLUSTER_COLORS[label],
                'color': CLUSTER_TEXT_COLORS[label],
            }
        summary[label]['count'] += 1
        summary[label]['avg_health'].append(p['health_score_raw'])
        summary[label]['avg_protein'].append(p['protein'] or 0)

    result = []
    for label, s in summary.items():
        result.append({
            'label': s['label'],
            'count': s['count'],
            'avg_health': round(np.mean(s['avg_health']), 1),
            'avg_protein': round(np.mean(s['avg_protein']), 1),
            'bg': s['bg'],
            'color': s['color'],
        })
    result.sort(key=lambda x: x['avg_health'])
    return result