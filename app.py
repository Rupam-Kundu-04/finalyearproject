from flask import Flask, render_template, request, jsonify
import json
import os
from ml_model import get_similar, get_cluster_label, get_recommendations_for_profile, get_cluster_summary

app = Flask(__name__)

# Load static database directly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'data.json'), 'r') as f:
    PRODUCTS = json.load(f)

def get_all_brands():
    brands = sorted(set(p['brand'] for p in PRODUCTS if p['brand']))
    return brands

def get_grade(score_raw):
    if score_raw == 0: return 'F'
    if score_raw >= 150: return 'A'
    if score_raw >= 130: return 'B'
    if score_raw >= 110: return 'C'
    if score_raw >= 90: return 'D'
    return 'F'

def get_eco_grade(score_raw):
    if score_raw == 0: return 'C'
    if score_raw >= 90: return 'A'
    elif score_raw >= 80: return 'B'
    elif score_raw >= 70: return 'C'
    else: return 'D'

def enrich_product(p):
    p = dict(p)
    p['health_grade'] = get_grade(p['health_score_raw'])
    p['eco_grade'] = get_eco_grade(p['eco_score_raw'])
    
    badges = []
    if p.get('vegan'): badges.append({'label': 'Vegan', 'color': 'green'})
    if p.get('organic'): badges.append({'label': 'Organic', 'color': 'emerald'})
    if p.get('cruelty_free'): badges.append({'label': 'Cruelty-Free', 'color': 'teal'})
    if not p.get('trans_fat'): badges.append({'label': 'No Trans Fat', 'color': 'blue'})
    if not p.get('artificial_colors'): badges.append({'label': 'No Artificial Colors', 'color': 'violet'})
    if not p.get('preservatives'): badges.append({'label': 'No Preservatives', 'color': 'amber'})
    p['badges'] = badges
    
    concerns = []
    if p.get('added_sugar'): concerns.append('Added Sugar')
    if p.get('added_salt'): concerns.append('High Salt')
    if p.get('preservatives'): concerns.append('Preservatives')
    if p.get('artificial_flavours'): concerns.append('Artificial Flavours')
    if p.get('artificial_colors'): concerns.append('Artificial Colors')
    if p.get('trans_fat'): concerns.append('Trans Fat')
    p['concerns'] = concerns
    
    return p

def recommend(filters):
    products = list(PRODUCTS)
    
    query = filters.get('query', '').strip().lower()
    if query:
        products = [p for p in products if 
                    query in p['name'].lower() or 
                    query in p['brand'].lower() or
                    query in p.get('ingredients', '').lower()]
    
    brand = filters.get('brand', '').strip()
    if brand:
        products = [p for p in products if p['brand'].lower() == brand.lower()]
    
    if filters.get('vegan'): products = [p for p in products if p['vegan']]
    if filters.get('organic'): products = [p for p in products if p['organic']]
    if filters.get('cruelty_free'): products = [p for p in products if p['cruelty_free']]
    if filters.get('no_trans_fat'): products = [p for p in products if not p['trans_fat']]
    if filters.get('no_artificial_colors'): products = [p for p in products if not p['artificial_colors']]
    if filters.get('no_preservatives'): products = [p for p in products if not p['preservatives']]
    if filters.get('no_added_sugar'): products = [p for p in products if not p['added_sugar']]
    if filters.get('no_artificial_flavours'): products = [p for p in products if not p['artificial_flavours']]

    sort_by = filters.get('sort', 'health')
    if sort_by == 'health':
        products.sort(key=lambda x: -(x['health_score_raw']) if x['health_score_raw'] > 0 else 999)
    elif sort_by == 'eco':
        products.sort(key=lambda x: -(x['eco_score_raw']))
    elif sort_by == 'protein':
        products.sort(key=lambda x: -(x['protein']))
    elif sort_by == 'low_sugar':
        products.sort(key=lambda x: x['sugars'])
    elif sort_by == 'low_fat':
        products.sort(key=lambda x: x['fat'])
    elif sort_by == 'name':
        products.sort(key=lambda x: x['name'].lower())

    return [enrich_product(p) for p in products]

@app.route('/')
def index():
    brands = get_all_brands()
    featured = [enrich_product(p) for p in sorted(
        [p for p in PRODUCTS if p['health_score_raw'] > 0],
        key=lambda x: -x['health_score_raw']
    )[:6]]
    stats = {
        'total': len(PRODUCTS),
        'brands': len(get_all_brands()),
        'vegan': sum(1 for p in PRODUCTS if p['vegan']),
        'no_preserv': sum(1 for p in PRODUCTS if not p['preservatives']),
    }
    return render_template('index.html', brands=brands, featured=featured, stats=stats)

@app.route('/category/<cat_name>')
def show_category(cat_name):
    """Filters data matching category fields provided in data.json manually"""
    brands = get_all_brands()
    filtered_raw = [p for p in PRODUCTS if p.get('category', '').lower() == cat_name.lower()]
    enriched = [enrich_product(p) for p in filtered_raw]
    stats = {
        'total': len(filtered_raw),
        'brands': len(set(p['brand'] for p in filtered_raw if p['brand'])),
        'vegan': sum(1 for p in filtered_raw if p['vegan']),
        'no_preserv': sum(1 for p in filtered_raw if not p['preservatives']),
    }
    return render_template('index.html', brands=brands, products=enriched, stats=stats, current_category=cat_name)

@app.route('/recommend', methods=['GET'])
def recommend_page():
    brands = get_all_brands()
    filters = {
        'query': request.args.get('query', ''),
        'brand': request.args.get('brand', ''),
        'vegan': request.args.get('vegan') == '1',
        'organic': request.args.get('organic') == '1',
        'cruelty_free': request.args.get('cruelty_free') == '1',
        'no_trans_fat': request.args.get('no_trans_fat') == '1',
        'no_artificial_colors': request.args.get('no_artificial_colors') == '1',
        'no_preservatives': request.args.get('no_preservatives') == '1',
        'no_added_sugar': request.args.get('no_added_sugar') == '1',
        'no_artificial_flavours': request.args.get('no_artificial_flavours') == '1',
        'sort': request.args.get('sort', 'health'),
    }
    results = recommend(filters)
    return render_template('recommend.html', products=results, brands=brands, filters=filters, total=len(results))

@app.route('/product/<product_id>')
def product_detail(product_id):
    product = next((p for p in PRODUCTS if str(p['id']) == str(product_id)), None)
    if not product:
        return "Product not found", 404
    product = enrich_product(product)
    
    similar_raw = get_similar(product_id, top_n=6)
    similar = [enrich_product(p) for p in similar_raw]
    cluster = get_cluster_label(product_id)

    return render_template('product.html', product=product, similar=similar, cluster=cluster)

@app.route('/compare')
def compare():
    ids = request.args.getlist('ids')
    products = []
    for pid in ids[:4]:
        p = next((p for p in PRODUCTS if str(p['id']) == str(pid)), None)
        if p:
            products.append(enrich_product(p))
    brands = get_all_brands()
    all_products_min = [{'id': p['id'], 'name': p['name'], 'brand': p['brand']} for p in PRODUCTS]
    return render_template('compare.html', products=products, brands=brands, all_products=all_products_min)

@app.route('/api/search')
def api_search():
    query = request.args.get('q', '').strip().lower()
    if not query or len(query) < 2:
        return jsonify([])
    results = [{'id': p['id'], 'name': p['name'], 'brand': p['brand']} 
               for p in PRODUCTS 
               if query in p['name'].lower() or query in p['brand'].lower()][:10]
    return jsonify(results)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/ml-recommend', methods=['GET'])
def ml_recommend():
    prefs = {
        'no_preservatives':     request.args.get('no_preservatives') == '1',
        'no_artificial_colors': request.args.get('no_artificial_colors') == '1',
        'no_trans_fat':         request.args.get('no_trans_fat') == '1',
        'vegan':                request.args.get('vegan') == '1',
        'organic':              request.args.get('organic') == '1',
        'health_preference':    request.args.get('health_preference', 'any'),
        'max_sugar':            float(request.args.get('max_sugar', 50)),
        'max_fat':              float(request.args.get('max_fat', 35)),
        'min_protein':          float(request.args.get('min_protein', 0)),
    }
    results_raw = get_recommendations_for_profile(prefs, top_n=12)
    results = [enrich_product(p) for p in results_raw]
    cluster_summary = get_cluster_summary()
    return render_template('ml_recommend.html', products=results, prefs=prefs, cluster_summary=cluster_summary, total=len(results))

@app.route('/api/similar/<product_id>')
def api_similar(product_id):
    results = get_similar(product_id, top_n=6)
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)