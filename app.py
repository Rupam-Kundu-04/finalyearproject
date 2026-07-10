from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash
from ml_model import get_similar, get_cluster_label, get_recommendations_for_profile, get_cluster_summary

app = Flask(__name__)
# Secure secret key used to encrypt Flask session cookies safely
app.secret_key = 'hetc_cse_final_year_project_secret_key_biscuit_iq'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, 'data.json')
USERS_PATH = os.path.join(BASE_DIR, 'users.json')

# Load the core static database array
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    PRODUCTS = json.load(f)

# ─── USER STORAGE HELPERS ───

def load_users():
    """Loads authenticated user profiles securely from users.json file with error fallbacks."""
    if not os.path.exists(USERS_PATH):
        return {}
        
    try:
        with open(USERS_PATH, 'r', encoding='utf-8') as f:
            # Check if file is completely blank
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, IOError, PermissionError) as e:
        print(f"Database Read Warning: {e}. Initializing clean user memory structure.")
        return {}

def save_users(users_data):
    """Saves updated user account structures cleanly into users.json."""
    with open(USERS_PATH, 'w', encoding='utf-8') as f:
        json.dump(users_data, f, indent=2, ensure_ascii=False)

def get_all_brands():
    """Extracts a unique sorted list of all active brand names."""
    return sorted(set(p['brand'] for p in PRODUCTS if p['brand']))

# ─── CORE EVALUATION METRICS ───

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
    p['product_category'] = p.get('category', 'General')
    
    # ─── EXTRA SAFE SESSION AND KEY BOUNDARY CHECK ───
    p['is_favorite'] = False
    if 'user_email' in session and session['user_email']:
        users = load_users()
        user_info = users.get(session['user_email'])
        
        # Ensure user profile exists AND has a valid favorites list
        if user_info and isinstance(user_info, dict):
            favorites_list = user_info.get('favorites', [])
            # Fallback if the key explicitly holds None instead of a list
            if favorites_list is None:
                favorites_list = []
                
            if str(p['id']) in [str(fid) for fid in favorites_list]:
                p['is_favorite'] = True
    # ──────────────────────────────────────────────────
            
    badges = []
    if p.get('category'): badges.append({'label': p['category'], 'color': 'indigo'})
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
    """Filters data matching dietary parameters and user sorting requests."""
    products = list(PRODUCTS)
    
    query = filters.get('query', '').strip().lower()
    if query:
        products = [p for p in products if 
                    query in p['name'].lower() or 
                    query in p['brand'].lower() or
                    query in p.get('ingredients', '').lower() or
                    query in p.get('category', '').lower()]
    
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

# ─── USER SESSION AUTHENTICATION PATHS ───

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        if not name or not email or not password:
            return render_template('signup.html', error="All fields are required.")
            
        users = load_users()
        if email in users:
            return render_template('signup.html', error="An account with this email already exists.")
            
        # Securely hash user password before storage
        users[email] = {
            'name': name,
            'password': generate_password_hash(password),
            'favorites': []
        }
        save_users(users)
        
        # Initialize session records instantly
        session['user_email'] = email
        session['user_name'] = name
        return redirect(url_for('dashboard'))
        
    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        users = load_users()
        if email not in users or not check_password_hash(users[email]['password'], password):
            return render_template('signin.html', error="Invalid email or password.")
            
        session['user_email'] = email
        session['user_name'] = users[email]['name']
        return redirect(url_for('dashboard'))
        
    return render_template('signin.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ─── USER DASHBOARD & FAVORITES MIDDLEWARE ───

@app.route('/dashboard')
def dashboard():
    if 'user_email' not in session:
        return redirect(url_for('signin'))
        
    users = load_users()
    user_info = users.get(session['user_email'], {})
    favorite_ids = user_info.get('favorites', [])
    
    # Capture only the items bookmarked by the active user
    fav_products = [enrich_product(p) for p in PRODUCTS if str(p['id']) in favorite_ids]
    return render_template('dashboard.html', name=session['user_name'], products=fav_products)

@app.route('/toggle-favorite/<product_id>', methods=['POST'])
def toggle_favorite(product_id): # <-- MAKE SURE 'product_id' IS EXACTLY HERE!
    if 'user_email' not in session:
        return jsonify({'status': 'unauthorized', 'message': 'Please login first'}), 401
        
    email = session['user_email']
    product_id = str(product_id)
    
    users = load_users()
    
    if email not in users:
        users[email] = {
            'name': session.get('user_name', 'User'),
            'password': '',
            'favorites': []
        }
    
    if 'favorites' not in users[email]:
        users[email]['favorites'] = []
        
    if product_id in users[email]['favorites']:
        users[email]['favorites'].remove(product_id)
        action_status = 'removed'
    else:
        users[email]['favorites'].append(product_id)
        action_status = 'added'
        
    save_users(users)
    return jsonify({'status': 'success', 'action': action_status})

# ─── SYSTEM CATALOG SEARCH & VIEW ROUTES ───

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
    products = [enrich_product(p) for pid in ids[:4] for p in PRODUCTS if str(p['id']) == str(pid)]
    brands = get_all_brands()
    all_products_min = [{'id': p['id'], 'name': p['name'], 'brand': p['brand']} for p in PRODUCTS]
    return render_template('compare.html', products=products, brands=brands, all_products=all_products_min)

@app.route('/api/search')
def api_search():
    query = request.args.get('q', '').strip().lower()
    if not query or len(query) < 2: return jsonify([])
    results = [{'id': p['id'], 'name': p['name'], 'brand': p['brand']} 
               for p in PRODUCTS if query in p['name'].lower() or query in p['brand'].lower()][:10]
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
    return jsonify(get_similar(product_id, top_n=6))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)