
import json
import os
import pickle
import re
from http.server import BaseHTTPRequestHandler

# Load model once at module level (cold start cache)
_bundle = None

def load_bundle():
    global _bundle
    if _bundle is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model_v2.json')
        with open(model_path, 'r') as f:
            _bundle = json.load(f)
    return _bundle

def extract_street(address):
    s = re.sub(r'^(Flat|Unit|Apartment)\s+[^,]+,\s*', '', address, flags=re.IGNORECASE)
    street = re.sub(r',.*$', '', re.sub(r'^\d+[a-zA-Z]?\s*,?\s*', '', s)).lower().strip()
    if street == 'harvist road':
        m = re.match(r'^(\d+)', address)
        if m:
            num = int(m.group(1))
            return 'harvist road west' if num <= 80 else 'harvist road east'
    return street

def get_sector(postcode):
    parts = postcode.strip().split()
    return parts[0] + ' ' + parts[1][0]

CONDITION_ORDER = {'full_renovation':1,'modernisation':2,'cosmetic':3,'good':4,'fully_refurbished':5}

STREET_FEATURES = {
    'all souls avenue':     dict(pct_pre1919=0.54, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'ashburnham road':      dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'buchanan gardens':     dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'chamberlayne road':    dict(pct_pre1919=0.19, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'charteris road':       dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'chevening road':       dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'clifford gardens':     dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'creighton road':       dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'donaldson road':       dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'douglas road':         dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=2, mode_kitchen='rear', mode_loft='bedroom'),
    'doyle gardens':        dict(pct_pre1919=0.38, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'droop street':         dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=2, mode_kitchen='rear', mode_loft='bedroom'),
    'esmond road':          dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'fifth avenue':         dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=2, mode_kitchen='rear', mode_loft='none'),
    'first avenue':         dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'furness road':         dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='unknown'),
    'galton street':        dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'glengall road':        dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'greyhound road':       dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'hanover road':         dict(pct_pre1919=0.60, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'harvist road east':    dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'harvist road west':    dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=2, mode_kitchen='rear', mode_loft='none'),
    'hazelmere road':       dict(pct_pre1919=0.75, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'herbert gardens':      dict(pct_pre1919=0.48, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'honiton road':         dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'hopefield avenue':     dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=2, mode_kitchen='rear', mode_loft='bedroom'),
    'huxley street':        dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'kempe road':           dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=2, mode_kitchen='rear', mode_loft='bedroom'),
    'keslake road':         dict(pct_pre1919=0.94, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'kilburn lane':         dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'kingsley road':        dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='storage'),
    'langler road':         dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'leigh gardens':        dict(pct_pre1919=0.67, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'leighton gardens':     dict(pct_pre1919=0.81, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'liddell gardens':      dict(pct_pre1919=0.73, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'longstone avenue':     dict(pct_pre1919=0.17, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'marne street':         dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'montrose avenue':      dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'mortimer road':        dict(pct_pre1919=0.89, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'nutbourne street':     dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'oliphant street':      dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'peach road':           dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'pember road':          dict(pct_pre1919=0.80, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'priory park road':     dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'rainham road':         dict(pct_pre1919=0.88, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='storage'),
    'ravensworth road':     dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'sixth avenue':         dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'st hildas close':      dict(pct_pre1919=0.00, avg_reception=1, avg_ensuite=1, avg_extension=0, mode_kitchen='rear', mode_loft='none'),
    'summerfield avenue':   dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'tennyson road':        dict(pct_pre1919=1.00, avg_reception=1, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'torbay road':          dict(pct_pre1919=1.00, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
    'treetop mews':         dict(pct_pre1919=0.00, avg_reception=0, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='none'),
    'whitmore gardens':     dict(pct_pre1919=0.36, avg_reception=2, avg_ensuite=0, avg_extension=1, mode_kitchen='rear', mode_loft='bedroom'),
}

def predict(address, postcode, sqft, condition, property_type, bedrooms=None):
    bundle = load_bundle()
    model = bundle['model']
    features = bundle['features']
    street_psf = bundle['street_psf']
    sector_psf = bundle['sector_psf']
    inference_anchors = bundle.get('inference_anchors', {})
    street_lat = bundle['street_lat']
    street_lng = bundle['street_lng']
    street_beds = bundle['street_beds']

    sector = get_sector(postcode)
    street = extract_street(address)
    cond_ord = CONDITION_ORDER.get(condition, 3) if condition else 3
    sp_psf = street_psf.get(street, sector_psf.get(sector, 800))
    anchor_key = street + '|' + condition if condition else None
    anchor = inference_anchors.get(anchor_key, sp_psf) if anchor_key else sp_psf
    lat = street_lat.get(street, 51.53)
    lng = street_lng.get(street, -0.22)
    beds = bedrooms or round(street_beds.get(street, 3))
    sf = STREET_FEATURES.get(street, {})
    era = 'pre_1919' if sf.get('pct_pre1919', 0.8) >= 0.5 else 'post_1919'
    reception = sf.get('avg_reception', 2)
    ensuite = sf.get('avg_ensuite', 0)
    extension = sf.get('avg_extension', 1)
    kitchen = sf.get('mode_kitchen', 'rear')
    loft = sf.get('mode_loft', 'unknown')
    ptype = (property_type or 'terraced_house').lower().replace(' ', '_').replace('-', '_')

    row = {
        'sector': sector, 'street_name': street, 'construction_era': era,
        'property_sub_type': ptype, 'tenure': 'freehold',
        'kitchen_position': kitchen, 'loft_type': loft, 'extension_type': 'unknown',
        'has_floorplan': 1, 'off_street_parking': 0, 'has_garage': 0,
        'has_roof_terrace': 0, 'has_basement': 0, 'has_utility_room': 0,
        'has_ground_floor_wc': 0, 'has_converted_garage': 0,
        'best_sqft': sqft,
        'days_since_2018': (pd.Timestamp('2025-06-01') - pd.Timestamp('2018-01-01')).days,
        'lat': lat, 'lng': lng, 'bedrooms': beds,
        'ensuite_count': ensuite, 'reception_count': reception,
        'sqft_per_bedroom': sqft / max(beds, 1),
        'bath_to_bed': (ensuite + 1) / max(beds, 1),
        'extension_count': extension,
        'street_psf': sp_psf, 'anchor_psf': anchor,
        'condition_ordinal': cond_ord,
        'condition_x_psf': cond_ord * anchor,
    }

    import pandas as pd
    cat_features = ['sector','street_name','construction_era','property_sub_type',
                    'tenure','kitchen_position','loft_type','extension_type']
    pandas_categorical = bundle['model'].get('pandas_categorical', [])
    cat_maps = {}
    for i, cf in enumerate(cat_features):
        if i < len(pandas_categorical):
            cat_maps[cf] = {v: j for j, v in enumerate(pandas_categorical[i])}
    X = pd.DataFrame([row])[features]
    for c in cat_features:
        if c in X.columns:
            if c in cat_maps:
                X[c] = X[c].map(cat_maps[c]).fillna(-1).astype(int)
            else:
                X[c] = X[c].astype('category')
    log_pred = model.predict(X)[0]
    estimate = int(np.exp(log_pred))

    # Confidence interval: +/- 10% (roughly 1 std dev of model error)
    low = int(estimate * 0.91)
    high = int(estimate * 1.09)

    return {
        'estimate': estimate,
        'low': low,
        'high': high,
        'sector': sector,
        'street': street,
        'era': era,
        'streetAnchorPsf': round(sp_psf),
        'conditionAnchorPsf': round(anchor),
        'model': 'v2',
    }

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
            address = data.get('address', '')
            postcode = data.get('postcode', '')
            sqft = float(data.get('sqft', 0))
            condition = data.get('condition')
            property_type = data.get('type', 'Terraced house')
            bedrooms = data.get('bedrooms')
            if bedrooms:
                bedrooms = int(bedrooms)

            if not address or not postcode or not sqft:
                self._respond(400, {'error': 'address, postcode and sqft required'})
                return

            result = predict(address, postcode, sqft, condition, property_type, bedrooms)
            self._respond(200, result)
        except Exception as e:
            import traceback
            self._respond(500, {'error': str(e), 'traceback': traceback.format_exc()})

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _respond(self, code, data):
        self.send_response(code)
        self._cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass
