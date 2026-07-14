
import json
import os
import ctypes
import ctypes.util

# Vercel OpenMP workaround for LightGBM
try:
    ctypes.CDLL('libgomp.so.1')
except:
    try:
        import subprocess
        subprocess.run(['apt-get', 'install', '-y', 'libgomp1'], capture_output=True)
    except:
        pass
import re
import statistics
from http.server import BaseHTTPRequestHandler

import json as _json
import numpy as _np
import numpy as np

# Load model once at module level (cold start cache)
_bundle = None

def load_bundle():
    global _bundle
    if _bundle is None:
        model_path = os.path.join(os.path.dirname(__file__), 'model_v2.json')
        with open(model_path, 'r') as f:
            _bundle = _json.load(f)
    return _bundle

def _predict_tree(tree, x):
    node = tree['tree_structure']
    while 'split_feature' in node:
        feat_idx = node['split_feature']
        val = x[feat_idx]
        threshold = node['threshold']
        if node.get('decision_type', '<=') == '==':
            go_left = str(int(val)) in [c.strip() for c in str(threshold).split('||')]
        else:
            go_left = val <= float(threshold)
        node = node['left_child'] if go_left else node['right_child']
    return node['leaf_value']

def _run_model(tree_info, vec):
    """Walks a list of trees (tree_info) and sums leaf values.
    Works for the main model OR either quantile model -- just
    pass in the right tree_info list."""
    total = 0.0
    for tree in tree_info:
        total += _predict_tree(tree, vec)
    return total

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

CONDITION_ORDER = {'full_renovation':1,'modernisation':2,'good':3,'fully_refurbished':4}

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

def _normalize_address(addr):
    return re.sub(r'\s+', ' ', (addr or '').strip().lower())

def select_comps(bundle, street, sector, condition, sqft, limit=6, subject_address=None):
    """Real comparable sales for this property: same street preferred,
    falls back to sector if the street has too few. Sorted so the most
    relevant (matching condition, closest sqft, most recent) come first.
    Excludes the subject property's own past sale if it happens to be in
    the training data. Returns (comps_list, pool_size, median_price, scope)."""
    all_comps = bundle.get('comps', [])
    subject_norm = _normalize_address(subject_address) if subject_address else None
    if subject_norm:
        all_comps = [c for c in all_comps if _normalize_address(c.get('address')) != subject_norm]
    street_comps = [c for c in all_comps if c.get('street') == street]
    if len(street_comps) >= 3:
        pool = street_comps
        scope = 'street'
    else:
        pool = [c for c in all_comps if c.get('sector') == sector]
        scope = 'sector'

    def sort_key(c):
        cond_match = 0 if c.get('condition') == condition else 1
        sqft_diff = abs((c.get('sqft') or 0) - (sqft or 0)) if sqft else 0
        # ISO date strings sort correctly as strings -- most recent first
        recency = c.get('date') or ''
        return (cond_match, sqft_diff, ''.join(chr(255 - ord(ch)) for ch in recency))

    pool_sorted = sorted(pool, key=sort_key)
    top = pool_sorted[:limit]
    prices = [c['price'] for c in pool if c.get('price')]
    median_price = int(statistics.median(prices)) if prices else None

    return top, len(pool), median_price, scope

CONDITIONS_IN_ORDER = ['full_renovation', 'modernisation', 'good', 'fully_refurbished']

def predict(address, postcode, sqft, condition, property_type, bedrooms=None):
    bundle = load_bundle()
    features = bundle['features']
    street_psf = bundle['street_psf']
    sector_psf = bundle['sector_psf']
    inference_anchors = bundle.get('inference_anchors', {})
    street_lat = bundle['street_lat']
    street_lng = bundle['street_lng']
    street_beds = bundle['street_beds']

    sector = get_sector(postcode)
    street = extract_street(address)
    sp_psf = street_psf.get(street, sector_psf.get(sector, 800))
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

    cat_feature_names = ['sector','street_name','construction_era','property_sub_type',
                          'tenure','kitchen_position','loft_type','extension_type']
    pandas_categorical = bundle['model'].get('pandas_categorical', [])
    cat_encodings = {}
    for i, cf in enumerate(cat_feature_names):
        if i < len(pandas_categorical):
            cat_encodings[cf] = {v: j for j, v in enumerate(pandas_categorical[i])}

    anchors_by_cond = {}
    def build_row_and_vec(cond):
        cond_ord = CONDITION_ORDER.get(cond, 3) if cond else 3
        anchor_key = street + '|' + cond if cond else None
        anchor = inference_anchors.get(anchor_key, sp_psf) if anchor_key else sp_psf
        # If full_renovation has no anchor, cap it below modernisation
        if cond == 'full_renovation' and anchor_key not in inference_anchors:
            mod_anchor = inference_anchors.get(street + '|modernisation', sp_psf)
            anchor = min(anchor, mod_anchor * 0.93)
        anchors_by_cond[cond] = anchor
        row = {
            'sector': sector, 'street_name': street, 'construction_era': era,
            'property_sub_type': ptype, 'tenure': 'freehold',
            'kitchen_position': kitchen, 'loft_type': loft, 'extension_type': 'unknown',
            'has_floorplan': 1, 'off_street_parking': 0, 'has_garage': 0,
            'has_roof_terrace': 0, 'has_basement': 0, 'has_utility_room': 0,
            'has_ground_floor_wc': 0, 'has_converted_garage': 0,
            'best_sqft': sqft,
            'days_since_2018': 2708,  # days from 2018-01-01 to 2025-06-01
            'lat': lat, 'lng': lng, 'bedrooms': beds,
            'ensuite_count': ensuite, 'reception_count': reception,
            'sqft_per_bedroom': sqft / max(beds, 1),
            'bath_to_bed': (ensuite + 1) / max(beds, 1),
            'extension_count': extension,
            'street_psf': sp_psf, 'anchor_psf': anchor,
            'condition_ordinal': cond_ord,
            'condition_x_psf': cond_ord * anchor,
        }
        vec = []
        for f in features:
            val = row.get(f, 0)
            if f in cat_encodings:
                enc = cat_encodings[f]
                vec.append(float(enc.get(str(val), -1)))
            else:
                vec.append(float(val) if val is not None else 0.0)
        return vec

    # Compute the raw model estimate for EVERY condition, then force the
    # sequence to be non-decreasing (full_renovation -> ... -> fully_refurbished).
    # This guarantees a "better" condition can never price lower than a
    # "worse" one for the same property, which the raw per-condition street
    # anchors cannot guarantee on their own (thin/noisy comps per tier).
    raw_estimates = {}
    vecs = {}
    for cond in CONDITIONS_IN_ORDER:
        vec = build_row_and_vec(cond)
        vecs[cond] = vec
        log_pred = _run_model(bundle['model']['tree_info'], vec)
        raw_estimates[cond] = float(np.exp(log_pred))

    corrected = {}
    running_max = 0.0
    for cond in CONDITIONS_IN_ORDER:
        running_max = max(running_max, raw_estimates[cond])
        corrected[cond] = running_max

    estimate = corrected[condition] if condition in corrected else corrected['good']
    raw_estimate_for_requested = raw_estimates.get(condition, estimate)
    correction_delta = estimate - raw_estimate_for_requested
    vec = vecs.get(condition, vecs['good'])
    anchor = anchors_by_cond.get(condition, anchors_by_cond['good'])

    # Real calibrated range using the P10/P90 quantile models, matching
    # the 1.4x band-width multiplier and safety clamp proven at 81%
    # real coverage against the 100-property test set. Falls back to
    # the old +/-9% behaviour only if quantile trees aren't present
    # in model_v2.json (e.g. an older deploy). The band is computed
    # against the RAW estimate then shifted by the same monotonic
    # correction applied to the point estimate, so width is preserved.
    if 'quantile_models' in bundle:
        p10_log = _run_model(bundle['quantile_models']['p10']['tree_info'], vec)
        p90_log = _run_model(bundle['quantile_models']['p90']['tree_info'], vec)
        p10_raw = float(np.exp(p10_log))
        p90_raw = float(np.exp(p90_log))
        multiplier = bundle.get('band_width_multiplier', 1.0)

        # Scale the band as a PERCENTAGE of the raw estimate, then apply that
        # same percentage to the corrected estimate -- rather than shifting an
        # absolute £ offset, which can push the band past its own edge when
        # the monotonic correction is large relative to the original band
        # (this was collapsing some bands to zero width, e.g. 84 Oliphant St).
        lower_gap_pct = max(0.0, (raw_estimate_for_requested - p10_raw) / raw_estimate_for_requested) if raw_estimate_for_requested else 0.0
        upper_gap_pct = max(0.0, (p90_raw - raw_estimate_for_requested) / raw_estimate_for_requested) if raw_estimate_for_requested else 0.0
        # Quantile crossing guard: the P10/P90 models are trained
        # independently from the main model, so nothing stops P90 from
        # predicting BELOW (or P10 from predicting ABOVE) the main
        # model's own point estimate on some inputs (seen on 84 Oliphant
        # St -- P90 came in under the estimate for 'good' and
        # 'fully_refurbished'). When that happens, fall back to a
        # symmetric band using whichever side's gap IS trustworthy,
        # rather than silently collapsing that side to zero width.
        if upper_gap_pct <= 0 and lower_gap_pct > 0:
            upper_gap_pct = lower_gap_pct
        elif lower_gap_pct <= 0 and upper_gap_pct > 0:
            lower_gap_pct = upper_gap_pct
        low = estimate * (1 - multiplier * lower_gap_pct)
        high = estimate * (1 + multiplier * upper_gap_pct)
        # Safety clamp: estimate is ALWAYS inside the band, band is NEVER negative
        low = max(0.0, min(low, estimate))
        high = max(high, estimate)
    else:
        low = estimate * 0.91
        high = estimate * 1.09

    estimate = int(estimate)
    low = int(low)
    high = int(high)

    comps_top, pool_size, median_price, comp_scope = select_comps(
        bundle, street, sector, condition, sqft, subject_address=address
    )

    if pool_size >= 15:
        confidence = 'High'
    elif pool_size >= 5:
        confidence = 'Medium'
    else:
        confidence = 'Low'

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
        'poolSize': pool_size,
        'median': median_price,
        'comparables': comps_top,
        'comparablesScope': comp_scope,
        'psf': round(estimate / sqft) if sqft else None,
        'sectorPsf': round(sector_psf.get(sector, 800)),
        'confidence': confidence,
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
            self._respond(500, {'error': str(e)})

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
