"""
Garment classification system for temporal fashion prediction.

This module provides functionality to classify and standardize garment descriptions
into consistent categories to ensure temporal coherence in fashion predictions.
For example, categorizes 'leggings', 'jeans', 'trousers' all as 'pants'
to maintain category consistency across time periods.
"""

import re
from typing import Dict, List, Set

# Define garment category mappings
GARMENT_CATEGORIES = {
    # Top category - expanded to include more specific terms
    'top': [
        'top', 'shirt', 'blouse', 't-shirt', 'tee', 'tank', 'cami',
        'sweater', 'cardigan', 'hoodie', 'vest', 'crop top', 'tunic',
        'polo', 'henley', 'bodysuit', 'camisole', 'bustier', 'tube top',
        'halter', 'bandeau', 'muscle tee', 'tank top', 'sleeveless top',
        'long sleeve top', 'short sleeve top', 'three-quarter sleeve',
        'turtleneck', 'mock neck', 'crew neck', 'v-neck', 'scoop neck',
        'off-shoulder', 'one-shoulder', 'strapless top', 'wrap top',
        'peplum', 'ruffle top', 'lace top', 'mesh top', 'sheer top',
        'graphic tee', 'printed top', 'striped top', 'solid top',
        'fitted top', 'loose top', 'oversized top', 'cropped shirt',
        'button-up', 'button-down', 'pullover', 'sweatshirt'
    ],

    # Pants category
    'pants': [
        'pants', 'trousers', 'jeans', 'leggings', 'joggers', 'sweatpants',
        'chinos', 'slacks', 'cargo pants', 'wide leg', 'skinny pants',
        'straight leg', 'bootcut', 'flare', 'capri', 'cropped pants',
        'high-waisted pants', 'low-rise pants', 'ankle pants', 'track pants',
        'palazzo pants', 'culottes', 'bell bottoms', 'boyfriend jeans',
        'mom jeans', 'skinny jeans', 'ripped jeans', 'distressed jeans',
        'stretch pants', 'dress pants', 'formal pants', 'casual pants',
        'denim', 'corduroy pants', 'linen pants', 'khakis'
    ],

    # Skirt category
    'skirt': [
        'skirt', 'mini skirt', 'maxi skirt', 'pencil skirt', 'a-line skirt',
        'pleated skirt', 'wrap skirt', 'flared skirt', 'circle skirt',
        'midi skirt', 'knee-length skirt', 'asymmetrical skirt', 'tiered skirt',
        'ruffled skirt', 'denim skirt', 'leather skirt', 'tulle skirt'
    ],

    # Dress category
    'dress': [
        'dress', 'gown', 'sundress', 'maxi dress', 'mini dress',
        'midi dress', 'cocktail dress', 'formal dress', 'casual dress',
        'wrap dress', 'shift dress', 'bodycon dress', 'a-line dress',
        'fit and flare', 'sheath dress', 'shirt dress', 'sweater dress',
        'slip dress', 'off-shoulder dress', 'strapless dress', 'halter dress',
        'long-sleeved dress', 'sleeveless dress', 'empire waist dress',
        'high-low dress', 'asymmetrical dress', 'backless dress',
        'cut-out dress', 'lace dress', 'floral dress', 'striped dress',
        'solid dress', 'printed dress', 'evening gown', 'ball gown',
        'jumpsuit', 'romper', 'overall dress', 'pinafore'
    ],

    # Shoes category
    'shoes': [
        'shoes', 'sneakers', 'boots', 'sandals', 'heels', 'flats',
        'loafers', 'oxfords', 'pumps', 'stilettos', 'wedges',
        'ankle boots', 'knee boots', 'running shoes', 'athletic shoes',
        'high-top sneakers', 'low-top sneakers', 'slip-on shoes',
        'lace-up shoes', 'mary janes', 'ballet flats', 'platform shoes',
        'combat boots', 'chelsea boots', 'hiking boots', 'rain boots',
        'flip-flops', 'slides', 'espadrilles', 'clogs', 'moccasins',
        'dress shoes', 'casual shoes', 'formal shoes', 'work boots',
        'steel-toe boots', 'riding boots', 'cowboy boots', 'thigh-high boots'
    ],

    # Underwear category - expanded significantly based on dataset
    'underwear': [
        'bra', 'panties', 'underwear', 'lingerie', 'thong', 'briefs',
        'boxers', 'sports bra', 'camisole', 'bodysuit', 'teddy',
        'push-up bra', 'strapless bra', 'wireless bra', 'bralette',
        'triangle bra', 'full-coverage bra', 'demi bra', 'plunge bra',
        'racerback bra', 'convertible bra', 'nursing bra', 'maternity bra',
        'bikini', 'swimsuit', 'bathing suit', 'swimwear', 'tankini',
        'one-piece swimsuit', 'two-piece swimsuit', 'bikini top', 'bikini bottom',
        'swim trunks', 'board shorts', 'rash guard', 'swim dress',
        'chemise', 'slip', 'nightgown', 'pajamas', 'pyjamas', 'sleepwear',
        'robe', 'bathrobe', 'negligee', 'babydoll', 'corset', 'bustier',
        'garter belt', 'hosiery', 'tights', 'stockings', 'pantyhose',
        'thermal underwear', 'long johns', 'undershirt', 'tank undershirt'
    ],

    # Shorts category
    'shorts': [
        'shorts', 'short pants', 'bermuda shorts', 'cargo shorts',
        'denim shorts', 'athletic shorts', 'running shorts', 'gym shorts',
        'board shorts', 'swim shorts', 'hot pants', 'high-waisted shorts',
        'bike shorts', 'cycling shorts', 'compression shorts', 'basketball shorts',
        'tennis shorts', 'golf shorts', 'chino shorts', 'khaki shorts',
        'linen shorts', 'cotton shorts', 'drawstring shorts', 'elastic waist shorts'
    ],

    # Outerwear category - significantly expanded
    'outerwear': [
        'jacket', 'coat', 'blazer', 'windbreaker', 'raincoat', 'parka',
        'trench coat', 'bomber', 'denim jacket', 'leather jacket',
        'wool coat', 'peacoat', 'overcoat', 'topcoat', 'anorak',
        'puffer jacket', 'down jacket', 'quilted jacket', 'fleece jacket',
        'softshell jacket', 'hardshell jacket', 'ski jacket', 'snowboard jacket',
        'motorcycle jacket', 'biker jacket', 'varsity jacket', 'track jacket',
        'windbreaker jacket', 'rain jacket', 'waterproof jacket',
        'insulated jacket', 'thermal jacket', 'hiking jacket', 'outdoor jacket',
        'blazer jacket', 'suit jacket', 'sport coat', 'dinner jacket',
        'tuxedo jacket', 'formal jacket', 'casual jacket', 'cropped jacket',
        'longline coat', 'midi coat', 'maxi coat', 'wrap coat',
        'belted coat', 'hooded coat', 'fur coat', 'faux fur coat',
        'shearling coat', 'cashmere coat', 'tweed jacket', 'corduroy jacket',
        'velvet jacket', 'satin jacket', 'kimono', 'poncho', 'cape', 'shawl'
    ],

    # Accessories category - greatly expanded based on dataset findings
    'accessories': [
        'belt', 'hat', 'cap', 'scarf', 'bag', 'handbag', 'purse', 'backpack',
        'wallet', 'watch', 'bracelet', 'necklace', 'earrings', 'ring',
        'sunglasses', 'glasses', 'gloves', 'tie', 'bow tie', 'suspenders',
        'bucket hat', 'beanie', 'baseball cap', 'tote', 'crossbody', 'clutch',
        'fedora', 'beret', 'visor', 'headband', 'hair accessory', 'scrunchie',
        'straw hat', 'sun hat', 'winter hat', 'knit cap', 'newsboy cap',
        'panama hat', 'cowboy hat', 'trucker hat', 'snapback', 'fitted cap',
        'shoulder bag', 'messenger bag', 'duffle bag', 'gym bag', 'laptop bag',
        'evening bag', 'wristlet', 'coin purse', 'fanny pack', 'belt bag',
        'satchel', 'hobo bag', 'bucket bag', 'drawstring bag', 'makeup bag',
        'travel bag', 'weekender bag', 'diaper bag', 'school bag',
        'leather belt', 'chain belt', 'fabric belt', 'elastic belt',
        'statement belt', 'skinny belt', 'wide belt', 'waist belt',
        'silk scarf', 'wool scarf', 'cashmere scarf', 'infinity scarf',
        'blanket scarf', 'lightweight scarf', 'winter scarf', 'pashmina',
        'bandana', 'neckerchief', 'pocket square', 'hair scarf',
        'jewelry', 'costume jewelry', 'fine jewelry', 'fashion jewelry',
        'pendant', 'chain', 'choker', 'statement necklace', 'tennis bracelet',
        'bangle', 'charm bracelet', 'cuff bracelet', 'stackable rings',
        'statement earrings', 'stud earrings', 'hoop earrings', 'drop earrings',
        'reading glasses', 'prescription glasses', 'safety glasses',
        'contact lenses', 'aviator sunglasses', 'cat-eye sunglasses',
        'oversized sunglasses', 'round sunglasses', 'square sunglasses',
        'winter gloves', 'driving gloves', 'work gloves', 'mittens',
        'fingerless gloves', 'leather gloves', 'knit gloves', 'rubber gloves'
    ]
}

# Define standard replacements for each category
CATEGORY_REPLACEMENTS = {
    'top': 'top',
    'pants': 'pants',
    'skirt': 'skirt',
    'dress': 'dress',
    'shoes': 'shoes',
    'underwear': 'underwear',
    'shorts': 'shorts',
    'outerwear': 'jacket',
    'accessories': 'accessories'
}

def classify_garment(text: str) -> str:
    """
    Classify a garment description into one of the main categories.

    Args:
        text: The garment description text

    Returns:
        The identified garment category or 'unknown' if no category matches
    """
    text_lower = text.lower()

    # Count matches for each category
    category_scores = {}
    for category, terms in GARMENT_CATEGORIES.items():
        score = 0
        for term in terms:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text_lower):
                score += 1
        category_scores[category] = score

    # Return the category with the highest score
    if any(score > 0 for score in category_scores.values()):
        return max(category_scores, key=category_scores.get)

    return 'unknown'

def find_garment_words(text: str) -> Set[str]:
    """
    Find all garment-related words in the text that match any category.

    Args:
        text: The text to search

    Returns:
        Set of garment words found in the text
    """
    text_lower = text.lower()
    found_words = set()

    for category, terms in GARMENT_CATEGORIES.items():
        for term in terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text_lower):
                found_words.add(term)

    return found_words

def standardize_garment_text(current_text: str, target_category: str) -> str:
    """
    Standardize garment text to use consistent terminology for the target category.

    Args:
        current_text: The text to standardize
        target_category: The target garment category to standardize to

    Returns:
        Text with garment words replaced to match the target category
    """
    if target_category not in CATEGORY_REPLACEMENTS:
        return current_text

    # Find all garment words that need replacement (from any category)
    garment_words = find_garment_words(current_text)

    # Replace with appropriate term for target category
    replacement_word = CATEGORY_REPLACEMENTS[target_category]
    result_text = current_text

    # Replace any garment words with the target category's standard term
    for word in garment_words:
        # Use word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(word) + r'\b'
        result_text = re.sub(pattern, replacement_word, result_text, flags=re.IGNORECASE)

    return result_text

def ensure_garment_consistency(current_week_text: str, next_week_text: str) -> str:
    """
    Ensure that next week's text uses the same garment category as current week.

    Args:
        current_week_text: Current week's garment description
        next_week_text: Next week's garment description to be standardized

    Returns:
        Modified next week text that uses the same garment category
    """
    # Classify current week's garment
    current_category = classify_garment(current_week_text)

    if current_category == 'unknown':
        return next_week_text

    # Standardize next week's text to match current week's category
    standardized_text = standardize_garment_text(next_week_text, current_category)

    return standardized_text