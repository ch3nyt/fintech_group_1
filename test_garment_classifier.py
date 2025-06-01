#!/usr/bin/env python3
"""
Test script for the comprehensive garment classifier.

This script tests the enhanced garment classification system with real-world
examples and validation scenarios to ensure proper temporal consistency.
"""

import sys
import os
sys.path.append('src')

from utils.garment_classifier import (
    classify_garment,
    ensure_garment_consistency,
    standardize_garment_text,
    find_garment_words,
    GARMENT_CATEGORIES,
    CATEGORY_REPLACEMENTS
)

def test_basic_classification():
    """Test basic garment classification functionality."""
    print("🧪 Testing Basic Classification...")

    test_cases = [
        ("Black skinny jeans with distressed details", "pants"),
        ("White cotton t-shirt with crew neck", "top"),
        ("Red floral maxi dress", "dress"),
        ("Short cocktail dress with sequins", "dress"),
        ("Black leather ankle boots", "shoes"),
        ("White running sneakers", "shoes"),
        ("Lace bra with adjustable straps", "underwear"),
        ("Cotton underwear set", "underwear"),
        ("Modern, sleek black leather belt with silver buckle", "accessories"),
        ("Black bucket hat, ribbed band, button closure", "accessories"),
        ("Stylish handbag with gold chain accents", "accessories"),
        ("Denim shorts with frayed hem", "shorts"),
        ("Navy blue blazer with gold buttons", "outerwear"),
        ("Wool peacoat with double-breasted front", "outerwear"),
        ("Swimsuit with floral print", "underwear"),
        ("Sports bra for athletic activities", "underwear"),
        ("Kimono-style wrap jacket", "outerwear"),
        ("Tank top with lace trim", "top"),
        ("Jumpsuit with wide-leg pants", "dress"),
        ("High-heeled sandals with ankle strap", "shoes")
    ]

    passed = 0
    for text, expected in test_cases:
        result = classify_garment(text)
        if result == expected:
            print(f"  ✅ '{text}' → {result}")
            passed += 1
        else:
            print(f"  ❌ '{text}' → {result} (expected: {expected})")

    print(f"📊 Basic Classification: {passed}/{len(test_cases)} passed\n")

def test_consistency_enforcement():
    """Test garment consistency enforcement."""
    print("🔄 Testing Consistency Enforcement...")

    test_cases = [
        {
            "current": "Modern, sleek black leather belt with silver buckle",
            "next": "Black bucket hat, ribbed band, button closure, unlined"
        },
        {
            "current": "Blue skinny jeans with faded wash",
            "next": "White wide-leg trousers with button closure"
        },
        {
            "current": "White cotton t-shirt with logo print",
            "next": "Black silk blouse with long sleeves"
        },
        {
            "current": "Red floral maxi dress",
            "next": "Black cocktail dress with sequins"
        },
        {
            "current": "Black lace bra with underwire",
            "next": "Cotton sports bra with wide straps"
        },
        {
            "current": "White running sneakers with mesh",
            "next": "Black ankle boots with zipper"
        },
        {
            "current": "Navy blue wool coat",
            "next": "Brown leather jacket with zipper"
        }
    ]

    passed = 0
    for case in test_cases:
        current_category = classify_garment(case["current"])
        next_category = classify_garment(case["next"])

        # Test consistency enforcement
        standardized = ensure_garment_consistency(case["current"], case["next"])
        standardized_category = classify_garment(standardized)

        if standardized_category == current_category:
            print(f"  ✅ {current_category}: '{case['current']}' → '{standardized}'")
            passed += 1
        else:
            print(f"  ❌ Failed to enforce consistency for {current_category}")
            print(f"     Current: '{case['current']}'")
            print(f"     Next: '{case['next']}'")
            print(f"     Standardized: '{standardized}'")
            print(f"     Expected category: {current_category}, Got: {standardized_category}")

    print(f"📊 Consistency Enforcement: {passed}/{len(test_cases)} passed\n")

def test_word_finding():
    """Test garment word detection."""
    print("🔍 Testing Word Finding...")

    test_cases = [
        ("Black skinny jeans with button closure", {"jeans", "button"}),
        ("Red floral maxi dress with long sleeves", {"dress", "sleeves"}),
        ("Leather jacket with zipper and pockets", {"jacket", "leather", "zipper"}),
        ("Sports bra with adjustable straps", {"sports bra", "bra", "straps"}),
        ("High-heeled sandals with gold buckles", {"sandals", "heels", "buckles"})
    ]

    passed = 0
    for text, expected_subset in test_cases:
        found_words = find_garment_words(text)
        if expected_subset.issubset(found_words):
            print(f"  ✅ '{text}' → found {len(found_words)} words including {expected_subset}")
            passed += 1
        else:
            print(f"  ❌ '{text}' → missing expected words")
            print(f"     Expected subset: {expected_subset}")
            print(f"     Found words: {found_words}")

    print(f"📊 Word Finding: {passed}/{len(test_cases)} passed\n")

def test_real_data_scenarios():
    """Test with real data scenarios from captions.json."""
    print("📋 Testing Real Data Scenarios...")

    # Real examples from the dataset and inference logs
    test_cases = [
        {
            "current": "Casual, distressed denim with faded wash and rips at knees and thighs. Denim, blue. Casual wear, denim jeans.",
            "next": "Black, slim-fit pants with zipper closure. Black, smooth fabric pants with white buttons. Smart casual wear for everyday comfort."
        },
        {
            "current": "Floral print, puffy sleeves, ruffled hem, v-neck. Floral pattern, pastel pink with white accents. Dress for special occasions.",
            "next": "Off-white, lace dress with ruffled sleeves. Cream-colored dress with white lace detailing. Casual, feminine dress for daytime wear."
        },
        {
            "current": "Black lace bra with scalloped trim. Lace bra with black lace, white background. Intimates, lace bra.",
            "next": "Sleek, modern tank top with black straps and cross-over front. Black, smooth fabric. Supports athletic activity, provides comfort and style."
        },
        {
            "current": "Modern, sleek black leather belt with silver buckle, loops, and hardware details. Black leather with silver metal accents. Accessory for style and function.",
            "next": "Black bucket hat, ribbed band, button closure, unlined.. Black, textured fabric.. Protect from rain, sun, and wind with style."
        }
    ]

    passed = 0
    for case in test_cases:
        current_category = classify_garment(case["current"])
        standardized = ensure_garment_consistency(case["current"], case["next"])
        standardized_category = classify_garment(standardized)

        if standardized_category == current_category:
            print(f"  ✅ Real scenario {current_category}: consistency maintained")
            print(f"     Standardized: '{standardized[:100]}...'")
            passed += 1
        else:
            print(f"  ❌ Real scenario failed for {current_category}")
            print(f"     Current: '{case['current'][:80]}...'")
            print(f"     Next: '{case['next'][:80]}...'")
            print(f"     Standardized: '{standardized[:80]}...'")
            print(f"     Expected: {current_category}, Got: {standardized_category}")

    print(f"📊 Real Data Scenarios: {passed}/{len(test_cases)} passed\n")

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("⚠️  Testing Edge Cases...")

    test_cases = [
        ("", "unknown"),  # Empty string
        ("This is not clothing related", "unknown"),  # No garment words
        ("Blue and red colors", "unknown"),  # Only colors, no garments
        ("Beautiful fabric texture", "unknown"),  # Only fabric, no garments
        ("Size medium available", "unknown"),  # Only size, no garments
    ]

    passed = 0
    for text, expected in test_cases:
        result = classify_garment(text)
        if result == expected:
            print(f"  ✅ Edge case: '{text}' → {result}")
            passed += 1
        else:
            print(f"  ❌ Edge case: '{text}' → {result} (expected: {expected})")

    print(f"📊 Edge Cases: {passed}/{len(test_cases)} passed\n")

def test_category_coverage():
    """Test that all categories are properly represented."""
    print("📈 Testing Category Coverage...")

    categories = ['top', 'pants', 'dress', 'skirt', 'shoes', 'underwear', 'shorts', 'outerwear', 'accessories']

    print(f"  📊 Total categories: {len(categories)}")
    print(f"  📚 Defined categories: {len(GARMENT_CATEGORIES)}")
    print(f"  🔄 Replacement mappings: {len(CATEGORY_REPLACEMENTS)}")

    # Check that all categories have replacements
    missing_replacements = set(GARMENT_CATEGORIES.keys()) - set(CATEGORY_REPLACEMENTS.keys())
    if missing_replacements:
        print(f"  ❌ Missing replacements for: {missing_replacements}")
    else:
        print(f"  ✅ All categories have replacement mappings")

    # Check category sizes
    for category, terms in GARMENT_CATEGORIES.items():
        print(f"  📝 {category}: {len(terms)} terms")

    print()

def main():
    """Run all tests."""
    print("🧥 Comprehensive Garment Classifier Test Suite")
    print("=" * 50)

    test_basic_classification()
    test_consistency_enforcement()
    test_word_finding()
    test_real_data_scenarios()
    test_edge_cases()
    test_category_coverage()

    print("✨ Test suite completed!")

if __name__ == "__main__":
    main()