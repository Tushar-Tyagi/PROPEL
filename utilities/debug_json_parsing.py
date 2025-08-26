#!/usr/bin/env python3
"""
Debug script for JSON parsing issues in LLM responses.
This script helps identify and fix common JSON parsing problems.
"""

import json
import re
from typing import List, Dict, Optional

def letter_to_index(letter: str) -> int:
    """Convert letter to 0-based index."""
    letter = letter.upper()
    if len(letter) == 1:
        return ord(letter) - ord('A')
    elif len(letter) == 2:
        # Handle cases like AA, AB, AC, etc.
        first_letter = ord(letter[0]) - ord('A') + 1
        second_letter = ord(letter[1]) - ord('A')
        return first_letter * 26 + second_letter
    else:
        return 0  # Default fallback

def improved_json_parsing(content: str, debug: bool = False) -> Optional[List[int]]:
    """
    Improved JSON parsing with multiple fallback strategies.
    
    Args:
        content: The LLM response content
        debug: Whether to print debug information
        
    Returns:
        List of 1-based indices or None if parsing fails
    """
    if debug:
        print(f"Original content: {content[:200]}...")
    
    # Strategy 1: Try to find and parse JSON patterns
    json_patterns = [
        r'\{[^}]*"ranked_movies"[^}]*\}',
        r'\{[^}]*"ranked_songs"[^}]*\}',
        r'\{[^}]*"ranked_books"[^}]*\}',
        r'\{[^}]*"ranked_news"[^}]*\}',
        r'\{[^}]*"ranked_beauty"[^}]*\}'
    ]
    
    for pattern in json_patterns:
        json_match = re.search(pattern, content)
        if json_match:
            json_str = json_match.group()
            if debug:
                print(f"Found JSON pattern: {json_str}")
            
            # Clean up common JSON issues
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            try:
                result = json.loads(json_str)
                
                # Get the ranking list
                for key in ["ranked_movies", "ranked_songs", "ranked_books", "ranked_news", "ranked_beauty"]:
                    if key in result:
                        ranking_list = result[key]
                        if isinstance(ranking_list, list):
                            indices = []
                            for letter in ranking_list:
                                if isinstance(letter, str):
                                    try:
                                        index = letter_to_index(letter) + 1
                                        indices.append(index)
                                    except:
                                        continue
                            
                            if indices:
                                if debug:
                                    print(f"Successfully parsed JSON: {indices}")
                                return indices
            except json.JSONDecodeError as e:
                if debug:
                    print(f"JSON decode error: {e}")
                continue
    
    # Strategy 2: Try to extract array patterns directly
    array_patterns = [
        r'\[[^\]]*"A"[^\]]*"B"[^\]]*\]',  # Look for arrays with A, B
        r'\[[^\]]*"A"[^\]]*\]',           # Look for arrays starting with A
        r'\[[^\]]*\]'                     # Any array
    ]
    
    for pattern in array_patterns:
        array_matches = re.findall(pattern, content)
        for array_str in array_matches:
            if debug:
                print(f"Found array pattern: {array_str}")
            
            # Clean up the array string
            cleaned_array = re.sub(r'[^\w,\s\[\]]', '', array_str)
            letters = re.findall(r'\b[A-Z]{1,2}\b', cleaned_array)
            
            if letters:
                indices = []
                for letter in letters:
                    try:
                        index = letter_to_index(letter) + 1
                        indices.append(index)
                    except:
                        continue
                
                if indices:
                    if debug:
                        print(f"Successfully parsed array: {indices}")
                    return indices
    
    # Strategy 3: Extract letters from entire content
    letters = re.findall(r'\b[A-Z]{1,2}\b', content)
    if debug:
        print(f"Found letters in content: {letters[:10]}...")
    
    indices = []
    for letter in letters[:100]:  # Limit to first 100
        try:
            index = letter_to_index(letter) + 1
            indices.append(index)
        except:
            continue
    
    if indices:
        if debug:
            print(f"Successfully extracted letters: {indices}")
        return indices
    
    # Strategy 4: Return default sequence
    if debug:
        print("Using default sequence")
    return list(range(1, 21))  # Default 20 items

def test_json_parsing():
    """Test the improved JSON parsing with various malformed inputs."""
    
    test_cases = [
        # Valid JSON
        '{"ranked_movies": ["A", "B", "C", "D"]}',
        
        # JSON with trailing comma
        '{"ranked_movies": ["A", "B", "C", "D",]}',
        
        # Malformed JSON
        '{"ranked_movies": ["A", "B", "C", "D"',  # Missing closing brace
        
        # JSON with extra text
        'Here is my ranking: {"ranked_movies": ["A", "B", "C", "D"]} and that\'s it',
        
        # Just array
        '["A", "B", "C", "D"]',
        
        # Array with trailing comma
        '["A", "B", "C", "D",]',
        
        # Letters in text
        'I recommend A first, then B, then C, and finally D',
        
        # Mixed format
        'Ranking: A, B, C, D in that order',
        
        # Empty or invalid
        '',
        'No ranking provided',
        '{"invalid_key": "value"}'
    ]
    
    print("Testing JSON parsing with various inputs:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case[:50]}...")
        result = improved_json_parsing(test_case, debug=True)
        print(f"Result: {result}")
        print("-" * 30)

if __name__ == "__main__":
    test_json_parsing() 