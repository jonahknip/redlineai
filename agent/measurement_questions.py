"""
Measurement Questions Module
Generates specific questions for REVIEW items that need user measurements.
Re-evaluates items based on user-provided answers.
"""

import re
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Question type definitions
QUESTION_TYPES = {
    'distance': {
        'template': "What is the distance between {element_a} and {element_b} at {location}?",
        'unit': 'feet',
        'validation': 'numeric'
    },
    'slope': {
        'template': "What is the slope at {location}?",
        'unit': '%',
        'validation': 'percentage'
    },
    'elevation': {
        'template': "What is the elevation at {location}?",
        'unit': 'feet',
        'validation': 'numeric'
    },
    'clearance': {
        'template': "What is the {clearance_type} clearance between {element_a} and {element_b}?",
        'unit': 'inches',
        'validation': 'numeric'
    },
    'quantity': {
        'template': "How many {item} are shown on {location}?",
        'unit': 'count',
        'validation': 'integer'
    },
    'dimension': {
        'template': "What is the {dimension_type} of {element} at {location}?",
        'unit': 'varies',
        'validation': 'numeric'
    },
    'confirmation': {
        'template': "Is {condition}?",
        'unit': 'Yes/No',
        'validation': 'boolean'
    },
    'k_value': {
        'template': "What is the K value for the vertical curve at {location}?",
        'unit': 'K',
        'validation': 'numeric'
    },
    'radius': {
        'template': "What is the curve radius at {location}?",
        'unit': 'feet',
        'validation': 'numeric'
    }
}

# Patterns to detect what type of measurement is needed based on checklist item text
MEASUREMENT_PATTERNS = [
    # Distance/separation
    (r'separation|distance|spacing|apart|between', 'distance'),
    (r'\d+\s*(?:feet|ft|\')\s*(?:from|between|separation)', 'distance'),
    
    # Slope
    (r'slope|grade|%.*slope|slope.*%|running slope|cross slope', 'slope'),
    (r'(?:less than|greater than|<|>)\s*\d+\s*%', 'slope'),
    
    # Elevation
    (r'elevation|invert|rim|top of|bottom of', 'elevation'),
    
    # Clearance  
    (r'clearance|vertical separation|horizontal separation|cover', 'clearance'),
    (r'\d+\s*(?:inches|in|")\s*(?:clearance|separation|cover)', 'clearance'),
    
    # Dimensions
    (r'width|height|length|diameter|thickness|depth', 'dimension'),
    (r'handrail.*(?:diameter|height|extension)', 'dimension'),
    
    # K-value
    (r'k\s*value|k-value|vertical curve', 'k_value'),
    
    # Radius
    (r'radius|curve.*radius|horizontal curve', 'radius'),
    
    # Quantity
    (r'number of|count|how many|quantity', 'quantity'),
    
    # Confirmation
    (r'approved|verified|confirmed|signed|sealed', 'confirmation'),
]

# Threshold patterns to extract from checklist items
THRESHOLD_PATTERNS = [
    (r'(?:>=?|greater than|at least|minimum)\s*(\d+(?:\.\d+)?)\s*(?:feet|ft|\'|inches|in|"|%)?', '>='),
    (r'(?:<=?|less than|maximum|no more than)\s*(\d+(?:\.\d+)?)\s*(?:feet|ft|\'|inches|in|"|%)?', '<='),
    (r'(\d+(?:\.\d+)?)\s*(?:feet|ft|\'|inches|in|"|%)?\s*(?:minimum|min)', '>='),
    (r'(\d+(?:\.\d+)?)\s*(?:feet|ft|\'|inches|in|"|%)?\s*(?:maximum|max)', '<='),
    (r'(?:between|range)\s*(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)', 'range'),
]


def detect_question_type(item_text: str) -> Optional[str]:
    """Detect what type of measurement question is needed based on item text."""
    text_lower = item_text.lower()
    
    for pattern, q_type in MEASUREMENT_PATTERNS:
        if re.search(pattern, text_lower):
            return q_type
    
    return None


def extract_threshold(item_text: str) -> Optional[Dict[str, Any]]:
    """Extract threshold requirements from checklist item text."""
    text_lower = item_text.lower()
    
    for pattern, comparison in THRESHOLD_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            if comparison == 'range':
                return {
                    'type': 'range',
                    'min': float(match.group(1)),
                    'max': float(match.group(2))
                }
            else:
                return {
                    'type': comparison,
                    'value': float(match.group(1))
                }
    
    return None


def generate_question_for_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate a measurement question for a checklist item if applicable.
    
    Args:
        item: Dict with 'id', 'text', 'status', 'comment', etc.
    
    Returns:
        Question dict or None if no question needed
    """
    # Only generate questions for REVIEW items
    if item.get('status') != 'REVIEW':
        return None
    
    item_text = item.get('text', '')
    comment = item.get('comment', '')
    
    # Detect question type
    q_type = detect_question_type(item_text)
    if not q_type:
        return None
    
    # Extract threshold
    threshold = extract_threshold(item_text)
    
    # Build question text
    q_config = QUESTION_TYPES.get(q_type, {})
    
    if q_type == 'distance':
        question_text = f"What is the measured {item_text.split()[0].lower()} distance/separation? (in feet)"
    elif q_type == 'slope':
        question_text = f"What is the slope mentioned in this item? (as percentage, e.g., 4.5)"
    elif q_type == 'clearance':
        question_text = f"What is the clearance/separation? (in inches)"
    elif q_type == 'dimension':
        question_text = f"What is the dimension specified? (include units)"
    elif q_type == 'elevation':
        question_text = f"What is the elevation? (in feet)"
    elif q_type == 'k_value':
        question_text = "What is the K value?"
    elif q_type == 'radius':
        question_text = "What is the curve radius? (in feet)"
    elif q_type == 'quantity':
        question_text = "How many are shown?"
    elif q_type == 'confirmation':
        question_text = f"Can you confirm this requirement is met? (Yes/No)"
    else:
        question_text = f"Please provide the measurement for: {item_text}"
    
    # Format threshold for display
    threshold_display = None
    if threshold:
        if threshold['type'] == 'range':
            threshold_display = f"Must be between {threshold['min']} and {threshold['max']}"
        elif threshold['type'] == '>=':
            threshold_display = f"Must be >= {threshold['value']}"
        elif threshold['type'] == '<=':
            threshold_display = f"Must be <= {threshold['value']}"
    
    return {
        'question_type': q_type,
        'question_text': question_text,
        'expected_unit': q_config.get('unit', 'varies'),
        'threshold': threshold,
        'threshold_display': threshold_display,
        'validation': q_config.get('validation', 'numeric'),
        'page_refs': item.get('page_refs', [])
    }


def add_questions_to_eval_data(eval_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add measurement questions to REVIEW items in eval_data.
    
    Args:
        eval_data: Dict mapping item_id to item data
    
    Returns:
        Updated eval_data with 'questions' field for applicable items
    """
    for item_id, item in eval_data.items():
        question = generate_question_for_item(item)
        if question:
            item['questions'] = [question]
    
    return eval_data


def parse_answer(answer: str, validation_type: str) -> Any:
    """
    Parse user's answer string into appropriate type.
    
    Args:
        answer: User's input string
        validation_type: 'numeric', 'percentage', 'integer', 'boolean'
    
    Returns:
        Parsed value or None if invalid
    """
    answer = answer.strip().lower()
    
    if validation_type == 'boolean':
        if answer in ['yes', 'y', 'true', '1', 'confirmed']:
            return True
        elif answer in ['no', 'n', 'false', '0']:
            return False
        return None
    
    elif validation_type == 'integer':
        # Extract number from answer
        match = re.search(r'(\d+)', answer)
        if match:
            return int(match.group(1))
        return None
    
    else:  # numeric or percentage
        # Remove common units and extract number
        cleaned = re.sub(r'[^\d.\-]', '', answer.replace(',', ''))
        try:
            return float(cleaned)
        except ValueError:
            return None


def evaluate_against_threshold(value: Any, threshold: Dict[str, Any]) -> bool:
    """
    Compare a value against a threshold.
    
    Args:
        value: Parsed numeric value
        threshold: Dict with 'type' and 'value' or 'min'/'max'
    
    Returns:
        True if passes threshold, False otherwise
    """
    if value is None or threshold is None:
        return None  # Cannot determine
    
    threshold_type = threshold.get('type')
    
    if threshold_type == '>=':
        return value >= threshold['value']
    elif threshold_type == '<=':
        return value <= threshold['value']
    elif threshold_type == 'range':
        return threshold['min'] <= value <= threshold['max']
    elif threshold_type == '==':
        return value == threshold['value']
    
    return None


def re_evaluate_items(eval_data: Dict[str, Any], answers: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Re-evaluate REVIEW items based on user-provided answers.
    
    Args:
        eval_data: Current evaluation data
        answers: Dict mapping item_id to list of answer dicts:
                 [{'question_index': 0, 'answer': '12.5', 'skip': False}]
    
    Returns:
        Updated eval_data with new statuses
    """
    for item_id, item_answers in answers.items():
        if item_id not in eval_data:
            continue
        
        item = eval_data[item_id]
        questions = item.get('questions', [])
        
        if not questions or not item_answers:
            continue
        
        # Process each answer
        all_pass = True
        any_skipped = False
        results = []
        
        for ans in item_answers:
            q_index = ans.get('question_index', 0)
            answer_text = ans.get('answer', '')
            skip = ans.get('skip', False)
            
            if skip or not answer_text:
                any_skipped = True
                continue
            
            if q_index >= len(questions):
                continue
            
            question = questions[q_index]
            
            # Parse the answer
            parsed_value = parse_answer(answer_text, question.get('validation', 'numeric'))
            
            if parsed_value is None:
                results.append({
                    'question_index': q_index,
                    'raw_answer': answer_text,
                    'parsed_value': None,
                    'result': 'invalid'
                })
                all_pass = False
                continue
            
            # Evaluate against threshold
            threshold = question.get('threshold')
            if threshold:
                passes = evaluate_against_threshold(parsed_value, threshold)
                results.append({
                    'question_index': q_index,
                    'raw_answer': answer_text,
                    'parsed_value': parsed_value,
                    'result': 'pass' if passes else 'fail'
                })
                if not passes:
                    all_pass = False
            else:
                # No threshold - just record the value, mark as answered
                results.append({
                    'question_index': q_index,
                    'raw_answer': answer_text,
                    'parsed_value': parsed_value,
                    'result': 'answered'
                })
        
        # Update item status based on results
        if any_skipped and not results:
            # All questions skipped
            item['status'] = 'SKIPPED'
            item['comment'] = 'User skipped measurement verification'
        elif any_skipped:
            # Some skipped, some answered
            if all_pass:
                item['status'] = 'PASS'
                item['comment'] = f"Verified by user measurement. Value: {results[0].get('parsed_value', 'N/A')}"
            else:
                item['status'] = 'FAIL'
                item['comment'] = f"Failed threshold check. Value: {results[0].get('parsed_value', 'N/A')}"
        elif results:
            if all_pass:
                item['status'] = 'PASS'
                item['comment'] = f"Verified by user measurement. Value: {results[0].get('parsed_value', 'N/A')}"
            else:
                item['status'] = 'FAIL'
                failed_result = next((r for r in results if r['result'] == 'fail'), results[0])
                item['comment'] = f"Failed threshold check. Measured: {failed_result.get('parsed_value', 'N/A')}"
        
        # Store measurement results
        item['measurement_results'] = results
    
    return eval_data


def count_statuses(eval_data: Dict[str, Any]) -> Dict[str, int]:
    """Count items by status."""
    counts = {'PASS': 0, 'FAIL': 0, 'REVIEW': 0, 'SKIPPED': 0, 'N/A': 0}
    
    for item in eval_data.values():
        status = item.get('status', 'REVIEW')
        if status in counts:
            counts[status] += 1
        else:
            counts['REVIEW'] += 1
    
    return counts
