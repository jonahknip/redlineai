"""
Phase-specific prompt management for Plan Review Agent.

This module loads and manages prompt templates for different review phases (30%, 60%, 90%).
Prompts are stored as JSON files in the prompts/ directory for easy editing without code changes.
"""

import json
import os
from typing import Dict, List, Optional, Any

# Path to prompts directory (relative to this file)
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'prompts')

# Cache for loaded prompts
_prompt_cache: Dict[str, Dict] = {}


def load_phase_prompt(phase: str) -> Optional[Dict]:
    """
    Load prompt configuration for a specific phase.
    
    Args:
        phase: The review phase ('30%', '60%', '90%')
        
    Returns:
        Dict with prompt configuration or None if not found
    """
    # Normalize phase string
    phase_key = phase.replace('%', '').strip()
    
    # Check cache first
    if phase_key in _prompt_cache:
        return _prompt_cache[phase_key]
    
    # Build file path
    filename = f"{phase_key}_percent_prompt.json"
    filepath = os.path.join(PROMPTS_DIR, filename)
    
    try:
        with open(filepath, 'r') as f:
            prompt_data = json.load(f)
            _prompt_cache[phase_key] = prompt_data
            return prompt_data
    except FileNotFoundError:
        print(f"[PROMPTS] Warning: Prompt file not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"[PROMPTS] Error parsing prompt file {filepath}: {e}")
        return None


def get_phase_context(phase: str) -> str:
    """
    Get the phase context string for the evaluation prompt.
    
    Args:
        phase: The review phase ('30%', '60%', '90%')
        
    Returns:
        Phase context string or empty string if not found
    """
    prompt_data = load_phase_prompt(phase)
    if not prompt_data:
        return ""
    return prompt_data.get('phase_context', '')


def get_section_guidance(phase: str, section_title: str) -> str:
    """
    Get guidance for a specific section within a phase.
    
    Args:
        phase: The review phase ('30%', '60%', '90%')
        section_title: The section title (e.g., 'UTILITIES & STRUCTURES')
        
    Returns:
        Section guidance string or empty string if not found
    """
    prompt_data = load_phase_prompt(phase)
    if not prompt_data:
        return ""
    
    section_guidance = prompt_data.get('section_guidance', {})
    return section_guidance.get(section_title, '')


def get_example_evaluations(phase: str, limit: int = 3) -> List[Dict]:
    """
    Get example evaluations for few-shot prompting.
    
    Args:
        phase: The review phase ('30%', '60%', '90%')
        limit: Maximum number of examples to return
        
    Returns:
        List of example evaluation dicts
    """
    prompt_data = load_phase_prompt(phase)
    if not prompt_data:
        return []
    
    examples = prompt_data.get('example_evaluations', [])
    return examples[:limit]


def get_pass_fail_criteria(phase: str) -> Dict[str, str]:
    """
    Get the pass/fail/review criteria for a phase.
    
    Args:
        phase: The review phase ('30%', '60%', '90%')
        
    Returns:
        Dict with 'pass', 'fail', and 'review' criteria strings
    """
    prompt_data = load_phase_prompt(phase)
    if not prompt_data:
        return {
            'pass': 'Item is clearly verified/met in the plans',
            'fail': 'Item is missing, incomplete, or has issues',
            'review': 'Cannot determine from available information'
        }
    
    return {
        'pass': prompt_data.get('pass_criteria', 'Item is clearly verified/met in the plans'),
        'fail': prompt_data.get('fail_criteria', 'Item is missing, incomplete, or has issues'),
        'review': prompt_data.get('review_criteria', 'Cannot determine from available information')
    }


def format_example_evaluations(examples: List[Dict]) -> str:
    """
    Format example evaluations for inclusion in prompt.
    
    Args:
        examples: List of example evaluation dicts
        
    Returns:
        Formatted string with examples
    """
    if not examples:
        return ""
    
    lines = ["EXAMPLE EVALUATIONS (for reference):"]
    for ex in examples:
        lines.append(f"- {ex.get('item_id', 'N/A')}: {ex.get('item_text', '')}")
        lines.append(f"  Status: {ex.get('status', 'REVIEW')}")
        lines.append(f"  Comment: {ex.get('comment', '')}")
        lines.append("")
    
    return "\n".join(lines)


def build_evaluation_prompt(
    phase: str,
    checklist_items: List[Dict],
    vision_results: str,
    extracted_text: str,
    training_examples: str = ""
) -> str:
    """
    Build the complete evaluation prompt with phase-specific context.
    
    Args:
        phase: The review phase ('30%', '60%', '90%')
        checklist_items: List of checklist item dicts with 'id' and 'text'
        vision_results: AI vision analysis results
        extracted_text: Extracted text from PDF
        training_examples: Additional few-shot examples from training data
        
    Returns:
        Complete evaluation prompt string
    """
    # Load phase-specific prompt data
    prompt_data = load_phase_prompt(phase)
    
    # Get phase context
    phase_context = ""
    if prompt_data:
        phase_context = f"""
REVIEW PHASE: {phase}
{prompt_data.get('phase_context', '')}

WHAT SHOULD EXIST AT THIS PHASE:
{chr(10).join('- ' + item for item in prompt_data.get('what_should_exist', []))}

WHAT SHOULD NOT EXIST YET:
{chr(10).join('- ' + item for item in prompt_data.get('what_should_not_exist_yet', []))}
"""
    
    # Get example evaluations
    examples = get_example_evaluations(phase, limit=3)
    examples_text = format_example_evaluations(examples)
    
    # Get pass/fail criteria
    criteria = get_pass_fail_criteria(phase)
    
    # Build checklist items text with section guidance
    items_text_parts = []
    current_section = None
    
    for item in checklist_items:
        # Check if this is a new section (infer from item ID pattern)
        item_id = item.get('id', '')
        
        items_text_parts.append(f"- {item_id}: {item.get('text', '')}")
    
    items_text = "\n".join(items_text_parts)
    
    # Build the complete prompt
    eval_prompt = f"""Evaluate each checklist item based on the plan analysis.

{phase_context}

PLAN ANALYSIS:
{vision_results if vision_results else 'No vision analysis available'}

EXTRACTED TEXT:
{extracted_text}

{training_examples}

{examples_text}

CHECKLIST ITEMS TO EVALUATE:
{items_text}

Return a JSON array with your evaluation of each item. Format:
[
  {{"id": "30-GEN-001", "status": "PASS", "comment": "Verified - north arrow present on all sheets"}},
  {{"id": "30-GEN-002", "status": "FAIL", "comment": "ADA ramps not identified in scope"}},
  {{"id": "30-GEN-003", "status": "REVIEW", "comment": "Cannot verify utility locations from available sheets"}},
  {{"id": "30-GEN-004", "status": "N/A", "comment": "Not applicable to this project type"}}
]

Status options: PASS, FAIL, REVIEW, N/A
- PASS = {criteria['pass']}
- FAIL = {criteria['fail']}
- REVIEW = {criteria['review']}
- N/A = Does not apply to this project type

IMPORTANT GUIDELINES:
1. Be DECISIVE. If you can see evidence in the plans, mark as PASS or FAIL with specific comments.
2. Only mark REVIEW when you truly cannot determine the answer from the analysis provided.
3. Provide SPECIFIC, ACTIONABLE comments referencing what you observed (sheet numbers, stations, etc.)
4. For FAIL items, explain exactly what is wrong or missing and what needs to be corrected.
5. For PASS items, cite the specific evidence you found (e.g., "North arrow on sheets C-1 through C-5")

Return ONLY the JSON array, no additional text."""

    return eval_prompt


def get_system_prompt(phase: str) -> str:
    """
    Get the system prompt for the AI evaluation.
    
    Args:
        phase: The review phase ('30%', '60%', '90%')
        
    Returns:
        System prompt string
    """
    prompt_data = load_phase_prompt(phase)
    phase_context = prompt_data.get('phase_context', '') if prompt_data else ''
    
    return f"""You are an expert Civil Engineering QA/QC reviewer performing a {phase} design review.

{phase_context}

Your role is to:
1. Carefully evaluate each checklist item against the plan analysis
2. Provide specific, actionable feedback
3. Be decisive - avoid excessive use of REVIEW status
4. Reference specific sheet numbers, stations, and details when possible
5. Identify issues that must be corrected before advancing to the next phase

Return evaluations as a JSON array only."""


def clear_cache():
    """Clear the prompt cache (useful for development/testing)."""
    global _prompt_cache
    _prompt_cache = {}
