"""
Follow-up Conversation Module for Redline.AI
Handles follow-up questions after initial review, with smart token management
and optional vision analysis.
"""

import re
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Keywords that indicate vision analysis is needed
VISION_KEYWORDS = [
    'look at', 'check page', 'analyze page', 'see on', 'show', 'visible',
    'drawing', 'detail', 'sheet', 'plan page', 'examine', 'inspect',
    'verify on', 'confirm on', 'page number', 'page #'
]

# Keywords for text-only queries
TEXT_ONLY_KEYWORDS = [
    'explain', 'why', 'clarify', 'what does', 'meaning', 'understand',
    'priority', 'importance', 'order', 'list', 'summarize', 'summary',
    'how many', 'count', 'which items', 'what items'
]

# Query type patterns
QUERY_PATTERNS = {
    'recheck': [
        r're-?check', r'check again', r'look again', r'verify again',
        r'double.?check', r'review again', r'analyze again'
    ],
    'page_analysis': [
        r'page\s*(\d+)', r'sheet\s*(\d+)', r'look at page', r'check page',
        r'analyze page', r'on page'
    ],
    'clarification': [
        r'explain', r'why.*(fail|pass|review)', r'what does.*mean',
        r'clarify', r'understand', r'elaborate'
    ],
    'prioritize': [
        r'priorit', r'import', r'order', r'rank', r'which.*first',
        r'most critical', r'urgent', r'severity'
    ],
    'item_specific': [
        r'\d{2}-[A-Z]+-\d{3}',  # Checklist item ID pattern
        r'item\s+\d+', r'checklist item'
    ]
}


def analyze_query_intent(query: str, eval_data: dict = None) -> Dict[str, Any]:
    """
    Analyze the user's query to determine intent and requirements.
    
    Returns:
        {
            'query_type': str,  # 'recheck', 'page_analysis', 'clarification', 'prioritize', 'general'
            'needs_vision': bool,
            'pages_mentioned': list[int],
            'items_mentioned': list[str],
            'is_specific': bool
        }
    """
    query_lower = query.lower()
    
    result = {
        'query_type': 'general',
        'needs_vision': False,
        'pages_mentioned': [],
        'items_mentioned': [],
        'is_specific': False
    }
    
    # Check for page numbers
    page_matches = re.findall(r'page\s*#?\s*(\d+)', query_lower)
    if page_matches:
        result['pages_mentioned'] = [int(p) for p in page_matches]
    
    # Check for sheet numbers
    sheet_matches = re.findall(r'sheet\s*#?\s*(\d+)', query_lower)
    if sheet_matches:
        result['pages_mentioned'].extend([int(s) for s in sheet_matches])
    
    # Check for checklist item IDs
    item_matches = re.findall(r'(\d{2}-[A-Z]+-\d{3})', query.upper())
    if item_matches:
        result['items_mentioned'] = item_matches
        result['is_specific'] = True
    
    # Determine query type
    for query_type, patterns in QUERY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                result['query_type'] = query_type
                break
        if result['query_type'] != 'general':
            break
    
    # Determine if vision is needed
    result['needs_vision'] = determine_vision_need(query, result)
    
    # Check if specific items are mentioned by name/description
    if eval_data and not result['items_mentioned']:
        for item_id, item_data in eval_data.items():
            # Check if user mentions part of the item text
            if 'ada' in query_lower and 'ADA' in item_id:
                result['items_mentioned'].append(item_id)
            elif 'grading' in query_lower and 'GRD' in item_id:
                result['items_mentioned'].append(item_id)
            elif 'utility' in query_lower and 'UTL' in item_id:
                result['items_mentioned'].append(item_id)
            elif 'traffic' in query_lower and 'TRF' in item_id:
                result['items_mentioned'].append(item_id)
    
    return result


def determine_vision_need(query: str, query_intent: dict) -> bool:
    """
    Determine if GPT-4 Vision is needed for this query.
    
    Uses vision for:
    - Questions about specific pages
    - Questions about visual elements
    - Re-check requests
    
    Text-only for:
    - Explanations/clarifications
    - Prioritization requests
    - Summary requests
    """
    query_lower = query.lower()
    
    # If specific pages are mentioned, likely needs vision
    if query_intent.get('pages_mentioned'):
        return True
    
    # Check for vision keywords
    for keyword in VISION_KEYWORDS:
        if keyword in query_lower:
            return True
    
    # Check for text-only keywords (these override)
    for keyword in TEXT_ONLY_KEYWORDS:
        if keyword in query_lower:
            return False
    
    # Recheck queries may need vision
    if query_intent.get('query_type') == 'recheck':
        return True
    
    # Page analysis definitely needs vision
    if query_intent.get('query_type') == 'page_analysis':
        return True
    
    # Default to text-only to save tokens
    return False


def summarize_turn(content: str, role: str, max_length: int = 150) -> str:
    """
    Create a condensed summary of a conversation turn.
    Used for older turns to save tokens.
    """
    if len(content) <= max_length:
        return content
    
    # For user messages, extract key intent
    if role == 'user':
        # Try to extract the main action/question
        if '?' in content:
            # Get the question part
            parts = content.split('?')
            summary = parts[0][:max_length-3] + '...'
        else:
            summary = content[:max_length-3] + '...'
        return f"[User asked: {summary}]"
    
    # For assistant messages, extract key findings
    else:
        # Look for key phrases
        key_phrases = []
        
        if 'FAIL' in content.upper():
            fail_match = re.search(r'(found|identified|marked).*?(FAIL|failed)', content, re.IGNORECASE)
            if fail_match:
                key_phrases.append("identified failures")
        
        if 'updated' in content.lower():
            key_phrases.append("updated evaluations")
        
        if 'page' in content.lower():
            page_match = re.search(r'page\s*(\d+)', content, re.IGNORECASE)
            if page_match:
                key_phrases.append(f"analyzed page {page_match.group(1)}")
        
        if key_phrases:
            return f"[Assistant: {', '.join(key_phrases)}]"
        else:
            return f"[Assistant provided analysis: {content[:100]}...]"


def build_followup_prompt(
    review_context: dict,
    conversation_history: list,
    user_query: str,
    query_intent: dict,
    recent_turns: int = 2
) -> Tuple[str, str]:
    """
    Build system and user prompts for follow-up conversation.
    
    Manages token usage by:
    - Summarizing older turns
    - Including only relevant eval items
    - Keeping recent turns in full
    
    Returns:
        (system_prompt, user_prompt)
    """
    
    # Build system prompt
    system_prompt = f"""You are an expert civil engineering QA/QC reviewer assistant. You are continuing a review conversation about a planset.

PROJECT CONTEXT:
- Project: {review_context.get('project_name', 'Unknown')}
- Review Type: {review_context.get('review_type', 'Unknown')}
- Total Pages: {review_context.get('page_count', 'Unknown')}

YOUR ROLE:
- Answer follow-up questions about the review
- Provide detailed analysis when asked
- Update evaluations if new information changes your assessment
- Be specific and reference page numbers when relevant

RESPONSE FORMAT:
- Be concise but thorough
- If you update any checklist item evaluations, clearly state:
  EVALUATION UPDATE: [Item ID] changed from [OLD_STATUS] to [NEW_STATUS]
  Reason: [explanation]
- Reference specific page numbers when discussing issues
- If asked to prioritize, list items in order of importance with brief justification"""

    # Build conversation context
    context_parts = []
    
    # Add summarized older turns
    if len(conversation_history) > recent_turns * 2:
        context_parts.append("EARLIER IN CONVERSATION:")
        older_turns = conversation_history[:-recent_turns * 2]
        for turn in older_turns:
            summary = turn.get('content_summary') or summarize_turn(
                turn['content'], 
                turn['role']
            )
            context_parts.append(summary)
        context_parts.append("")
    
    # Add recent turns in full
    if conversation_history:
        context_parts.append("RECENT CONVERSATION:")
        recent = conversation_history[-recent_turns * 2:] if len(conversation_history) > recent_turns * 2 else conversation_history
        for turn in recent:
            role_label = "User" if turn['role'] == 'user' else "Assistant"
            context_parts.append(f"{role_label}: {turn['content']}")
        context_parts.append("")
    
    # Add relevant eval data
    eval_data = review_context.get('eval_data', {})
    if eval_data:
        # Filter to relevant items based on query intent
        relevant_items = {}
        items_mentioned = query_intent.get('items_mentioned', [])
        
        if items_mentioned:
            # User mentioned specific items
            for item_id in items_mentioned:
                if item_id in eval_data:
                    relevant_items[item_id] = eval_data[item_id]
        elif query_intent.get('query_type') == 'prioritize':
            # For prioritization, include all FAIL and REVIEW items
            for item_id, item in eval_data.items():
                if item.get('status') in ['FAIL', 'REVIEW']:
                    relevant_items[item_id] = item
        else:
            # Include items related to mentioned pages
            pages_mentioned = query_intent.get('pages_mentioned', [])
            if pages_mentioned:
                for item_id, item in eval_data.items():
                    page_refs = item.get('page_refs', [])
                    if any(p in pages_mentioned for p in page_refs):
                        relevant_items[item_id] = item
            else:
                # Default: include FAIL items
                for item_id, item in eval_data.items():
                    if item.get('status') == 'FAIL':
                        relevant_items[item_id] = item
        
        if relevant_items:
            context_parts.append("RELEVANT CHECKLIST ITEMS:")
            for item_id, item in list(relevant_items.items())[:15]:  # Limit to 15 items
                status = item.get('status', 'UNKNOWN')
                comment = item.get('comment', '')[:100]
                pages = item.get('page_refs', [])
                context_parts.append(f"- {item_id}: {status} - {comment} (Pages: {pages})")
            context_parts.append("")
    
    # Build user prompt
    user_prompt_parts = []
    
    if context_parts:
        user_prompt_parts.append("\n".join(context_parts))
    
    user_prompt_parts.append(f"USER QUESTION: {user_query}")
    
    # Add specific instructions based on query type
    query_type = query_intent.get('query_type', 'general')
    if query_type == 'recheck':
        user_prompt_parts.append("\nPlease re-examine the relevant items carefully and provide updated assessments if warranted.")
    elif query_type == 'prioritize':
        user_prompt_parts.append("\nList the items in order of importance/urgency for fixing, with the most critical first.")
    elif query_type == 'clarification':
        user_prompt_parts.append("\nProvide a clear explanation addressing the user's question.")
    elif query_type == 'page_analysis':
        pages = query_intent.get('pages_mentioned', [])
        user_prompt_parts.append(f"\nFocus your analysis on page(s) {pages}.")
    
    return system_prompt, "\n".join(user_prompt_parts)


def extract_eval_updates(ai_response: str, current_eval: dict) -> Dict[str, Dict]:
    """
    Parse AI response for any evaluation updates.
    
    Looks for patterns like:
    - "EVALUATION UPDATE: 90-ADA-001 changed from REVIEW to FAIL"
    - "[Item ID] should be marked as FAIL"
    - "Updating [Item ID] to PASS"
    
    Returns dict of updated items with new status/comment.
    """
    updates = {}
    response_upper = ai_response.upper()
    
    # Pattern 1: Explicit EVALUATION UPDATE format
    update_pattern = r'EVALUATION\s*UPDATE:\s*(\d{2}-[A-Z]+-\d{3})\s*(?:changed\s*from\s*\w+\s*to\s*|:?\s*)(\w+)'
    matches = re.findall(update_pattern, ai_response, re.IGNORECASE)
    
    for item_id, new_status in matches:
        item_id = item_id.upper()
        new_status = new_status.upper()
        if new_status in ['PASS', 'FAIL', 'REVIEW', 'N/A']:
            if item_id in current_eval:
                updates[item_id] = {
                    'id': item_id,
                    'status': new_status,
                    'comment': current_eval[item_id].get('comment', '') + ' [Updated via follow-up]',
                    'page_refs': current_eval[item_id].get('page_refs', [])
                }
    
    # Pattern 2: "should be marked as" or "updating to"
    should_pattern = r'(\d{2}-[A-Z]+-\d{3})\s*(?:should\s*be\s*(?:marked\s*as)?|updating\s*(?:to)?|changed?\s*to)\s*(PASS|FAIL|REVIEW|N/?A)'
    matches = re.findall(should_pattern, ai_response, re.IGNORECASE)
    
    for item_id, new_status in matches:
        item_id = item_id.upper()
        new_status = new_status.upper().replace('N/A', 'N/A')
        if item_id not in updates and item_id in current_eval:
            updates[item_id] = {
                'id': item_id,
                'status': new_status,
                'comment': current_eval[item_id].get('comment', '') + ' [Updated via follow-up]',
                'page_refs': current_eval[item_id].get('page_refs', [])
            }
    
    # Extract updated comments if present
    comment_pattern = r'(\d{2}-[A-Z]+-\d{3}).*?(?:comment|reason|because)[:\s]*([^\.]+\.)'
    comment_matches = re.findall(comment_pattern, ai_response, re.IGNORECASE)
    
    for item_id, comment in comment_matches:
        item_id = item_id.upper()
        if item_id in updates:
            updates[item_id]['comment'] = comment.strip()
        elif item_id in current_eval:
            # Update comment only
            updates[item_id] = {
                'id': item_id,
                'status': current_eval[item_id].get('status', 'REVIEW'),
                'comment': comment.strip(),
                'page_refs': current_eval[item_id].get('page_refs', [])
            }
    
    return updates


def classify_query_for_training(query: str, response: str, eval_updates: dict) -> str:
    """
    Classify the query/response pair for training categorization.
    
    Returns one of:
    - 'recheck': Re-examination of items
    - 'page_analysis': Analysis of specific pages
    - 'clarification': Explanation/clarification
    - 'prioritize': Prioritization request
    - 'general': General follow-up
    """
    query_lower = query.lower()
    
    if eval_updates:
        return 'recheck'
    
    if re.search(r'page\s*\d+', query_lower):
        return 'page_analysis'
    
    if any(word in query_lower for word in ['explain', 'why', 'clarify', 'understand']):
        return 'clarification'
    
    if any(word in query_lower for word in ['priority', 'important', 'order', 'rank', 'critical']):
        return 'prioritize'
    
    return 'general'


def generate_followup_response(
    review_context: dict,
    conversation_history: list,
    user_query: str,
    planset_images: list = None
) -> Dict[str, Any]:
    """
    Generate a follow-up response using GPT-4.
    
    Args:
        review_context: Dict with project_name, review_type, eval_data, page_count
        conversation_history: List of previous turns
        user_query: The user's follow-up question
        planset_images: Optional list of base64 images for vision analysis
    
    Returns:
        {
            'response': str,
            'eval_updates': dict,
            'used_vision': bool,
            'query_type': str
        }
    """
    import httpx
    
    try:
        from openai import OpenAI
    except ImportError:
        return {
            'response': 'OpenAI is not available. Please try again later.',
            'eval_updates': {},
            'used_vision': False,
            'query_type': 'error'
        }
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return {
            'response': 'API key not configured. Please contact support.',
            'eval_updates': {},
            'used_vision': False,
            'query_type': 'error'
        }
    
    # Analyze query intent
    query_intent = analyze_query_intent(user_query, review_context.get('eval_data', {}))
    
    # Build prompts
    system_prompt, user_prompt = build_followup_prompt(
        review_context,
        conversation_history,
        user_query,
        query_intent
    )
    
    # Determine if we should use vision
    use_vision = query_intent['needs_vision'] and planset_images
    
    try:
        http_client = httpx.Client()
        client = OpenAI(api_key=api_key, http_client=http_client)
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if use_vision and planset_images:
            # Build message with images
            content = [{"type": "text", "text": user_prompt}]
            
            # Add relevant page images (limit to 3 for token management)
            pages_to_show = query_intent.get('pages_mentioned', [])[:3]
            if not pages_to_show:
                pages_to_show = [0]  # Default to first page
            
            for idx, page_num in enumerate(pages_to_show):
                if page_num < len(planset_images):
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{planset_images[page_num]}",
                            "detail": "high"
                        }
                    })
            
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_prompt})
        
        # Make API call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2000,
            temperature=0.3
        )
        
        ai_response = response.choices[0].message.content
        
        # Extract any evaluation updates
        eval_updates = extract_eval_updates(
            ai_response, 
            review_context.get('eval_data', {})
        )
        
        # Classify for training
        query_type = classify_query_for_training(user_query, ai_response, eval_updates)
        
        return {
            'response': ai_response,
            'eval_updates': eval_updates,
            'used_vision': use_vision,
            'query_type': query_type
        }
        
    except Exception as e:
        logger.error(f"Error generating follow-up response: {e}")
        return {
            'response': f'Sorry, I encountered an error processing your request. Please try again.',
            'eval_updates': {},
            'used_vision': False,
            'query_type': 'error'
        }
