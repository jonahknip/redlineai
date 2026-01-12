#!/usr/bin/env python3
"""
Training Data Extraction Script for Redline.ai
Parses filled-out QA/QC checklist PDFs and extracts training examples.
"""

import fitz
import json
import re
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load checklist definitions
CHECKLISTS_DIR = Path(__file__).parent.parent / 'checklists'
TRAINING_DIR = Path(__file__).parent.parent / 'training' / 'examples'

def load_checklist(phase: str) -> dict:
    """Load a checklist definition by phase (30, 60, 90)"""
    filename = f"{phase}_percent.json"
    filepath = CHECKLISTS_DIR / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def detect_checklist_phase(pdf_path: str, fields: dict, text: str) -> str:
    """Detect which phase checklist this is (30%, 60%, 90%)"""
    filename_lower = pdf_path.lower()
    text_lower = text.lower()
    
    if '90%' in filename_lower or '90 checklist' in filename_lower or '90% engineering' in text_lower:
        return '90'
    elif '60%' in filename_lower or '60 checklist' in filename_lower or '60% engineering' in text_lower:
        return '60'
    elif '30%' in filename_lower or '30 checklist' in filename_lower or '30% engineering' in text_lower:
        return '30'
    
    # Default to 30% if can't determine
    return '30'

def extract_project_info(fields: dict) -> dict:
    """Extract project metadata from form fields"""
    info = {
        'project_number': '',
        'project_name': '',
        'project_manager': '',
        'reviewer': '',
        'review_date': ''
    }
    
    # Common field name patterns
    for field_name, field_value in fields.items():
        name_lower = field_name.lower()
        value = str(field_value or '').strip()
        
        if 'project no' in name_lower or 'project_no' in name_lower or field_name == 'Project No':
            info['project_number'] = value
        elif 'project name' in name_lower or field_name == 'Project Name':
            info['project_name'] = value
        elif 'manager' in name_lower or field_name == 'Project Manager':
            info['project_manager'] = value
        elif 'reviewer' in name_lower or field_name == 'Reviewer':
            info['reviewer'] = value
        elif 'date' in name_lower and 'update' not in name_lower:
            info['review_date'] = value
    
    return info

def parse_checkbox_groups(fields: dict) -> list:
    """
    Parse checkbox fields into groups of 3 (YES, NO, N/A).
    Returns list of dicts with status and associated comment field number.
    """
    # Group checkboxes by their number pattern
    checkbox_pattern = re.compile(r'Check Box(\d+)')
    text_pattern = re.compile(r'Text(\d+)')
    
    checkboxes = {}
    texts = {}
    
    for field_name, field_value in fields.items():
        cb_match = checkbox_pattern.match(field_name)
        if cb_match:
            num = int(cb_match.group(1))
            checkboxes[num] = field_value
        
        txt_match = text_pattern.match(field_name)
        if txt_match:
            num = int(txt_match.group(1))
            texts[num] = field_value
    
    # Checkboxes are typically in groups of 3 (YES, NO, N/A)
    # Starting from Check Box1, Check Box2, Check Box3 for item 1, etc.
    results = []
    
    # Sort checkbox numbers
    sorted_nums = sorted(checkboxes.keys())
    
    if not sorted_nums:
        return results
    
    # Group into sets of 3
    i = 0
    item_index = 0
    while i < len(sorted_nums):
        # Get the next 3 checkboxes
        group_nums = []
        base_num = sorted_nums[i]
        
        # Find consecutive or near-consecutive numbers
        for j in range(3):
            if i + j < len(sorted_nums):
                num = sorted_nums[i + j]
                # Allow some gap but they should be close
                if j == 0 or num - group_nums[-1] <= 2:
                    group_nums.append(num)
        
        if len(group_nums) >= 3:
            yes_val = checkboxes.get(group_nums[0], '')
            no_val = checkboxes.get(group_nums[1], '')
            na_val = checkboxes.get(group_nums[2], '')
            
            # Determine status
            status = 'REVIEW'  # Default if none checked
            if yes_val and yes_val not in ('', 'Off'):
                status = 'PASS'
            elif no_val and no_val not in ('', 'Off'):
                status = 'FAIL'
            elif na_val and na_val not in ('', 'Off'):
                status = 'N/A'
            
            # Try to find associated comment field
            # Comments are usually numbered differently, try common patterns
            comment = ''
            possible_text_nums = [
                group_nums[2] + 1,  # Right after the NA checkbox
                45 + item_index,     # Starting from Text45
                item_index + 1,      # Simple index
            ]
            
            for txt_num in possible_text_nums:
                if txt_num in texts and texts[txt_num]:
                    comment = str(texts[txt_num]).strip()
                    break
            
            results.append({
                'item_index': item_index,
                'checkbox_nums': group_nums,
                'status': status,
                'comment': comment
            })
            
            item_index += 1
            i += 3
        else:
            i += 1
    
    return results

def extract_checklist_items_from_text(doc) -> list:
    """Extract checklist item text from the PDF by analyzing the layout"""
    items = []
    
    for page in doc:
        # Get text with positions
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" not in block:
                continue
            
            for line in block["lines"]:
                text = " ".join([span["text"] for span in line["spans"]]).strip()
                
                # Skip headers and empty lines
                if not text or len(text) < 10:
                    continue
                
                # Skip known headers
                skip_patterns = [
                    'GENERAL REVIEW', 'ALL SHEETS', 'UTILITIES', 'CROSS SECTION',
                    'ADA', 'TRAFFIC', 'EROSION', 'CALCULATIONS', 'COST',
                    'PERMITS', 'SPECIFICATIONS', 'YES', 'NO', 'N/A', 'COMMENTS',
                    'Project No', 'Project Name', 'Project Manager', 'Reviewer', 'Date',
                    '30 %', '60 %', '90 %', 'ENGINEERING QA/QC'
                ]
                
                if any(skip.lower() in text.lower() for skip in skip_patterns):
                    continue
                
                # This might be a checklist item
                if '?' in text or text[0].isupper():
                    items.append({
                        'text': text,
                        'y_pos': line["bbox"][1],
                        'page': page.number
                    })
    
    # Sort by position (top to bottom, page by page)
    items.sort(key=lambda x: (x['page'], x['y_pos']))
    
    return items

def match_items_to_checklist(extracted_items: list, checkbox_results: list, checklist: dict) -> list:
    """Match extracted PDF items to checklist definitions"""
    
    if not checklist:
        return []
    
    # Flatten checklist items
    checklist_items = []
    for section in checklist.get('sections', []):
        section_title = section.get('title', '')
        for item in section.get('items', []):
            checklist_items.append({
                'id': item.get('id', ''),
                'text': item.get('text', ''),
                'section': section_title,
                'required': item.get('required', False)
            })
    
    matched = []
    
    # Match by index order (checkboxes should align with checklist items)
    for i, cb_result in enumerate(checkbox_results):
        if i < len(checklist_items):
            item = checklist_items[i]
            matched.append({
                'checklist_item_id': item['id'],
                'checklist_item_text': item['text'],
                'section': item['section'],
                'status': cb_result['status'],
                'comment': cb_result['comment'],
                'required': item['required']
            })
    
    return matched

def parse_pdf(pdf_path: str) -> list:
    """Parse a single PDF and extract training examples"""
    doc = fitz.open(pdf_path)
    
    # Extract form fields
    fields = {}
    for page in doc:
        for widget in page.widgets():
            if widget.field_name:
                fields[widget.field_name] = widget.field_value
    
    # Get full text for analysis
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    # Detect phase
    phase = detect_checklist_phase(pdf_path, fields, full_text)
    checklist = load_checklist(phase)
    
    # Extract project info
    project_info = extract_project_info(fields)
    
    # Parse checkboxes
    checkbox_results = parse_checkbox_groups(fields)
    
    # Match to checklist items
    matched_items = match_items_to_checklist([], checkbox_results, checklist)
    
    # Build training examples
    examples = []
    source_file = Path(pdf_path).name
    
    for item in matched_items:
        example = {
            'checklist_item_id': item['checklist_item_id'],
            'checklist_item_text': item['checklist_item_text'],
            'section': item['section'],
            'status': item['status'],
            'comment': item['comment'],
            'project_type': 'road',  # Default, could be detected
            'project_name': project_info['project_name'],
            'project_number': project_info['project_number'],
            'reviewer': project_info['reviewer'],
            'review_date': project_info['review_date'],
            'source_file': source_file,
            'phase': f"{phase}_percent"
        }
        examples.append(example)
    
    doc.close()
    return examples

def load_existing_training_data() -> list:
    """Load existing training examples"""
    training_file = TRAINING_DIR / 'training_examples.json'
    if training_file.exists():
        with open(training_file, 'r') as f:
            return json.load(f)
    return []

def save_training_data(examples: list):
    """Save training examples to JSON file"""
    training_file = TRAINING_DIR / 'training_examples.json'
    with open(training_file, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {len(examples)} training examples to {training_file}")

def deduplicate_examples(examples: list) -> list:
    """Remove duplicate examples based on key fields"""
    seen = set()
    unique = []
    
    for ex in examples:
        # Create a key from unique identifying fields
        key = (
            ex.get('checklist_item_id', ''),
            ex.get('status', ''),
            ex.get('project_name', ''),
            ex.get('source_file', '')
        )
        
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    
    return unique

def process_directory(directory: str, merge_existing: bool = True) -> list:
    """Process all PDFs in a directory"""
    all_examples = []
    
    if merge_existing:
        all_examples = load_existing_training_data()
        print(f"Loaded {len(all_examples)} existing training examples")
    
    pdf_dir = Path(directory)
    pdf_files = list(pdf_dir.glob('*.pdf'))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        try:
            examples = parse_pdf(str(pdf_path))
            print(f"  Extracted {len(examples)} examples")
            
            # Count by status
            status_counts = defaultdict(int)
            for ex in examples:
                status_counts[ex['status']] += 1
            print(f"  Status breakdown: {dict(status_counts)}")
            
            all_examples.extend(examples)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Deduplicate
    original_count = len(all_examples)
    all_examples = deduplicate_examples(all_examples)
    print(f"\nDeduplicated: {original_count} -> {len(all_examples)} examples")
    
    return all_examples

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract training data from QA/QC checklist PDFs')
    parser.add_argument('directory', help='Directory containing PDF files')
    parser.add_argument('--no-merge', action='store_true', help='Do not merge with existing training data')
    parser.add_argument('--dry-run', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    examples = process_directory(args.directory, merge_existing=not args.no_merge)
    
    if not args.dry_run:
        save_training_data(examples)
    else:
        print(f"\nDry run - would save {len(examples)} examples")
    
    # Print statistics
    print("\n=== Training Data Statistics ===")
    status_counts = defaultdict(int)
    phase_counts = defaultdict(int)
    project_counts = defaultdict(int)
    
    for ex in examples:
        status_counts[ex.get('status', 'UNKNOWN')] += 1
        phase_counts[ex.get('phase', 'unknown')] += 1
        project_counts[ex.get('project_name', 'Unknown')[:30]] += 1
    
    print(f"Total examples: {len(examples)}")
    print(f"\nBy Status:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    print(f"\nBy Phase:")
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count}")
    
    print(f"\nBy Project:")
    for project, count in sorted(project_counts.items(), key=lambda x: -x[1]):
        print(f"  {project}: {count}")

if __name__ == '__main__':
    main()
