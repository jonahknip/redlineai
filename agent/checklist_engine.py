"""
Checklist Engine - Orchestrates plan review against QA/QC checklists
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .plan_reviewer import PlanReviewer


class ChecklistEngine:
    """
    Manages checklist evaluation against plan sheets.
    Coordinates PlanReviewer to analyze each checklist item.
    """
    
    CHECKLIST_DIR = Path(__file__).parent.parent / 'checklists'
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the checklist engine."""
        self.api_key = api_key
        self._plan_reviewer = None  # Lazy initialization
        self.checklists = self._load_checklists()
    
    @property
    def plan_reviewer(self):
        """Lazy load the plan reviewer only when needed."""
        if self._plan_reviewer is None:
            self._plan_reviewer = PlanReviewer(api_key=self.api_key)
        return self._plan_reviewer
        
    def _load_checklists(self) -> Dict[str, Dict]:
        """Load all available checklists from JSON files."""
        checklists = {}
        checklist_files = {
            '30%': '30_percent.json',
            '60%': '60_percent.json',
            '90%': '90_percent.json',
            'CADD': 'cadd_review.json'
        }
        
        for phase, filename in checklist_files.items():
            filepath = self.CHECKLIST_DIR / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    checklists[phase] = json.load(f)
        
        return checklists
    
    def get_available_checklists(self) -> List[Dict[str, str]]:
        """Get list of available checklist phases."""
        return [
            {
                'phase': phase,
                'name': checklist.get('name', phase),
                'description': checklist.get('description', ''),
                'item_count': sum(len(section['items']) for section in checklist.get('sections', []))
            }
            for phase, checklist in self.checklists.items()
        ]
    
    def get_checklist(self, phase: str) -> Optional[Dict]:
        """Get a specific checklist by phase."""
        return self.checklists.get(phase)
    
    def _determine_relevant_sheets(
        self, 
        sheets: List[Dict[str, Any]], 
        checklist_item: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Determine which sheets are relevant for a checklist item.
        Uses sheet classification to match items to appropriate sheets.
        """
        item_text = checklist_item['text'].lower()
        item_id = checklist_item['id']
        
        # Keywords mapping to sheet types
        sheet_type_keywords = {
            'title': ['title', 'index', 'sheet number', 'project name', 'location map', 'vicinity'],
            'site_plan': ['site plan', 'layout', 'property', 'boundary'],
            'grading': ['grading', 'contour', 'elevation', 'spot grade', 'slope'],
            'demolition': ['removal', 'demolition', 'existing'],
            'utility': ['utility', 'water', 'sewer', 'sanitary', 'storm', 'manhole', 'pipe'],
            'stormwater': ['stormwater', 'drainage', 'detention', 'retention', 'inlet'],
            'erosion_control': ['erosion', 'sediment', 'sesc', 'soil erosion'],
            'mot': ['traffic', 'mot', 'maintenance', 'sign', 'barricade', 'detour'],
            'cross_section': ['cross section', 'typical section', 'pavement'],
            'profile': ['profile', 'vertical', 'station'],
            'detail': ['detail', 'standard'],
            'landscape': ['landscape', 'planting', 'tree'],
            'structural': ['structural', 'wall', 'retaining']
        }
        
        # Determine what sheet types this item might apply to
        relevant_types = set()
        
        for sheet_type, keywords in sheet_type_keywords.items():
            if any(kw in item_text for kw in keywords):
                relevant_types.add(sheet_type)
        
        # Some items apply to all sheets
        all_sheet_keywords = ['all sheet', 'each sheet', 'every sheet', 'spell check', 
                             'north arrow', 'scale', 'title block', 'text', 'layer']
        if any(kw in item_text for kw in all_sheet_keywords):
            return sheets  # Return all sheets
        
        # Filter sheets by relevant types
        if relevant_types:
            relevant_sheets = []
            for sheet in sheets:
                sheet_type = sheet.get('classification', {}).get('sheet_type', 'unknown')
                if sheet_type in relevant_types or sheet_type == 'unknown':
                    relevant_sheets.append(sheet)
            return relevant_sheets if relevant_sheets else sheets[:3]  # Fallback to first 3
        
        # Default: check first sheet (title) and a few representative sheets
        return sheets[:min(3, len(sheets))]
    
    def run_review(
        self,
        pdf_paths: List[str],
        phase: str,
        custom_instructions: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run a complete plan review against a checklist.
        
        Args:
            pdf_paths: List of PDF file paths to review
            phase: Checklist phase ('30%', '60%', '90%', 'CADD')
            custom_instructions: Optional additional review instructions
            progress_callback: Optional callback for progress updates (current, total, message)
            
        Returns:
            Complete review results with checklist item evaluations
        """
        checklist = self.get_checklist(phase)
        if not checklist:
            raise ValueError(f"Unknown checklist phase: {phase}")
        
        # Extract all sheets from all PDFs
        if progress_callback:
            progress_callback(0, 100, "Extracting sheets from PDFs...")
        
        sheets = self.plan_reviewer.extract_sheets_from_multiple_pdfs(pdf_paths)
        
        # Classify each sheet
        if progress_callback:
            progress_callback(5, 100, "Classifying sheets...")
        
        for i, sheet in enumerate(sheets):
            sheet = self.plan_reviewer.classify_sheet(sheet)
            sheets[i] = sheet
        
        # Get project summary
        if progress_callback:
            progress_callback(10, 100, "Extracting project summary...")
        
        project_summary = self.plan_reviewer.get_overall_plan_summary(sheets)
        
        # Calculate total items for progress
        total_items = sum(len(section['items']) for section in checklist.get('sections', []))
        current_item = 0
        
        # Process each checklist section and item
        results = {
            'review_date': datetime.now().isoformat(),
            'phase': phase,
            'checklist_name': checklist.get('name', phase),
            'next_phase': checklist.get('next_phase'),
            'project_summary': project_summary,
            'pdf_files': [os.path.basename(p) for p in pdf_paths],
            'total_sheets': len(sheets),
            'custom_instructions': custom_instructions,
            'sections': [],
            'summary': {
                'total_items': total_items,
                'yes_count': 0,
                'no_count': 0,
                'na_count': 0,
                'items_for_next_phase': []
            }
        }
        
        for section in checklist.get('sections', []):
            section_results = {
                'title': section['title'],
                'items': []
            }
            
            for item in section['items']:
                current_item += 1
                progress_pct = 10 + int((current_item / total_items) * 85)
                
                if progress_callback:
                    progress_callback(
                        progress_pct, 100, 
                        f"Reviewing: {item['text'][:50]}..."
                    )
                
                # Find relevant sheets for this item
                relevant_sheets = self._determine_relevant_sheets(sheets, item)
                
                # Analyze item against relevant sheets
                item_results = []
                for sheet in relevant_sheets[:3]:  # Limit to 3 sheets per item for efficiency
                    analysis = self.plan_reviewer.analyze_sheet_for_checklist_item(
                        sheet, item, custom_instructions
                    )
                    item_results.append(analysis)
                
                # Consolidate results across sheets
                consolidated = self._consolidate_item_results(item, item_results)
                section_results['items'].append(consolidated)
                
                # Update summary counts
                status = consolidated['status']
                if status == 'YES':
                    results['summary']['yes_count'] += 1
                elif status == 'NO':
                    results['summary']['no_count'] += 1
                    if item.get('required', True):
                        results['summary']['items_for_next_phase'].append({
                            'id': item['id'],
                            'text': item['text'],
                            'comments': consolidated['comments']
                        })
                else:
                    results['summary']['na_count'] += 1
            
            results['sections'].append(section_results)
        
        # Extract quantities from relevant sheets
        if progress_callback:
            progress_callback(95, 100, "Extracting quantities...")
        
        quantities = []
        for sheet in sheets:
            sheet_type = sheet.get('classification', {}).get('sheet_type', '')
            if sheet_type in ['site_plan', 'grading', 'demolition', 'utility', 'stormwater']:
                qty_data = self.plan_reviewer.extract_quantities(sheet)
                if qty_data.get('quantities_found'):
                    quantities.append({
                        'sheet': f"{sheet['pdf_name']} - Page {sheet['page_num']}",
                        'quantities': qty_data['quantities_found']
                    })
        
        results['quantities'] = quantities
        
        if progress_callback:
            progress_callback(100, 100, "Review complete!")
        
        return results
    
    def _consolidate_item_results(
        self, 
        item: Dict[str, str], 
        sheet_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Consolidate analysis results from multiple sheets into a single item result.
        """
        if not sheet_results:
            return {
                'id': item['id'],
                'text': item['text'],
                'required': item.get('required', True),
                'status': 'N/A',
                'comments': 'No applicable sheets found for this item',
                'confidence': 0.0,
                'sheet_analyses': []
            }
        
        # Collect all statuses and find consensus
        statuses = []
        all_comments = []
        all_issues = []
        confidences = []
        applicable_results = []
        
        for result in sheet_results:
            analysis = result.get('analysis', {})
            if analysis.get('applies_to_sheet', True):
                applicable_results.append(result)
                statuses.append(analysis.get('status', 'N/A'))
                if analysis.get('comments'):
                    all_comments.append(f"[{result['sheet_pdf']} p{result['sheet_page']}]: {analysis['comments']}")
                if analysis.get('issues_found'):
                    all_issues.extend(analysis['issues_found'])
                confidences.append(analysis.get('confidence', 0.5))
        
        if not applicable_results:
            return {
                'id': item['id'],
                'text': item['text'],
                'required': item.get('required', True),
                'status': 'N/A',
                'comments': 'Item does not apply to any sheets in this plan set',
                'confidence': 0.8,
                'sheet_analyses': sheet_results
            }
        
        # Determine final status (if any NO, overall is NO; otherwise majority wins)
        if 'NO' in statuses:
            final_status = 'NO'
        elif statuses.count('YES') >= statuses.count('N/A'):
            final_status = 'YES'
        else:
            final_status = 'N/A'
        
        # Combine comments
        combined_comments = ' | '.join(all_comments) if all_comments else 'No specific comments'
        
        return {
            'id': item['id'],
            'text': item['text'],
            'required': item.get('required', True),
            'status': final_status,
            'comments': combined_comments,
            'issues': all_issues,
            'confidence': sum(confidences) / len(confidences) if confidences else 0.0,
            'sheets_analyzed': len(applicable_results),
            'sheet_analyses': sheet_results
        }
    
    def get_items_required_for_next_phase(self, review_results: Dict[str, Any]) -> List[Dict]:
        """
        Extract list of items that must be addressed before advancing to next phase.
        """
        return review_results.get('summary', {}).get('items_for_next_phase', [])
