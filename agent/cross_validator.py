"""
Cross-Sheet Validator - Validates consistency across drawing sheets
Checks for reference validity, scale consistency, annotation consistency
"""
import re
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule: str
    passed: bool
    severity: str = "info"  # critical, major, minor, info
    title: str = ""
    description: str = ""
    affected_sheets: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class CrossSheetValidator:
    """
    Validates consistency across all sheets in a planset.
    
    Checks for:
    - Sheet reference validity (references point to existing sheets)
    - Scale consistency (compatible scales used)
    - Title block consistency (project info matches across sheets)
    - North arrow presence on plan sheets
    - Legend completeness (all used symbols are defined)
    """
    
    def __init__(self, analysis_data: Dict[str, Any]):
        """
        Initialize with analysis data from FullDrawingAnalyzer.
        
        Args:
            analysis_data: Output from analyzer.to_json()
        """
        self.data = analysis_data
        self.sheets = analysis_data.get('sheets', [])
        self.project_info = analysis_data.get('project_info', {})
        self.all_elements = analysis_data.get('all_elements', {})
        
        # Build sheet index for lookups
        self.sheet_numbers = {s.get('sheet_number', ''): s for s in self.sheets}
        
        self.validation_results: List[ValidationResult] = []
    
    def validate_all(self) -> List[ValidationResult]:
        """
        Run all validation checks.
        
        Returns:
            List of ValidationResult objects
        """
        self.validation_results = []
        
        # Run all validators
        self.validate_sheet_references()
        self.validate_scale_consistency()
        self.validate_title_block_consistency()
        self.validate_north_arrow_presence()
        self.validate_dimension_formats()
        self.validate_annotation_consistency()
        
        return self.validation_results
    
    def validate_sheet_references(self) -> None:
        """Check that all sheet references point to valid sheets."""
        invalid_refs = []
        
        for sheet in self.sheets:
            sheet_num = sheet.get('sheet_number', 'Unknown')
            references = sheet.get('references', [])
            
            for ref in references:
                if ref and ref not in self.sheet_numbers:
                    invalid_refs.append({
                        'source': sheet_num,
                        'target': ref,
                        'page': sheet.get('page_number')
                    })
        
        if invalid_refs:
            self.validation_results.append(ValidationResult(
                rule='sheet_references',
                passed=False,
                severity='major',
                title='Invalid Sheet References Found',
                description=f"Found {len(invalid_refs)} references to sheets that don't exist in this planset.",
                affected_sheets=[r['source'] for r in invalid_refs],
                details={'invalid_references': invalid_refs}
            ))
        else:
            self.validation_results.append(ValidationResult(
                rule='sheet_references',
                passed=True,
                severity='info',
                title='Sheet References Valid',
                description='All sheet references point to existing sheets.'
            ))
    
    def validate_scale_consistency(self) -> None:
        """Check for compatible scales across related sheets."""
        scales_by_type: Dict[str, Set[str]] = {}
        
        for sheet in self.sheets:
            sheet_type = sheet.get('sheet_type', 'unknown')
            scale = sheet.get('scale', '').strip().upper()
            
            if scale and scale not in ['', 'VARIES', 'NTS', 'AS NOTED', 'AS SHOWN']:
                if sheet_type not in scales_by_type:
                    scales_by_type[sheet_type] = set()
                scales_by_type[sheet_type].add(scale)
        
        # Check for inconsistent scales within same sheet type
        inconsistent = []
        for sheet_type, scales in scales_by_type.items():
            if len(scales) > 1:
                inconsistent.append({
                    'type': sheet_type,
                    'scales': list(scales)
                })
        
        if inconsistent:
            self.validation_results.append(ValidationResult(
                rule='scale_consistency',
                passed=False,
                severity='minor',
                title='Inconsistent Scales Found',
                description=f"Found sheets of the same type with different scales.",
                details={'inconsistencies': inconsistent}
            ))
        else:
            self.validation_results.append(ValidationResult(
                rule='scale_consistency',
                passed=True,
                severity='info',
                title='Scales Consistent',
                description='Scales are consistent across sheet types.'
            ))
    
    def validate_title_block_consistency(self) -> None:
        """Check that title block info is consistent across all sheets."""
        # Collect title block data from all sheets
        project_names = set()
        project_numbers = set()
        engineers = set()
        
        for sheet in self.sheets:
            tb = sheet.get('title_block', {})
            if tb:
                pn = tb.get('project_name', '').strip()
                if pn:
                    project_names.add(pn)
                
                pnum = tb.get('project_number', '').strip()
                if pnum:
                    project_numbers.add(pnum)
                
                eng = tb.get('engineer', '').strip()
                if eng:
                    engineers.add(eng)
        
        issues = []
        
        if len(project_names) > 1:
            issues.append({
                'field': 'project_name',
                'values': list(project_names),
                'message': 'Multiple project names found'
            })
        
        if len(project_numbers) > 1:
            issues.append({
                'field': 'project_number',
                'values': list(project_numbers),
                'message': 'Multiple project numbers found'
            })
        
        if len(engineers) > 1:
            # This is usually okay (multiple firms), so just note it
            issues.append({
                'field': 'engineer',
                'values': list(engineers),
                'message': 'Multiple engineers/firms noted (may be intentional)'
            })
        
        if any(i['field'] != 'engineer' for i in issues):
            self.validation_results.append(ValidationResult(
                rule='title_block_consistency',
                passed=False,
                severity='major',
                title='Title Block Inconsistency',
                description='Project information varies across sheets.',
                details={'issues': issues}
            ))
        else:
            self.validation_results.append(ValidationResult(
                rule='title_block_consistency',
                passed=True,
                severity='info',
                title='Title Blocks Consistent',
                description='Project information is consistent across sheets.'
            ))
    
    def validate_north_arrow_presence(self) -> None:
        """Check that plan sheets have north arrows."""
        plan_sheets_missing_north = []
        
        for sheet in self.sheets:
            sheet_type = sheet.get('sheet_type', '').lower()
            sheet_num = sheet.get('sheet_number', 'Unknown')
            
            # Only check plan/site sheets
            if sheet_type in ['plan', 'site', 'layout', 'grading', 'utility']:
                # Check if north arrow mentioned in elements or observations
                elements = sheet.get('elements', {})
                summary = sheet.get('summary', '').lower()
                raw = sheet.get('raw_response', '').lower() if sheet.get('raw_response') else ''
                
                has_north = (
                    'north' in summary or
                    'north' in raw or
                    any('north' in str(e).lower() for e in elements.values())
                )
                
                if not has_north:
                    plan_sheets_missing_north.append(sheet_num)
        
        if plan_sheets_missing_north:
            self.validation_results.append(ValidationResult(
                rule='north_arrow_presence',
                passed=False,
                severity='minor',
                title='North Arrow May Be Missing',
                description=f"North arrow not detected on {len(plan_sheets_missing_north)} plan sheet(s).",
                affected_sheets=plan_sheets_missing_north
            ))
        else:
            self.validation_results.append(ValidationResult(
                rule='north_arrow_presence',
                passed=True,
                severity='info',
                title='North Arrows Present',
                description='North arrows detected on plan sheets.'
            ))
    
    def validate_dimension_formats(self) -> None:
        """Check for consistent dimension formatting."""
        dimensions = self.all_elements.get('dimensions', [])
        
        if not dimensions:
            return
        
        # Look for mixed formats
        has_feet = False
        has_decimal = False
        has_fractional = False
        
        for dim in dimensions:
            text = str(dim.get('text', '')).lower()
            
            if "'" in text or 'ft' in text:
                has_feet = True
            if '.' in text:
                has_decimal = True
            if '/' in text and '-' in text:  # e.g., 3'-4 1/2"
                has_fractional = True
        
        # Mixed decimal and fractional is usually an issue
        if has_decimal and has_fractional:
            self.validation_results.append(ValidationResult(
                rule='dimension_formats',
                passed=False,
                severity='minor',
                title='Mixed Dimension Formats',
                description='Both decimal and fractional dimensions found. Consider standardizing.',
                details={'has_decimal': has_decimal, 'has_fractional': has_fractional}
            ))
        else:
            self.validation_results.append(ValidationResult(
                rule='dimension_formats',
                passed=True,
                severity='info',
                title='Dimension Format Consistent',
                description='Dimension formatting appears consistent.'
            ))
    
    def validate_annotation_consistency(self) -> None:
        """Check for consistent annotation styles."""
        annotations = self.all_elements.get('annotations', [])
        
        if not annotations:
            return
        
        # Group similar annotations
        callout_patterns = {}
        
        for ann in annotations:
            text = str(ann.get('text', '')).upper()
            
            # Extract pattern (e.g., "SEE DETAIL", "SEE SHEET", "REFER TO")
            for pattern in ['SEE DETAIL', 'SEE SHEET', 'REFER TO', 'TYP', 'MIN', 'MAX']:
                if pattern in text:
                    if pattern not in callout_patterns:
                        callout_patterns[pattern] = []
                    callout_patterns[pattern].append(text)
        
        # Check for variations in common callouts
        variations = []
        for pattern, instances in callout_patterns.items():
            unique = set(instances)
            if len(unique) > 3:  # More than 3 variations suggests inconsistency
                variations.append({
                    'pattern': pattern,
                    'count': len(instances),
                    'unique_forms': len(unique)
                })
        
        if variations:
            self.validation_results.append(ValidationResult(
                rule='annotation_consistency',
                passed=False,
                severity='minor',
                title='Annotation Variations Detected',
                description='Some annotation types have multiple variations.',
                details={'variations': variations}
            ))
        else:
            self.validation_results.append(ValidationResult(
                rule='annotation_consistency',
                passed=True,
                severity='info',
                title='Annotations Consistent',
                description='Annotation styles appear consistent.'
            ))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        passed = sum(1 for r in self.validation_results if r.passed)
        failed = len(self.validation_results) - passed
        
        by_severity = {'critical': 0, 'major': 0, 'minor': 0, 'info': 0}
        for r in self.validation_results:
            if not r.passed:
                by_severity[r.severity] += 1
        
        return {
            'total_checks': len(self.validation_results),
            'passed': passed,
            'failed': failed,
            'by_severity': by_severity,
            'results': [
                {
                    'rule': r.rule,
                    'passed': r.passed,
                    'severity': r.severity,
                    'title': r.title,
                    'description': r.description,
                    'affected_sheets': r.affected_sheets,
                    'details': r.details
                }
                for r in self.validation_results
            ]
        }
