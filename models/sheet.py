"""
Sheet Analysis Model - Analysis results for a single drawing sheet
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class SheetAnalysis:
    """Analysis results for a single sheet/page"""
    page_number: int
    sheet_number: str = ""  # e.g., "C-101", "G-001"
    sheet_title: str = ""
    sheet_type: str = ""  # cover, plan, profile, detail, etc.
    discipline: str = ""  # civil, structural, landscape, etc.
    
    # Scale info
    scale: str = ""
    scale_valid: bool = True
    
    # Title block info
    title_block: Dict[str, str] = field(default_factory=dict)
    
    # Elements found on this sheet
    elements: Dict[str, List[Dict]] = field(default_factory=dict)
    # e.g., {'dimensions': [...], 'annotations': [...], 'utilities': [...]}
    
    # References to/from other sheets
    references_to: List[str] = field(default_factory=list)  # Sheets this references
    references_from: List[str] = field(default_factory=list)  # Sheets that reference this
    
    # AI analysis summary
    summary: str = ""
    key_observations: List[str] = field(default_factory=list)
    
    # Findings/issues on this sheet
    findings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Raw AI response for debugging
    raw_analysis: str = ""
    
    # Analysis metadata
    analyzed: bool = False
    analysis_error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SheetAnalysis':
        """Create from dictionary"""
        return cls(**data)
    
    def add_finding(self, finding: Dict[str, Any]) -> None:
        """Add a finding to this sheet"""
        finding['sheet_number'] = self.sheet_number
        finding['page_number'] = self.page_number
        self.findings.append(finding)
    
    def get_element_count(self) -> Dict[str, int]:
        """Get count of elements by category"""
        return {cat: len(items) for cat, items in self.elements.items()}
    
    @property
    def finding_count(self) -> int:
        """Total number of findings"""
        return len(self.findings)
    
    @property
    def critical_findings(self) -> List[Dict]:
        """Get critical severity findings"""
        return [f for f in self.findings if f.get('severity') == 'critical']
