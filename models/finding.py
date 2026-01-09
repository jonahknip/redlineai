"""
Finding Model - Issues/observations found during analysis
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid


@dataclass
class Finding:
    """A single finding/issue from analysis"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Location
    page_number: int = 0
    sheet_number: str = ""
    location: Dict[str, float] = field(default_factory=dict)
    # location = {'x': 0.5, 'y': 0.3, 'width': 0.1, 'height': 0.05}
    # Normalized coordinates (0-1) for PDF annotation
    
    # Classification
    category: str = ""  # dimension, annotation, reference, scale, etc.
    severity: str = "info"  # critical, major, minor, info
    finding_type: str = ""  # missing, incorrect, inconsistent, unclear
    
    # Description
    title: str = ""
    description: str = ""
    recommendation: str = ""
    
    # Related elements
    related_elements: List[str] = field(default_factory=list)
    related_sheets: List[str] = field(default_factory=list)
    
    # Status (for tracking)
    status: str = "open"  # open, acknowledged, resolved, wont_fix
    resolved_at: Optional[str] = None
    resolution_note: str = ""
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "ai"  # ai, cross_validation, user
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Finding':
        """Create from dictionary"""
        return cls(**data)
    
    def resolve(self, note: str = "") -> None:
        """Mark finding as resolved"""
        self.status = "resolved"
        self.resolved_at = datetime.now().isoformat()
        self.resolution_note = note
    
    @property
    def is_critical(self) -> bool:
        return self.severity == 'critical'
    
    @property
    def is_open(self) -> bool:
        return self.status == 'open'
    
    def get_annotation_color(self) -> str:
        """Get color for PDF annotation based on severity"""
        colors = {
            'critical': (1, 0, 0),      # Red
            'major': (1, 0.5, 0),       # Orange  
            'minor': (1, 0.8, 0),       # Yellow
            'info': (0, 0.5, 1),        # Blue
        }
        return colors.get(self.severity, (0.5, 0.5, 0.5))


def create_finding(
    title: str,
    description: str,
    severity: str = "info",
    category: str = "",
    page_number: int = 0,
    sheet_number: str = "",
    location: Dict[str, float] = None,
    recommendation: str = "",
    source: str = "ai"
) -> Finding:
    """Helper to create a Finding with common fields"""
    return Finding(
        title=title,
        description=description,
        severity=severity,
        category=category,
        page_number=page_number,
        sheet_number=sheet_number,
        location=location or {},
        recommendation=recommendation,
        source=source
    )
