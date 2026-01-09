"""
RedlineAI - AI-powered civil engineering plan review agent
"""

__all__ = ['PlanReviewer', 'ChecklistEngine', 'ReportGenerator']
__version__ = '0.1.0'

# Lazy imports to avoid import errors during startup
def __getattr__(name):
    if name == 'PlanReviewer':
        from .plan_reviewer import PlanReviewer
        return PlanReviewer
    elif name == 'ChecklistEngine':
        from .checklist_engine import ChecklistEngine
        return ChecklistEngine
    elif name == 'ReportGenerator':
        from .report_generator import ReportGenerator
        return ReportGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
