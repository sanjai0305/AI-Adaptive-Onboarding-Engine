# backend/multi_role_analyzer.py
"""
Multi-Role Comparison & Analysis
=================================
Compare candidate skills against multiple roles.
Identify best-fit role and transferable skills.
"""

from typing import List, Dict, Tuple
import pandas as pd
from backend.matcher import SkillMatcher
from backend.skill_extractor import SkillExtractor

class MultiRoleAnalyzer:
    """Analyze skills against multiple job descriptions."""
    
    @staticmethod
    def analyze_multiple_roles(
        resume_skills: List[str],
        role_descriptions: Dict[str, str]
    ) -> Dict[str, Dict]:
        """
        Analyze how well candidate fits multiple roles.
        
        Args:
            resume_skills (List[str]): Candidate's skills
            role_descriptions (Dict[str, str]): {role_name: jd_text}
            
        Returns:
            Dict: Comprehensive analysis for each role
        """
        analysis = {}
        
        for role_name, jd_text in role_descriptions.items():
            # Extract role skills
            role_skills = SkillExtractor.extract_skills(jd_text)
            
            # Match skills
            matched, missing, weak = SkillMatcher.match_skills(resume_skills, role_skills)
            
            # Calculate fit score
            fit_score = MultiRoleAnalyzer._calculate_fit_score(
                matched, missing, weak, role_skills
            )
            
            analysis[role_name] = {
                'fit_score': fit_score,
                'matched_count': len(matched),
                'weak_count': len(weak),
                'missing_count': len(missing),
                'total_required': len(role_skills),
                'matched_skills': [m['skill'] for m in matched],
                'missing_skills': [m['skill'] for m in missing],
                'weak_skills': [w['skill'] for w in weak],
                'transferable_skills': MultiRoleAnalyzer._find_transferable(
                    matched, weak, resume_skills
                ),
                'gap_percentage': (len(missing) + len(weak)) / len(role_skills) * 100 if role_skills else 0
            }
        
        return analysis
    
    @staticmethod
    def _calculate_fit_score(
        matched: List,
        missing: List,
        weak: List,
        total_skills: List
    ) -> float:
        """
        Calculate overall fit score (0-100).
        
        Weighted formula:
        - Matched: 100% weight
        - Weak: 50% weight
        - Missing: 0% weight
        """
        if not total_skills:
            return 0.0
        
        total = len(total_skills)
        score = (len(matched) * 100 + len(weak) * 50) / (total * 100)
        return round(score * 100, 2)
    
    @staticmethod
    def _find_transferable(matched: List, weak: List, resume_skills: List) -> List[str]:
        """Find skills that are transferable to other roles."""
        matched_weak = matched + weak
        matched_skill_names = {m['skill'] if isinstance(m, dict) else m for m in matched_weak}
        
        # Skills in resume but not explicitly in JD
        transferable = [
            s for s in resume_skills 
            if s not in matched_skill_names
        ]
        
        return transferable[:5]  # Top 5 transferable skills
    
    @staticmethod
    def rank_roles(analysis: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """
        Rank roles by fit score.
        
        Returns:
            List: [(role_name, fit_score), ...]
        """
        ranked = sorted(
            analysis.items(),
            key=lambda x: x[1]['fit_score'],
            reverse=True
        )
        return [(role, score['fit_score']) for role, score in ranked]
    
    @staticmethod
    def generate_comparison_report(analysis: Dict[str, Dict]) -> pd.DataFrame:
        """Generate comparison table."""
        data = []
        for role_name, metrics in analysis.items():
            data.append({
                'Role': role_name,
                'Fit Score': metrics['fit_score'],
                'Matched': metrics['matched_count'],
                'Weak': metrics['weak_count'],
                'Missing': metrics['missing_count'],
                'Gap %': metrics['gap_percentage']
            })
        
        return pd.DataFrame(data).sort_values('Fit Score', ascending=False)