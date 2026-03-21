# backend/advanced_parser.py
"""
Advanced Resume Parser with Entity Extraction
==============================================
Extracts: name, email, phone, experience, education, certifications.
"""

import re
from typing import Dict, List
import pdfplumber
from docx import Document

class AdvancedResumeParser:
    """
    Advanced resume parsing with structured information extraction.
    Extracts: Contact info, Education, Experience, Certifications.
    """
    
    @staticmethod
    def parse_resume_comprehensive(filepath: str) -> Dict:
        """
        Parse resume and extract structured information.
        
        Returns:
            Dict with: personal_info, education, experience, certifications, skills
        """
        text = AdvancedResumeParser._extract_text(filepath)
        
        return {
            'personal_info': AdvancedResumeParser._extract_personal_info(text),
            'education': AdvancedResumeParser._extract_education(text),
            'experience': AdvancedResumeParser._extract_experience(text),
            'certifications': AdvancedResumeParser._extract_certifications(text),
            'raw_text': text
        }
    
    @staticmethod
    def _extract_text(filepath: str) -> str:
        """Extract text from PDF or DOCX."""
        if filepath.endswith('.pdf'):
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        elif filepath.endswith('.docx'):
            doc = Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            raise ValueError("Unsupported format")
    
    @staticmethod
    def _extract_personal_info(text: str) -> Dict:
        """Extract name, email, phone."""
        info = {}
        
        # Email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_match:
            info['email'] = email_match.group()
        
        # Phone (various formats)
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            info['phone'] = phone_match.group()
        
        # Name (first non-email line with title case)
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            if re.match(r'^[A-Z][a-z]+ [A-Z]', line.strip()):
                info['name'] = line.strip()
                break
        
        return info
    
    @staticmethod
    def _extract_education(text: str) -> List[Dict]:
        """Extract education history."""
        education = []
        
        degree_patterns = [
            r'(B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|Ph\.?D\.?|MBA)',
            r'(Bachelor|Master|Doctorate|PhD|Associate)',
            r'(Computer Science|Information Technology|Data Science|Mathematics)'
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            for pattern in degree_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    education.append({
                        'degree': line.strip(),
                        'context': '\n'.join(lines[max(0, i-1):min(len(lines), i+3)])
                    })
                    break
        
        return education
    
    @staticmethod
    def _extract_experience(text: str) -> List[Dict]:
        """Extract work experience."""
        experience = []
        
        # Look for common job title patterns
        job_titles = [
            'Software Engineer', 'Data Scientist', 'Product Manager',
            'Designer', 'Developer', 'Analyst', 'Consultant'
        ]
        
        for title in job_titles:
            if title.lower() in text.lower():
                # Extract surrounding context
                idx = text.lower().find(title.lower())
                context = text[max(0, idx-100):min(len(text), idx+300)]
                
                experience.append({
                    'title': title,
                    'context': context.strip()
                })
        
        return experience
    
    @staticmethod
    def _extract_certifications(text: str) -> List[str]:
        """Extract certifications and credentials."""
        certifications = []
        
        cert_keywords = [
            'AWS', 'Azure', 'Google Cloud', 'Kubernetes',
            'PMP', 'CISSP', 'CCNA', 'CCNP',
            'CPA', 'CFA', 'FRM',
            'Certified', 'License', 'Credential'
        ]
        
        for cert in cert_keywords:
            if cert.lower() in text.lower():
                certifications.append(cert)
        
        return list(set(certifications))  # Deduplicate