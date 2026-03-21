# backend/content_recommender.py
"""
Advanced Content Recommendation Engine
======================================
Fetch real learning resources from APIs:
- Coursera, Udemy, YouTube, GitHubYour, Dev.to
"""

import requests
from typing import List, Dict
import os

class ContentRecommender:
    """Fetch and recommend learning resources from real platforms."""
    
    @staticmethod
    def get_youtube_courses(skill: str, limit: int = 5) -> List[Dict]:
        """
        Fetch YouTube courses for skill.
        Note: Requires YouTube Data API key.
        """
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            return ContentRecommender._get_fallback_youtube(skill)
        
        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                'part': 'snippet',
                'q': f'{skill} tutorial course',
                'type': 'video',
                'order': 'viewCount',
                'maxResults': limit,
                'key': api_key
            }
            
            response = requests.get(url, params=params)
            results = response.json().get('items', [])
            
            resources = []
            for item in results:
                resources.append({
                    'title': item['snippet']['title'],
                    'url': f"https://youtube.com/watch?v={item['id']['videoId']}",
                    'channel': item['snippet']['channelTitle'],
                    'platform': 'YouTube'
                })
            
            return resources
        except:
            return ContentRecommender._get_fallback_youtube(skill)
    
    @staticmethod
    def _get_fallback_youtube(skill: str) -> List[Dict]:
        """Fallback YouTube recommendations."""
        return [
            {
                'title': f'{skill} Full Course by Traversy Media',
                'url': f'https://www.youtube.com/results?search_query={skill}+tutorial',
                'channel': 'Traversy Media',
                'platform': 'YouTube'
            }
        ]
    
    @staticmethod
    def get_github_projects(skill: str, limit: int = 5) -> List[Dict]:
        """
        Fetch GitHub projects for skill learning.
        """
        try:
            url = "https://api.github.com/search/repositories"
            params = {
                'q': f'topic:{skill.lower()} language:python',
                'sort': 'stars',
                'order': 'desc',
                'per_page': limit
            }
            
            response = requests.get(url, params=params)
            repos = response.json().get('items', [])
            
            projects = []
            for repo in repos:
                projects.append({
                    'name': repo['name'],
                    'url': repo['html_url'],
                    'description': repo['description'],
                    'stars': repo['stargazers_count'],
                    'platform': 'GitHub'
                })
            
            return projects
        except:
            return []
    
    @staticmethod
    def get_dev_articles(skill: str, limit: int = 5) -> List[Dict]:
        """
        Fetch Dev.to articles for skill.
        """
        try:
            url = "https://dev.to/api/articles"
            params = {
                'tag': skill.lower(),
                'per_page': limit
            }
            
            response = requests.get(url, params=params)
            articles = response.json()
            
            resources = []
            for article in articles:
                resources.append({
                    'title': article['title'],
                    'url': article['url'],
                    'author': article['user']['username'],
                    'published': article['published_at'],
                    'platform': 'Dev.to'
                })
            
            return resources
        except:
            return []
    
    @staticmethod
    def get_documentation(skill: str) -> List[Dict]:
        """Get official documentation links."""
        doc_links = {
            'Python': 'https://docs.python.org/3/',
            'JavaScript': 'https://developer.mozilla.org/en-US/docs/Web/JavaScript/',
            'SQL': 'https://www.postgresql.org/docs/',
            'Machine Learning': 'https://scikit-learn.org/stable/',
            'TensorFlow': 'https://www.tensorflow.org/api',
            'PyTorch': 'https://pytorch.org/docs/',
            'Docker': 'https://docs.docker.com/',
            'Kubernetes': 'https://kubernetes.io/docs/',
            'AWS': 'https://docs.aws.amazon.com/',
            'Azure': 'https://learn.microsoft.com/en-us/azure/',
            'React': 'https://react.dev/',
            'FastAPI': 'https://fastapi.tiangolo.com/',
            'Django': 'https://docs.djangoproject.com/',
        }
        
        if skill in doc_links:
            return [{
                'title': f'{skill} Official Documentation',
                'url': doc_links[skill],
                'platform': 'Official Docs'
            }]
        
        return []
    
    @staticmethod
    def curate_learning_path(skill: str) -> Dict[str, List[Dict]]:
        """
        Curate complete learning path with multiple resource types.
        """
        return {
            'documentation': ContentRecommender.get_documentation(skill),
            'youtube': ContentRecommender.get_youtube_courses(skill),
            'github': ContentRecommender.get_github_projects(skill),
            'articles': ContentRecommender.get_dev_articles(skill)
        }