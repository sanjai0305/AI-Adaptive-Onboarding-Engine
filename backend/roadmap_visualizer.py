# backend/roadmap_visualizer.py
"""
Advanced Roadmap Visualization
===============================
Generate interactive roadmaps, skill trees, prerequisites visualization.
"""

import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List

class RoadmapVisualizer:
    """Create interactive visualizations of learning roadmaps."""
    
    @staticmethod
    def create_skill_tree(skills_graph: nx.DiGraph) -> go.Figure:
        """
        Create interactive skill tree visualization.
        
        Args:
            skills_graph (nx.DiGraph): Skill dependency graph
            
        Returns:
            plotly Figure: Interactive visualization
        """
        pos = nx.spring_layout(skills_graph, k=2, iterations=50)
        
        edge_x = []
        edge_y = []
        
        for edge in skills_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            showlegend=False
        )
        
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in skills_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Color by in-degree (dependencies)
            node_color.append(skills_graph.in_degree(node))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                showscale=True,
                color=node_color,
                size=20,
                colorscale='Viridis',
                line_width=2
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title='Skill Dependency Tree',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_learning_path_gantt(roadmap: Dict) -> go.Figure:
        """
        Create Gantt chart for learning timeline.
        
        Args:
            roadmap (Dict): Roadmap with skills and durations
            
        Returns:
            plotly Figure: Gantt chart
        """
        from datetime import datetime, timedelta
        
        tasks = []
        start_date = datetime.now()
        
        for i, (skill, data) in enumerate(roadmap.items()):
            duration = data.get('estimated_hours', 40)
            end_date = start_date + timedelta(hours=duration)
            
            tasks.append({
                'Task': skill,
                'Start': start_date,
                'Finish': end_date,
                'Duration': duration
            })
            
            start_date = end_date + timedelta(days=1)  # 1 day gap between skills
        
        import pandas as pd
        df = pd.DataFrame(tasks)
        
        fig = go.Figure(data=[
            go.Bar(
                y=df['Task'],
                x=df['Duration'],
                orientation='h',
                marker=dict(color='lightblue')
            )
        ])
        
        fig.update_layout(
            title='Learning Timeline (by hours)',
            xaxis_title='Hours',
            yaxis_title='Skill',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_skill_gap_sunburst(matched: List, weak: List, missing: List) -> go.Figure:
        """
        Create sunburst chart for skill gaps.
        
        Args:
            matched, weak, missing: Skill lists
            
        Returns:
            plotly Figure: Sunburst visualization
        """
        labels = ['Skill Gap', 'Matched', 'Weak', 'Missing']
        parents = ['', 'Skill Gap', 'Skill Gap', 'Skill Gap']
        values = [0, len(matched), len(weak), len(missing)]
        colors = ['#fff', '#2ecc71', '#f39c12', '#e74c3c']
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            hovertext=[
                'Total Skills',
                f'{len(matched)} Skills',
                f'{len(weak)} Skills',
                f'{len(missing)} Skills'
            ],
            hoverinfo='label+text+value'
        ))
        
        fig.update_layout(
            title='Skill Gap Distribution',
            height=600
        )
        
        return fig