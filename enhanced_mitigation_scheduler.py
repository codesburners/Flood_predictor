# enhanced_mitigation_scheduler.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from rl_agent import RLMitigationAgent

class AdaptiveMitigationScheduler:
    def __init__(self, use_rl=True):
        self.use_rl = use_rl
        self.rl_agent = None
        self.mitigation_actions = {
            'Emergency_Evacuation': {'duration': 1, 'teams': 3, 'cost': 5000, 'effectiveness': 0.8},
            'Temporary_Barriers': {'duration': 3, 'teams': 2, 'cost': 3000, 'effectiveness': 0.6},
            'Pump_Installation': {'duration': 4, 'teams': 2, 'cost': 4000, 'effectiveness': 0.7},
            'Drainage_Clearing': {'duration': 2, 'teams': 1, 'cost': 800, 'effectiveness': 0.4},
            'Sandbag_Deployment': {'duration': 2, 'teams': 1, 'cost': 500, 'effectiveness': 0.3},
            'Warning_System_Activation': {'duration': 1, 'teams': 1, 'cost': 300, 'effectiveness': 0.2}
        }
        
        if use_rl:
            self.rl_agent = RLMitigationAgent({})  # Will be initialized with actual data later
    
    def optimize_schedule(self, risk_assessments, available_teams, available_budget, time_horizon=7, method='hybrid'):
        """Optimize schedule using different methods"""
        
        if method == 'rl' and self.use_rl:
            return self._rl_optimize(risk_assessments, available_teams, available_budget, time_horizon)
        elif method == 'hybrid':
            return self._hybrid_optimize(risk_assessments, available_teams, available_budget, time_horizon)
        else:
            return self._rule_based_optimize(risk_assessments, available_teams, available_budget, time_horizon)
    
    def _rl_optimize(self, risk_assessments, available_teams, available_budget, time_horizon):
        """Use RL for optimization"""
        try:
            # Initialize RL agent with current districts data
            districts_data = {loc: data['location_data'] for loc, data in risk_assessments.items()}
            current_risks = {loc: data['predicted_risk'] for loc, data in risk_assessments.items()}
            
            if self.rl_agent is None:
                self.rl_agent = RLMitigationAgent(districts_data)
            
            schedule, info = self.rl_agent.predict_schedule(
                current_risks, available_teams, available_budget, time_horizon
            )
            
            metrics = self._calculate_metrics(schedule, risk_assessments, available_teams, time_horizon)
            metrics['optimization_method'] = 'Reinforcement Learning'
            metrics['final_total_risk'] = info.get('total_risk', 0)
            
            return schedule, metrics
            
        except Exception as e:
            print(f"RL optimization failed: {e}. Falling back to rule-based.")
            return self._rule_based_optimize(risk_assessments, available_teams, available_budget, time_horizon)
    
    def _hybrid_optimize(self, risk_assessments, available_teams, available_budget, time_horizon):
        """Combine RL and rule-based approaches"""
        # Use RL for high-risk scenarios, rule-based for others
        high_risk_locations = {
            loc: data for loc, data in risk_assessments.items() 
            if data['predicted_risk'] >= 7
        }
        
        if len(high_risk_locations) >= 2 and self.use_rl:
            # Use RL for complex high-risk scenarios
            rl_schedule, rl_metrics = self._rl_optimize(
                high_risk_locations, available_teams // 2, available_budget // 2, time_horizon
            )
            
            # Use rule-based for remaining locations
            remaining_locations = {
                loc: data for loc, data in risk_assessments.items() 
                if data['predicted_risk'] < 7
            }
            rule_schedule, rule_metrics = self._rule_based_optimize(
                remaining_locations, available_teams // 2, available_budget // 2, time_horizon
            )
            
            combined_schedule = rl_schedule + rule_schedule
            combined_metrics = self._calculate_metrics(
                combined_schedule, risk_assessments, available_teams, time_horizon
            )
            combined_metrics['optimization_method'] = 'Hybrid (RL + Rule-Based)'
            
            return combined_schedule, combined_metrics
        else:
            return self._rule_based_optimize(risk_assessments, available_teams, available_budget, time_horizon)
    
    def _rule_based_optimize(self, risk_assessments, available_teams, available_budget, time_horizon):
        """Traditional rule-based optimization"""
        # Your existing rule-based logic here
        locations_sorted = sorted(
            [(loc, data) for loc, data in risk_assessments.items()],
            key=lambda x: x[1]['predicted_risk'],
            reverse=True
        )
        
        schedule = []
        resource_usage = {day: {'teams': 0, 'budget': 0} for day in range(1, time_horizon + 1)}
        budget_remaining = available_budget
        
        for location, assessment in locations_sorted:
            if assessment['predicted_risk'] < 3:
                continue
                
            recommended_actions = self._get_recommended_actions(assessment['predicted_risk'])
            
            for action_name in recommended_actions:
                action = self.mitigation_actions[action_name]
                
                if action['cost'] > budget_remaining:
                    continue
                    
                start_day = self._find_optimal_slot(action, resource_usage, time_horizon)
                if start_day is not None:
                    schedule_item = {
                        'location': location,
                        'action': action_name,
                        'start_day': start_day,
                        'duration': action['duration'],
                        'end_day': start_day + action['duration'] - 1,
                        'teams_required': action['teams'],
                        'cost': action['cost'],
                        'effectiveness': action['effectiveness'],
                        'risk_reduction': assessment['predicted_risk'] * action['effectiveness'],
                        'predicted_risk': assessment['predicted_risk'],
                        'method': 'Rule-Based'
                    }
                    
                    schedule.append(schedule_item)
                    self._update_resource_usage(schedule_item, resource_usage)
                    budget_remaining -= action['cost']
                    break
        
        metrics = self._calculate_metrics(schedule, risk_assessments, available_teams, time_horizon)
        metrics['optimization_method'] = 'Rule-Based'
        
        return schedule, metrics
    
    def _get_recommended_actions(self, risk_score):
        """Get recommended actions based on risk score"""
        if risk_score >= 8:
            return ['Emergency_Evacuation', 'Pump_Installation', 'Temporary_Barriers']
        elif risk_score >= 6:
            return ['Temporary_Barriers', 'Pump_Installation', 'Drainage_Clearing']
        elif risk_score >= 4:
            return ['Drainage_Clearing', 'Sandbag_Deployment', 'Warning_System_Activation']
        else:
            return ['Warning_System_Activation']
    
    def _find_optimal_slot(self, action, resource_usage, time_horizon):
        """Find optimal time slot"""
        for start_day in range(1, time_horizon - action['duration'] + 2):
            slot_available = True
            for day in range(start_day, start_day + action['duration']):
                if day > time_horizon or resource_usage[day]['teams'] + action['teams'] > 10:
                    slot_available = False
                    break
            if slot_available:
                return start_day
        return None
    
    def _update_resource_usage(self, schedule_item, resource_usage):
        """Update resource usage"""
        for day in range(schedule_item['start_day'], schedule_item['start_day'] + schedule_item['duration']):
            if day in resource_usage:
                resource_usage[day]['teams'] += schedule_item['teams_required']
    
    def _calculate_metrics(self, schedule, risk_assessments, available_teams, time_horizon):
        """Calculate performance metrics"""
        total_cost = sum(item['cost'] for item in schedule)
        total_risk_reduction = sum(item['risk_reduction'] for item in schedule)
        covered_locations = set(item['location'] for item in schedule)
        
        team_days_used = sum(item['teams_required'] * item['duration'] for item in schedule)
        max_team_days = available_teams * time_horizon
        resource_utilization = (team_days_used / max_team_days) * 100 if max_team_days > 0 else 0
        
        return {
            'total_actions': len(schedule),
            'total_cost': total_cost,
            'total_risk_reduction': total_risk_reduction,
            'locations_covered': len(covered_locations),
            'total_locations': len(risk_assessments),
            'coverage_percentage': (len(covered_locations) / len(risk_assessments)) * 100,
            'resource_utilization': resource_utilization,
            'cost_per_risk_reduction': total_cost / total_risk_reduction if total_risk_reduction > 0 else 0,
            'average_risk_reduction_per_action': total_risk_reduction / len(schedule) if schedule else 0
        }