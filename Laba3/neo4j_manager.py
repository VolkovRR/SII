# neo4j_manager.py
from neo4j import GraphDatabase

class Neo4jRuleManager:
    """Менеджер для работы с правилами в Neo4j"""
    
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def create_navigation_ontology(self):
        """Создание онтологии навигации в Neo4j"""
        with self.driver.session() as session:
            # Очистка базы
            session.run("MATCH (n) DETACH DELETE n")
            
            # Создание узлов условий с совместимыми названиями
            session.run("""
            CREATE 
            // Погодные условия (совместимость с основной системой)
            (calm:WindCondition {type: 'calm', max_speed: 5}),
            (moderate:WindCondition {type: 'moderate', min_speed: 5, max_speed: 15}),
            (strong:WindCondition {type: 'strong', min_speed: 15}),
            
            // Условия видимости (совместимость с основной системой)
            (low:VisibilityCondition {type: 'low', max_visibility: 800}),
            (moderate:VisibilityCondition {type: 'moderate', min_visibility: 800, max_visibility: 2000}),
            (optimal:VisibilityCondition {type: 'optimal', min_visibility: 2000}),
            
            // Условия осадков (новые узлы для полноты)
            (none:PrecipitationCondition {type: 'none', max_precipitation: 1}),
            (light:PrecipitationCondition {type: 'light', min_precipitation: 1, max_precipitation: 5}),
            (heavy:PrecipitationCondition {type: 'heavy', min_precipitation: 5}),
            
            // Препятствия
            (low:ObstacleDensity {type: 'low', max_density: 0.3}),
            (medium:ObstacleDensity {type: 'medium', min_density: 0.3, max_density: 0.7}),
            (high:ObstacleDensity {type: 'high', min_density: 0.7}),
            
            // Навигационные действия (совместимость с основной системой)
            (maintain_course:NavigationAction {name: 'maintain_course', altitude_change: 0, speed_change: 0}),
            (avoid_obstacles:NavigationAction {name: 'avoid_obstacles', altitude_change: 40, speed_change: -5}),
            (reduce_speed_altitude:NavigationAction {name: 'reduce_speed_altitude', altitude_change: -25, speed_change: -8}),
            (cautious_navigation:NavigationAction {name: 'cautious_navigation', altitude_change: 15, speed_change: -6}),
            (reduce_speed:NavigationAction {name: 'reduce_speed', altitude_change: -10, speed_change: -10}),
            (increase_altitude:NavigationAction {name: 'increase_altitude', altitude_change: 30, speed_change: -3})
            """)
            
            # Создание правил (совместимость с основной системой)
            session.run("""
            // Идеальные условия -> Поддерживать курс
            MATCH (w:WindCondition {type: 'calm'}), (v:VisibilityCondition {type: 'optimal'}), 
                  (p:PrecipitationCondition {type: 'none'}), (d:ObstacleDensity {type: 'low'}), 
                  (a:NavigationAction {name: 'maintain_course'})
            CREATE (w)-[:LEADS_TO {confidence: 1.0}]->(a)
            CREATE (v)-[:LEADS_TO {confidence: 1.0}]->(a)
            CREATE (p)-[:LEADS_TO {confidence: 1.0}]->(a)
            CREATE (d)-[:LEADS_TO {confidence: 1.0}]->(a)
            
            // Сильный ветер + плохая видимость + сильные осадки -> Снизить скорость и высоту
            MATCH (w:WindCondition {type: 'strong'}), (v:VisibilityCondition {type: 'low'}), 
                  (p:PrecipitationCondition {type: 'heavy'}), (d:ObstacleDensity {type: 'low'}), 
                  (a:NavigationAction {name: 'reduce_speed_altitude'})
            CREATE (w)-[:LEADS_TO {confidence: 1.0}]->(a)
            CREATE (v)-[:LEADS_TO {confidence: 1.0}]->(a)
            CREATE (p)-[:LEADS_TO {confidence: 1.0}]->(a)
            CREATE (d)-[:LEADS_TO {confidence: 1.0}]->(a)
            
            // Умеренный ветер + много препятствий -> Осторожная навигация
            MATCH (w:WindCondition {type: 'moderate'}), (v:VisibilityCondition {type: 'low'}), 
                  (p:PrecipitationCondition {type: 'light'}), (d:ObstacleDensity {type: 'medium'}), 
                  (a:NavigationAction {name: 'cautious_navigation'})
            CREATE (w)-[:LEADS_TO {confidence: 0.8}]->(a)
            CREATE (v)-[:LEADS_TO {confidence: 0.7}]->(a)
            CREATE (p)-[:LEADS_TO {confidence: 0.8}]->(a)
            CREATE (d)-[:LEADS_TO {confidence: 0.7}]->(a)
            
            // Умеренный ветер + оптимальная видимость + много препятствий -> Облет препятствий
            MATCH (w:WindCondition {type: 'moderate'}), (v:VisibilityCondition {type: 'optimal'}), 
                  (p:PrecipitationCondition {type: 'none'}), (d:ObstacleDensity {type: 'high'}), 
                  (a:NavigationAction {name: 'avoid_obstacles'})
            CREATE (w)-[:LEADS_TO {confidence: 0.8}]->(a)
            CREATE (v)-[:LEADS_TO {confidence: 1.0}]->(a)
            CREATE (p)-[:LEADS_TO {confidence: 1.0}]->(a)
            CREATE (d)-[:LEADS_TO {confidence: 1.0}]->(a)
            """)
    
    def get_actions_from_rules(self, wind_speed, visibility, precipitation, obstacle_density):
        """Получение действий из базы правил Neo4j на основе точных значений"""
        
        # Сначала определяем категории как в основной системе
        wind_cat = 'calm' if wind_speed < 5 else 'moderate' if wind_speed < 15 else 'strong'
        vis_cat = 'optimal' if visibility > 2000 else 'moderate' if visibility > 800 else 'low'
        prec_cat = 'none' if precipitation < 1 else 'light' if precipitation < 5 else 'heavy'
        obs_cat = 'low' if obstacle_density < 0.3 else 'medium' if obstacle_density < 0.7 else 'high'
        
        conditions = [wind_cat, vis_cat, prec_cat, obs_cat]
        
        query = """
        MATCH (condition)-[r:LEADS_TO]->(action:NavigationAction)
        WHERE condition.type IN $conditions
        RETURN action.name as action, 
               action.altitude_change as altitude_change,
               action.speed_change as speed_change,
               r.confidence as confidence,
               condition.type as condition_type
        ORDER BY r.confidence DESC
        """
        
        with self.driver.session() as session:
            result = session.run(query, conditions=conditions)
            actions = [dict(record) for record in result]
            
            # Группируем действия и вычисляем среднюю уверенность
            action_groups = {}
            for act in actions:
                key = act['action']
                if key not in action_groups:
                    action_groups[key] = {
                        'action': act['action'],
                        'altitude_change': act['altitude_change'],
                        'speed_change': act['speed_change'],
                        'confidences': [],
                        'conditions': set()
                    }
                action_groups[key]['confidences'].append(act['confidence'])
                action_groups[key]['conditions'].add(act['condition_type'])
            
            # Вычисляем среднюю уверенность для каждого действия
            final_actions = []
            for action_data in action_groups.values():
                avg_confidence = sum(action_data['confidences']) / len(action_data['confidences'])
                final_actions.append({
                    'action': action_data['action'],
                    'altitude_change': action_data['altitude_change'],
                    'speed_change': action_data['speed_change'],
                    'confidence': avg_confidence,
                    'matching_conditions': len(action_data['conditions'])
                })
            
            # Сортируем по количеству совпавших условий и уверенности
            final_actions.sort(key=lambda x: (-x['matching_conditions'], -x['confidence']))
            
            return final_actions
    
    def close(self):
        self.driver.close()