from neo4j import GraphDatabase
import numpy as np
import random

class NavigationAction:
    def __init__(self, name, altitude_change, speed_change, confidence):
        self.name = name
        self.altitude_change = altitude_change
        self.speed_change = speed_change
        self.confidence = confidence

class FuzzyDroneNavigationSystem:
    def __init__(self):
        # База правил нечеткой логики
        self.rules = {
            ('calm', 'optimal', 'none', 'low'): {
                'action': 'maintain_course', 'altitude': 0, 'speed': 0
            },
            ('strong', 'low', 'heavy', 'low'): {
                'action': 'reduce_speed_altitude', 'altitude': -25, 'speed': -8
            },
            # ... другие правила
        }
        
    def fuzzify_wind(self, wind_speed):
        # Фаззификация скорости ветра
        if wind_speed < 5:
            return 'calm', 1.0
        elif wind_speed < 15:
            return 'moderate', 0.8
        else:
            return 'strong', 1.0
    
    def fuzzify_visibility(self, visibility):
        # Фаззификация видимости
        if visibility > 2000:
            return 'optimal', 1.0
        elif visibility > 800:
            return 'moderate', 0.7
        else:
            return 'low', 1.0
    
    def fuzzify_precipitation(self, precipitation):
        # Фаззификация осадков
        if precipitation < 1:
            return 'none', 1.0
        elif precipitation < 5:
            return 'light', 0.8
        else:
            return 'heavy', 1.0
    
    def fuzzify_obstacles(self, density, height):
        # Фаззификация препятствий
        if density < 0.3:
            return 'low', 1.0
        elif density < 0.7:
            return 'medium', 0.7
        else:
            return 'high', 1.0
    
    def get_navigation_action(self, sensor_data):
        # Получение действия на основе данных сенсоров
        wind_cat, wind_conf = self.fuzzify_wind(sensor_data['wind_speed'])
        vis_cat, vis_conf = self.fuzzify_visibility(sensor_data['visibility'])
        prec_cat, prec_conf = self.fuzzify_precipitation(sensor_data['precipitation'])
        obs_cat, obs_conf = self.fuzzify_obstacles(
            sensor_data['obstacle_density'], 
            sensor_data['obstacle_height']
        )
        
        # Поиск подходящего правила
        rule_key = (wind_cat, vis_cat, prec_cat, obs_cat)
        
        if rule_key in self.rules:
            rule = self.rules[rule_key]
            confidence = min(wind_conf, vis_conf, prec_conf, obs_conf)
            return NavigationAction(
                rule['action'],
                rule['altitude'],
                rule['speed'],
                confidence
            )
        else:
            # Резервные действия по умолчанию
            if wind_cat == 'strong':
                return NavigationAction('wind_compensation', -15, -5, 0.6)
            elif vis_cat == 'low':
                return NavigationAction('low_visibility', 10, -8, 0.7)
            elif obs_cat == 'high':
                return NavigationAction('obstacle_avoidance', 25, -7, 0.8)
            else:
                return NavigationAction('maintain_course', 0, 0, 0.5)

class DroneSimulator:
    def __init__(self, navigation_system):
        self.nav_system = navigation_system
        self.position = np.array([0.0, 0.0])
        self.altitude = 100.0
        self.speed = 15.0
        self.target = np.array([150.0, 150.0])
        
        # Параметры окружающей среды
        self.wind_speed = 8.0
        self.visibility = 1800.0
        self.precipitation = 1.5
        self.obstacle_density = 0.4
        self.obstacle_height = 25.0
    
    def update_environment(self):
        # Динамическое обновление условий среды
        self.wind_speed += random.uniform(-1.5, 1.5)
        self.wind_speed = max(0, min(25, self.wind_speed))
        
        self.visibility += random.uniform(-80, 50)
        self.visibility = max(300, min(5000, self.visibility))
        
        # ... аналогично для других параметров
        
        # Случайные события
        if random.random() < 0.1:
            if random.random() < 0.5:
                self.wind_speed = random.uniform(12, 18)
            else:
                self.visibility = random.uniform(600, 1000)
    
    def get_sensor_data(self):
        # Сбор данных с сенсоров
        distance_to_target = np.linalg.norm(self.target - self.position)
        
        return {
            'wind_speed': self.wind_speed,
            'visibility': self.visibility,
            'precipitation': self.precipitation,
            'obstacle_density': self.obstacle_density,
            'obstacle_height': self.obstacle_height,
            'distance_to_target': distance_to_target
        }
    
    def apply_action(self, action):
        # Применение навигационного действия
        self.altitude += action.altitude_change
        self.speed += action.speed_change
        
        # Ограничения по безопасности
        self.altitude = max(20, min(500, self.altitude))
        self.speed = max(2, min(30, self.speed))
        
        # Движение к цели
        direction = self.target - self.position
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            self.position += direction * self.speed
    
    def simulate(self, steps=25):
        # Основной цикл симуляции
        print("СИМУЛЯЦИЯ ПОЛЕТА ДРОНА")
        print(f"Цель: {self.target}")
        print(f"Старт: {self.position}")
        print(f"Начальное расстояние: {np.linalg.norm(self.target - self.position):.1f}м")
        
        actions_count = {}
        
        for step in range(steps):
            self.update_environment()
            sensor_data = self.get_sensor_data()
            action = self.nav_system.get_navigation_action(sensor_data)
            self.apply_action(action)
            
            # Статистика действий
            if action.name in actions_count:
                actions_count[action.name] += 1
            else:
                actions_count[action.name] = 1
            
            distance = sensor_data['distance_to_target']
            
            # Вывод информации о шаге
            print(f"Шаг {step + 1:2d}: "
                  f"  Поз({self.position[0]:6.1f}, {self.position[1]:6.1f}) | "
                  f"  Выс: {self.altitude:5.1f}м | "
                  f"  Скор: {self.speed:4.1f}м/с | "
                  f"  До цели: {distance:6.1f}м")
            print(f"  Действие: {action.name:20} | "
                  f"  Уверенность: {action.confidence:.2f}")
            
            # Условия завершения
            if distance < 20:
                print(f"ЦЕЛЬ ДОСТИГНУТА на шаге {step + 1}!")
                break
            
            if self.altitude < 15:
                print("АВАРИЙНАЯ ОСТАНОВКА!")
                break
        
        # Вывод статистики
        print("\nСТАТИСТИКА ДЕЙСТВИЙ:")
        for action, count in actions_count.items():
            print(f"  {action}: {count} раз")
        
        return distance < 20

def demo_system():
    # Демонстрация работы системы
    print("ДЕМОНСТРАЦИЯ СИСТЕМЫ")
    
    nav_system = FuzzyDroneNavigationSystem()
    
    test_scenarios = [
        {'wind': 3, 'visibility': 4000, 'precip': 0, 'density': 0.1, 'name': 'Идеальные условия'},
        {'wind': 22, 'visibility': 500, 'precip': 8, 'density': 0.2, 'name': 'Штормовые условия'},
        # ... другие сценарии
    ]
    
    for scenario in test_scenarios:
        sensor_data = {
            'wind_speed': scenario['wind'],
            'visibility': scenario['visibility'], 
            'precipitation': scenario['precip'],
            'obstacle_density': scenario['density'],
            'obstacle_height': 25.0
        }
        
        action = nav_system.get_navigation_action(sensor_data)
        print(f"{scenario['name']}:")
        print(f"Действие: {action.name}")
        print(f"Изменения: высота {action.altitude_change:+.1f}м, скорость {action.speed_change:+.1f}м/с")
        print()

if __name__ == "__main__":
    # Запуск демо и симуляции
    demo_system()
    
    print("\nЗАПУСК ПОЛНОЙ СИМУЛЯЦИИ")
    
    nav_system = FuzzyDroneNavigationSystem()
    simulator = DroneSimulator(nav_system)
    success = simulator.simulate(steps=100)
    
    if success:
        print("\nСИМУЛЯЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
    else:
        print("\nЦЕЛЬ НЕ ДОСТИГНУТА")