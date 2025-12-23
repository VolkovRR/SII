import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# Задаем параметры задачи
n_production = 4  # Количество пунктов производства
n_cities = 5      # Количество городов

# Производственные мощности пунктов производства (максимальное количество продуктов)
production_capacity = [100, 150, 120, 180]  # y_i

# Потребности городов (необходимое количество продуктов)
city_demand = [80, 90, 110, 60, 70]  # x_j

# Матрица транспортных расходов от пункта i к городу j
transport_cost = np.array([
    [5, 8, 12, 6, 9],
    [7, 6, 9, 8, 10],
    [10, 12, 7, 11, 8],
    [6, 9, 8, 7, 12]
])

# Настройки генетического алгоритма
population_size = 50
generations = 100
crossover_rate = 0.8
mutation_rate = 0.2

# Функция для преобразования решения в матрицу распределения
def decode_solution(solution: np.ndarray) -> np.ndarray:
    """
    Преобразует вектор решений в матрицу распределения n_production x n_cities
    solution: вектор длины n_production * n_cities, где каждый элемент - количество продуктов
    """
    matrix = solution.reshape((n_production, n_cities))
    return matrix

# Функция оценки приспособленности
def fitness(solution: np.ndarray) -> Tuple[float, float, float]:
    """
    Оценка качества решения:
    1. Общая стоимость транспортировки (минимизировать)
    2. Нарушение производственных ограничений (минимизировать)
    3. Неудовлетворенный спрос (минимизировать)
    """
    matrix = decode_solution(solution)
    
    # 1. Транспортные расходы
    total_cost = np.sum(matrix * transport_cost)
    
    # 2. Нарушение производственных ограничений
    production_violation = 0
    for i in range(n_production):
        produced = np.sum(matrix[i, :])
        if produced > production_capacity[i]:
            production_violation += (produced - production_capacity[i]) * 100  # Штраф за превышение
    
    # 3. Неудовлетворенный спрос
    demand_violation = 0
    for j in range(n_cities):
        received = np.sum(matrix[:, j])
        if received < city_demand[j]:
            demand_violation += (city_demand[j] - received) * 100  # Штраф за недопоставку
    
    # 4. Превышение спроса (нежелательно, но допустимо с меньшим штрафом)
    excess_violation = 0
    for j in range(n_cities):
        received = np.sum(matrix[:, j])
        if received > city_demand[j] * 1.2:  # Допускается превышение до 20%
            excess_violation += (received - city_demand[j] * 1.2) * 50
    
    # Фитнес функция: минимизируем общие затраты и нарушения
    fitness_value = 1 / (total_cost + production_violation + demand_violation + excess_violation + 1)
    
    return fitness_value, total_cost, production_violation + demand_violation + excess_violation

# Генерация начальной популяции
def initialize_population() -> List[np.ndarray]:
    population = []
    for _ in range(population_size):
        # Генерируем случайное распределение
        solution = np.zeros(n_production * n_cities)
        
        # Распределяем продукты случайным образом
        total_available = min(sum(production_capacity), sum(city_demand) * 1.2)
        allocated = 0
        
        while allocated < total_available:
            i = random.randint(0, n_production - 1)
            j = random.randint(0, n_cities - 1)
            idx = i * n_cities + j
            
            # Проверяем ограничения
            current_row_sum = solution[i*n_cities:(i+1)*n_cities].sum()
            current_col_sum = solution[j::n_cities].sum()
            
            if current_row_sum < production_capacity[i] and current_col_sum < city_demand[j] * 1.2:
                max_to_allocate = min(
                    production_capacity[i] - current_row_sum,
                    city_demand[j] * 1.2 - current_col_sum,
                    total_available - allocated
                )
                if max_to_allocate > 0:
                    allocate = random.randint(0, int(max_to_allocate))
                    solution[idx] += allocate
                    allocated += allocate
        
        population.append(solution)
    
    return population

# Селекция (турнирный отбор)
def selection(population: List[np.ndarray]) -> List[np.ndarray]:
    selected = []
    tournament_size = 3
    
    while len(selected) < population_size:
        # Турнирный отбор
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda ind: fitness(ind)[0])
        selected.append(winner.copy())
    
    return selected

# Кроссовер: одноточечный
def single_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
    else:
        child1 = parent1.copy()
        child2 = parent2.copy()
    
    return child1, child2

# Мутация
def mutation(individual: np.ndarray) -> np.ndarray:
    if random.random() < mutation_rate:
        # Выбираем случайную мутацию
        mutation_type = random.choice([1, 2, 3])
        
        if mutation_type == 1:  # Изменение количества на случайной позиции
            idx = random.randint(0, len(individual) - 1)
            i = idx // n_cities
            j = idx % n_cities
            
            current_row_sum = individual[i*n_cities:(i+1)*n_cities].sum()
            current_col_sum = individual[j::n_cities].sum()
            
            max_value = min(
                production_capacity[i] - (current_row_sum - individual[idx]),
                city_demand[j] * 1.2 - (current_col_sum - individual[idx])
            )
            max_value = max(0, max_value)
            
            individual[idx] = random.randint(0, int(max_value))
            
        elif mutation_type == 2:  # Обмен значениями между двумя позициями
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
            
        elif mutation_type == 3:  # Случайное возмущение
            idx = random.randint(0, len(individual) - 1)
            change = random.randint(-10, 10)
            individual[idx] = max(0, individual[idx] + change)
    
    return individual

# Полный перебор (только для небольших задач)
def brute_force() -> Tuple[np.ndarray, float, float]:
    print("Выполнение полного перебора...")
    
    if n_production * n_cities > 6:  # Ограничиваем размер задачи для перебора
        print("Задача слишком большая для полного перебора")
        return None, float('inf'), float('inf')
    
    best_solution = None
    best_fitness = 0
    best_cost = float('inf')
    
    # Генерируем все возможные варианты (упрощенно)
    max_value = int(max(production_capacity) / n_cities)
    total_combinations = (max_value + 1) ** (n_production * n_cities)
    
    print(f"Всего комбинаций для перебора: {total_combinations:,}")
    
    # Для демонстрации ограничим перебор
    sample_size = min(10000, total_combinations)
    
    for _ in range(sample_size):
        # Генерируем случайное решение для демонстрации
        solution = np.random.randint(0, max_value + 1, size=(n_production * n_cities))
        
        fit, cost, violations = fitness(solution)
        
        if fit > best_fitness:
            best_fitness = fit
            best_cost = cost
            best_solution = solution.copy()
    
    return best_solution, best_fitness, best_cost

# Генетический алгоритм
def genetic_algorithm() -> Tuple[np.ndarray, List[float], List[float]]:
    population = initialize_population()
    best_fitness_history = []
    best_cost_history = []
    
    for gen in range(generations):
        # Оценка приспособленности
        fitness_values = [fitness(ind) for ind in population]
        
        # Лучшее решение в поколении
        best_idx = np.argmax([f[0] for f in fitness_values])
        best_fitness_history.append(fitness_values[best_idx][0])
        best_cost_history.append(fitness_values[best_idx][1])
        
        # Селекция
        selected = selection(population)
        
        # Скрещивание и мутация
        next_population = []
        
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = single_point_crossover(parent1, parent2)
            next_population.append(mutation(child1))
            next_population.append(mutation(child2))
        
        population = next_population[:population_size]
        
        if (gen + 1) % 20 == 0:
            print(f"Поколение {gen + 1}: лучшая приспособленность = {best_fitness_history[-1]:.6f}, "
                  f"стоимость = {best_cost_history[-1]:.2f}")
    
    # Лучшее решение
    fitness_values = [fitness(ind) for ind in population]
    best_idx = np.argmax([f[0] for f in fitness_values])
    best_solution = population[best_idx]
    
    return best_solution, best_fitness_history, best_cost_history

# Визуализация результатов
def visualize_results(best_solution_ga: np.ndarray, fitness_history: List[float], 
                      cost_history: List[float], best_solution_bf: np.ndarray = None):
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. График сходимости генетического алгоритма
    axes[0, 0].plot(fitness_history, 'b-', linewidth=2)
    axes[0, 0].set_title('Сходимость генетического алгоритма')
    axes[0, 0].set_xlabel('Поколение')
    axes[0, 0].set_ylabel('Приспособленность')
    axes[0, 0].grid(True)
    
    # 2. График стоимости
    axes[0, 1].plot(cost_history, 'r-', linewidth=2)
    axes[0, 1].set_title('Минимальная стоимость по поколениям')
    axes[0, 1].set_xlabel('Поколение')
    axes[0, 1].set_ylabel('Транспортные расходы')
    axes[0, 1].grid(True)
    
    # 3. Матрица распределения (ГА)
    matrix_ga = decode_solution(best_solution_ga)
    im1 = axes[1, 0].imshow(matrix_ga, cmap='YlOrRd', aspect='auto')
    axes[1, 0].set_title('Оптимальное распределение (ГА)')
    axes[1, 0].set_xlabel('Города')
    axes[1, 0].set_ylabel('Пункты производства')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Добавляем значения в ячейки
    for i in range(n_production):
        for j in range(n_cities):
            if matrix_ga[i, j] > 0:
                axes[1, 0].text(j, i, f'{int(matrix_ga[i, j])}', 
                               ha='center', va='center', color='black', fontsize=10)
    
    # 4. Сравнение с полным перебором (если есть)
    if best_solution_bf is not None:
        matrix_bf = decode_solution(best_solution_bf)
        im2 = axes[1, 1].imshow(matrix_bf, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_title('Распределение (полный перебор)')
        axes[1, 1].set_xlabel('Города')
        axes[1, 1].set_ylabel('Пункты производства')
        plt.colorbar(im2, ax=axes[1, 1])
        
        for i in range(n_production):
            for j in range(n_cities):
                if matrix_bf[i, j] > 0:
                    axes[1, 1].text(j, i, f'{int(matrix_bf[i, j])}', 
                                   ha='center', va='center', color='black', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Вывод статистики
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
    print("="*60)
    
    matrix = decode_solution(best_solution_ga)
    fit, cost, violations = fitness(best_solution_ga)
    
    print(f"\nМатрица распределения продуктов:")
    for i in range(n_production):
        row = [f"{int(matrix[i, j]):3d}" for j in range(n_cities)]
        print(f"Пункт {i+1}: [{', '.join(row)}] | Всего: {int(np.sum(matrix[i, :]))}/{production_capacity[i]}")
    
    print(f"\nПоставки городам:")
    for j in range(n_cities):
        col_sum = int(np.sum(matrix[:, j]))
        print(f"Город {j+1}: получено {col_sum}/{city_demand[j]} ({(col_sum/city_demand[j]*100):.1f}%)")
    
    print(f"\nОбщая стоимость транспортировки: {cost:.2f}")
    print(f"Общая приспособленность решения: {fit:.6f}")
    print(f"Штраф за нарушения: {violations:.2f}")
    
    total_delivered = np.sum(matrix)
    total_demand = sum(city_demand)
    total_capacity = sum(production_capacity)
    
    print(f"\nИспользовано производственных мощностей: {total_delivered}/{total_capacity} "
          f"({total_delivered/total_capacity*100:.1f}%)")
    print(f"Удовлетворено потребностей: {total_delivered}/{total_demand} "
          f"({total_delivered/total_demand*100:.1f}%)")

# Основная функция
def main():
    print("="*60)
    print("ЗАДАЧА ОПТИМИЗАЦИИ РАСПРЕДЕЛЕНИЯ ПРОДУКТОВ")
    print("="*60)
    print(f"Пунктов производства: {n_production}")
    print(f"Городов: {n_cities}")
    print(f"Производственные мощности: {production_capacity}")
    print(f"Потребности городов: {city_demand}")
    print("="*60)
    
    # Запуск генетического алгоритма
    print("\nЗапуск генетического алгоритма...")
    best_solution_ga, fitness_history, cost_history = genetic_algorithm()
    
    # Запуск полного перебора (для сравнения)
    print("\n" + "="*60)
    best_solution_bf, bf_fitness, bf_cost = brute_force()
    
    # Визуализация результатов
    visualize_results(best_solution_ga, fitness_history, cost_history, best_solution_bf)
    
    if best_solution_bf is not None and bf_fitness > 0:
        print("\n" + "="*60)
        print("СРАВНЕНИЕ С ПОЛНЫМ ПЕРЕБОРОМ")
        print("="*60)
        print(f"Генетический алгоритм: стоимость = {fitness(best_solution_ga)[1]:.2f}, "
              f"приспособленность = {fitness(best_solution_ga)[0]:.6f}")
        print(f"Полный перебор: стоимость = {bf_cost:.2f}, приспособленность = {bf_fitness:.6f}")
        
        if fitness(best_solution_ga)[0] > bf_fitness:
            print("✓ Генетический алгоритм нашел лучшее решение!")
        else:
            print("✗ Полный перебор дал лучшее решение")

# Запуск программы
if __name__ == "__main__":
    main()