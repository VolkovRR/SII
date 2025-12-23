import numpy as np
import matplotlib.pyplot as plt

class TrapezoidalFuzzySet:
    def __init__(self, name, a, b, c, d):
        """
        Инициализация трапециевидного нечеткого множества.
        
        Параметры:
        name - название множества
        a - левая нижняя точка (начало подъема)
        b - левая верхняя точка (конец подъема)
        c - правая верхняя точка (начал спада)
        d - правая нижняя точка (конец спада)
        """
        self.name = name
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        # Проверка корректности параметров
        if not (a <= b <= c <= d):
            raise ValueError("Должно выполняться условие: a <= b <= c <= d")

    def membership_degree(self, x):
        """Вычисление степени принадлежности для трапециевидной функции."""
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.a < x < self.b:
            return (x - self.a) / (self.b - self.a)
        elif self.b <= x <= self.c:
            return 1.0
        elif self.c < x < self.d:
            return (self.d - x) / (self.d - self.c)
        else:
            return 0.0

    def display_parameters(self):
        """Отображение параметров нечеткого множества."""
        print(f"\nПараметры нечеткого множества '{self.name}':")
        print(f"  a (левая нижняя точка) = {self.a}")
        print(f"  b (левая верхняя точка) = {self.b}")
        print(f"  c (правая верхняя точка) = {self.c}")
        print(f"  d (правая нижняя точка) = {self.d}")
        print(f"  Носитель: [{self.a}, {self.d}]")
        print(f"  Ядро: [{self.b}, {self.c}]")

    def plot(self, x_min=None, x_max=None, num_points=1000):
        """Построение графика трапециевидной функции принадлежности."""
        if x_min is None:
            x_min = self.a - (self.d - self.a) * 0.1
        if x_max is None:
            x_max = self.d + (self.d - self.a) * 0.1
            
        x_values = np.linspace(x_min, x_max, num_points)
        y_values = [self.membership_degree(x) for x in x_values]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, 'b-', linewidth=2, label=self.name)
        plt.fill_between(x_values, y_values, alpha=0.2)
        
        # Добавление вертикальных линий для параметров
        plt.axvline(x=self.a, color='r', linestyle='--', alpha=0.7, label=f'a = {self.a}')
        plt.axvline(x=self.b, color='g', linestyle='--', alpha=0.7, label=f'b = {self.b}')
        plt.axvline(x=self.c, color='g', linestyle='--', alpha=0.7, label=f'c = {self.c}')
        plt.axvline(x=self.d, color='r', linestyle='--', alpha=0.7, label=f'd = {self.d}')
        
        plt.title(f'Трапециевидная функция принадлежности: {self.name}', fontsize=14)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('Степень принадлежности μ(x)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.show()

def create_fuzzy_set_interactively():
    """Интерактивное создание нечеткого множества."""
    print("=== Создание трапециевидного нечеткого множества ===")
    
    name = input("Введите название множества: ").strip()
    if not name:
        name = "Нечеткое множество"
    
    print("\nВведите параметры трапециевидной функции (должно выполняться: a <= b <= c <= d):")
    
    while True:
        try:
            a = float(input("a (левая нижняя точка, начало подъема): "))
            b = float(input("b (левая верхняя точка, конец подъема): "))
            c = float(input("c (правая верхняя точка, начало спада): "))
            d = float(input("d (правая нижняя точка, конец спада): "))
            
            # Проверка корректности параметров
            if a <= b <= c <= d:
                return TrapezoidalFuzzySet(name, a, b, c, d)
            else:
                print("Ошибка: Должно выполняться условие a <= b <= c <= d. Попробуйте снова.")
                
        except ValueError:
            print("Ошибка: Введите числовые значения.")
        except KeyboardInterrupt:
            print("\nПрограмма прервана пользователем.")
            return None

def calculate_membership_interactively(fuzzy_set):
    """Интерактивный расчет степени принадлежности."""
    print(f"\n=== Расчет степени принадлежности для '{fuzzy_set.name}' ===")
    
    while True:
        try:
            user_input = input("\nВведите значение x (или 'quit' для выхода): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            x = float(user_input)
            degree = fuzzy_set.membership_degree(x)
            
            print(f"Степень принадлежности μ({x}) = {degree:.4f}")
            
            # Дополнительная информация о положении точки
            if degree == 0.0:
                print(f"Точка {x} находится вне носителя множества")
            elif degree == 1.0:
                print(f"Точка {x} находится в ядре множества")
            elif 0 < degree < 1.0:
                print(f"Точка {x} находится в области нечеткости")
                
        except ValueError:
            print("Ошибка: Введите числовое значение.")
        except KeyboardInterrupt:
            print("\nРасчет прерван.")
            break

def main():
    """Основная функция программы."""
    try:
        # Создание нечеткого множества
        fuzzy_set = create_fuzzy_set_interactively()
        if fuzzy_set is None:
            return
        
        # Отображение параметров
        fuzzy_set.display_parameters()
        
        # Построение графика
        plot_choice = input("\nПостроить график функции принадлежности? (y/n): ").strip().lower()
        if plot_choice in ['y', 'yes', 'да']:
            try:
                x_min = input("Введите минимальное значение x (Enter для авто): ").strip()
                x_max = input("Введите максимальное значение x (Enter для авто): ").strip()
                
                x_min = float(x_min) if x_min else None
                x_max = float(x_max) if x_max else None
                
                fuzzy_set.plot(x_min, x_max)
            except ValueError:
                print("Используются значения по умолчанию для графика.")
                fuzzy_set.plot()
        
        # Расчет степени принадлежности
        calc_choice = input("\nВыполнить расчет степени принадлежности? (y/n): ").strip().lower()
        if calc_choice in ['y', 'yes', 'да']:
            calculate_membership_interactively(fuzzy_set)
            
        print("\nПрограмма завершена.")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")

# Пример использования с заранее заданными параметрами
def example_usage():
    """Пример использования класса с заранее заданными параметрами."""
    print("=== Пример использования ===")
    
    # Создание примера нечеткого множества "Средняя температура"
    example_set = TrapezoidalFuzzySet("Средняя температура", 15, 20, 25, 30)
    
    # Отображение параметров
    example_set.display_parameters()
    
    # Расчет для нескольких значений
    test_values = [10, 18, 22, 27, 32]
    print(f"\nРасчет степени принадлежности для тестовых значений:")
    for x in test_values:
        degree = example_set.membership_degree(x)
        print(f"μ({x}) = {degree:.2f}")
    
    # Построение графика
    example_set.plot(10, 35)

if __name__ == "__main__":
    # Запуск примера или основной программы
    choice = input("Запустить пример (1) или интерактивный режим (2)? ").strip()
    
    if choice == "1":
        example_usage()
    else:
        main()