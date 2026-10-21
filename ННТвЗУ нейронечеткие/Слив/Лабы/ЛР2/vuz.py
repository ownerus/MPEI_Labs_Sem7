import random
import sys
import matplotlib.pyplot as plt

# Параметры алгоритма
GENERATIONS_MAX_COUNT = 128  # Максимальное количество поколений
STOPPING_PATIENCE = 16       # Количество поколений без улучшений для остановки
STOPPING_MIN_DELTA = 0.     # Минимальное улучшение приспособленности
INITIAL_POPULATION_SIZE = 8  # Размер начальной популяции
CROSSOVER_PROB = 0.75       # Вероятность кроссинговера
MUTATION_PROB = 0.05        # Вероятность мутации
GENES_WEIGHTS = (1, 2, 3, 4, 5)  # Веса генов
CHROMA_SIZE = len(GENES_WEIGHTS) # Длина хромосомы
DISPLAY_LEARNING_HISTORY = True  # Флаг отображения графика

# Новые параметры для методов отбора
SELECTION_METHOD = "tournament"  # Варианты: "roulette", "elitist", "tournament"
TOURNAMENT_SIZE = 3            # Размер турнира для турнирного отбора
ELITISM_RATIO = 0.1           # Доля элитных особей для элитного отбора

def func(x):
    """Целевая функция для оптимизации."""
    return (2*x**3) + 125 * abs(2*x + 10)

def fitness(phenotype):
    """Функция приспособленности."""
    return func(phenotype)

def express_chroma(chromosome):
    """Преобразует хромосому в фенотип."""
    assert len(chromosome) == CHROMA_SIZE, "Длина хромосомы должна соответствовать количеству весов генов."
    return sum(gene * weight for gene, weight in zip(chromosome, GENES_WEIGHTS))

def create_individual():
    """Создает одну особь (хромосому)."""
    return [random.randrange(2) for _ in range(CHROMA_SIZE)]

def create_population(size):
    """Создает популяцию заданного размера."""
    return [create_individual() for _ in range(size)]

def selection_roulette(population, fitnesses):
    """Метод отбора: рулетка."""
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choices(population, k=len(population))

    probabilities = [fitness / total_fitness for fitness in fitnesses]
    selected_population = random.choices(population, weights=probabilities, k=len(population))
    return selected_population

def selection_tournament(population, fitnesses):
    """Метод отбора: турнирный."""
    new_population = []
    for _ in range(len(population)):
        # Случайно выбираем участников турнира
        tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        # Выбираем победителя
        winner_idx = tournament_indices[find_best_individual(tournament_fitnesses)]
        new_population.append(population[winner_idx][:])
    return new_population

def selection_elitist(population, fitnesses):
    """Метод отбора: элитный."""
    population_size = len(population)
    elite_size = int(population_size * ELITISM_RATIO)
    
    # Сортируем популяцию по приспособленности
    sorted_pairs = sorted(zip(fitnesses, population), reverse=True)
    sorted_fitnesses, sorted_population = zip(*sorted_pairs)
    
    # Сохраняем элитные особи
    new_population = list(sorted_population[:elite_size])
    
    # Заполняем оставшуюся часть с помощью рулеточного отбора
    remaining_size = population_size - elite_size
    remaining_population = selection_roulette(list(sorted_population[elite_size:]), 
                                           list(sorted_fitnesses[elite_size:]))
    new_population.extend(remaining_population[:remaining_size])
    
    return new_population

def select_population(population, fitnesses, method=SELECTION_METHOD):
    """Выбор метода отбора."""
    selection_methods = {
        "roulette": selection_roulette,
        "tournament": selection_tournament,
        "elitist": selection_elitist
    }
    
    if method not in selection_methods:
        raise ValueError(f"Неизвестный метод отбора: {method}")
    
    return selection_methods[method](population, fitnesses)

def crossover(parent1, parent2):
    """Кроссинговер с одной точкой."""
    point = random.randint(1, CHROMA_SIZE - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(chromosome):
    """Мутация хромосомы."""
    index = random.randrange(CHROMA_SIZE)
    chromosome[index] = 1 - chromosome[index]
    return chromosome

def apply_genetic_operators(population):
    """Применяет генетические операторы к популяции."""
    new_population = list(population)

    # Кроссинговер
    for i in range(0, len(population), 2):
        if random.random() < CROSSOVER_PROB:
            child1, child2 = crossover(population[i], population[i + 1])
            new_population[i] = child1
            if i + 1 < len(population):
                new_population[i + 1] = child2

    # Мутация
    for i in range(len(population)):
        if random.random() < MUTATION_PROB:
            new_population[i] = mutate(new_population[i])

    return new_population

def find_best_individual(population_fitnesses):
    """Находит индекс особи с наилучшей приспособленностью."""
    return max(range(len(population_fitnesses)), key=population_fitnesses.__getitem__)

def should_stop(population_fitnesses):
    """Проверяет условие остановки алгоритма."""
    quasiself = should_stop

    if not hasattr(quasiself, '_buffer'):
        quasiself._buffer = []

    best_fitness = max(population_fitnesses)
    quasiself._buffer.append(best_fitness)

    if len(quasiself._buffer) > STOPPING_PATIENCE:
        quasiself._buffer = quasiself._buffer[-STOPPING_PATIENCE:]

    if not ((quasiself._buffer[0] + STOPPING_MIN_DELTA) < quasiself._buffer[-1]):
        if len(quasiself._buffer) >= STOPPING_PATIENCE:
            return True
    else:
        quasiself._buffer = []

    return False

def main():
    """Основная функция генетического алгоритма."""
    population = create_population(INITIAL_POPULATION_SIZE)
    best_fitness_history = []
    median_fitness_history = []

    print(f"Используется метод отбора: {SELECTION_METHOD}")

    for generation in range(GENERATIONS_MAX_COUNT):
        phenotypes = [express_chroma(chromosome) for chromosome in population]
        fitnesses = [fitness(phenotype) for phenotype in phenotypes]

        best_individual_index = find_best_individual(fitnesses)
        best_fitness = fitnesses[best_individual_index]
        best_fitness_history.append(best_fitness)

        median_fitness = sorted(fitnesses)[len(fitnesses) // 2]
        median_fitness_history.append(median_fitness)

        print(f"Поколение {generation + 1}: Лучшая приспособленность = {best_fitness}, "
              f"Медианная приспособленность = {median_fitness}")

        if should_stop(best_fitness_history):
            print("Остановка: Нет значительного улучшения.")
            break

        population = select_population(population, fitnesses, SELECTION_METHOD)
        population = apply_genetic_operators(population)

    best_individual_index = find_best_individual(fitnesses)
    best_chromosome = population[best_individual_index]
    best_phenotype = express_chroma(best_chromosome)
    best_fitness = fitness(best_phenotype)

    print("Лучшая хромосома:", best_chromosome)
    print("Лучший фенотип:", best_phenotype)
    print("Лучшая приспособленность:", best_fitness)

    if DISPLAY_LEARNING_HISTORY:
        generations = range(1, len(best_fitness_history) + 1)
        plt.plot(generations, best_fitness_history, label='Лучшая приспособленность')
        plt.plot(generations, median_fitness_history, label='Медианная приспособленность')
        plt.xlabel('Поколение')
        plt.ylabel('Приспособленность')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    main()
