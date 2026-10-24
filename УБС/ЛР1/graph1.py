"""
Лабораторная работа №1 по дисциплине «Управление большими системами» (УБС).
Тема: Структурный анализ больших систем управления.
Бригада №3 (Вариант 3): Грудинин Е.К., Турханов А.К.

Функционал:
1. Проверка на контуры и разбиение графа по уровням (N_0 ... N_m).
2. Сквозная топологическая перенумерация (приведение матрицы к треугольному виду).
3. Пошаговый расчет минимального пути методом динамического программирования (уравнение Беллмана).
4. Восстановление всех оптимальных маршрутов и расчет критического пути.
"""


def get_variant_3_matrix():
    """Формирует матрицу смежности и цен графа для Варианта 3 (17 вершин)."""
    n = 17
    matrix = [[0] * n for _ in range(n)]

    # Базовые дуги из методички: (откуда, куда, цена) в нумерации 1..17
    base_edges = [
        (1, 9, 2),  (2, 17, 2), (3, 5, 4),  (3, 7, 5),  (3, 11, 3),
        (4, 9, 2),  (5, 4, 3),  (5, 6, 4),  (6, 12, 3), (7, 1, 2),
        (7, 4, 2),  (8, 14, 2), (9, 16, 3), (10, 15, 1), (11, 1, 3),
        (11, 2, 4), (12, 16, 2),(13, 15, 2),(14, 15, 2), (16, 10, 3),
        (16, 13, 3),(17, 8, 2),
        # Индивидуальные дуги Варианта 3:
        (11, 4, 5), (6, 17, 5)
    ]

    for u, v, w in base_edges:
        matrix[u - 1][v - 1] = w

    return n, matrix


def compute_levels(n, matrix):
    """
    Разбивает вершины графа на непересекающиеся подмножества (уровни N_0 ... N_m).
    Возвращает:
      - levels: список уровней со списками вершин;
      - old_to_new, new_to_old: словари соответствия исходной и новой нумерации.
    """
    # Множество непосредственных предшественников (левых инциденций G^-1)
    in_edges = {v: set() for v in range(1, n + 1)}
    for i in range(n):
        for j in range(n):
            if matrix[i][j] > 0:
                in_edges[j + 1].add(i + 1)

    levels = []
    assigned = set()

    while len(assigned) < n:
        # Вершина входит в текущий уровень, если все её предки уже на прошлых уровнях
        current_level = [
            v for v in range(1, n + 1)
            if v not in assigned and in_edges[v].issubset(assigned)
        ]

        if not current_level:
            unassigned = [v for v in range(1, n + 1) if v not in assigned]
            raise ValueError(f"В графе обнаружен контур (цикл)! Вершины: {unassigned}")

        current_level.sort()
        levels.append(current_level)
        assigned.update(current_level)

    # Сквозная топологическая нумерация (для приведения к треугольному виду)
    old_to_new = {}
    new_to_old = {}
    new_num = 1
    for lvl in levels:
        for old_v in lvl:
            old_to_new[old_v] = new_num
            new_to_old[new_num] = old_v
            new_num += 1

    return levels, old_to_new, new_to_old


def bellman_solve(n, matrix, new_to_old, old_to_new, start=3, target=15, find_max=False):
    """
    Решает задачу поиска экстремального (минимального/максимального) пути
    методом динамического программирования (уравнение Беллмана).
    """
    start_new = old_to_new[start]
    target_new = old_to_new[target]

    opt_val = {i: (float('-inf') if find_max else float('inf')) for i in range(1, n + 1)}
    parents = {i: [] for i in range(1, n + 1)}
    opt_val[start_new] = 0

    log_steps = []
    picker = max if find_max else min
    func_name = "max" if find_max else "min"

    # Движение в топологическом порядке новых номеров (1 .. n)
    for new_j in range(start_new, target_new + 1):
        old_j = new_to_old[new_j]
        incoming = []

        for new_i in range(start_new, new_j):
            old_i = new_to_old[new_i]
            w = matrix[old_i - 1][old_j - 1]
            prev_f = opt_val[new_i]
            if w > 0 and prev_f not in (float('inf'), float('-inf')):
                incoming.append((new_i, old_i, w, prev_f + w))

        if incoming:
            best_cost = picker(item[3] for item in incoming)
            opt_val[new_j] = best_cost
            parents[new_j] = [item[0] for item in incoming if item[3] == best_cost]

            terms = [f"F({old_i})+q({old_i},{old_j})={prev_f}+{w}={cand}" 
                     for _, old_i, w, cand in incoming]
            log_steps.append(
                f"Вершина #{new_j} (исх. {old_j}): F({old_j}) = {func_name}({', '.join(terms)}) = {best_cost}"
            )
        elif new_j == start_new:
            log_steps.append(f"Вершина #{new_j} (исх. {old_j}): Исток, F({old_j}) = 0")

    # Восстановление маршрутов (backtracking)
    paths_new = []

    def backtrack(curr, current_path):
        if curr == start_new:
            paths_new.append(list(reversed(current_path)))
            return
        for p in parents[curr]:
            backtrack(p, current_path + [p])

    backtrack(target_new, [target_new])
    paths_old = [[new_to_old[v] for v in p] for p in paths_new]

    return opt_val[target_new], paths_old, log_steps


def print_table(n, matrix, title="Матрица смежности"):
    """Печатает матрицу цен в аккуратном табличном виде."""
    print(f"\n--- {title} ---")
    header = "  " + "".join([f"{j:>3}" for j in range(1, n + 1)])
    print(header)
    print("  " + "---" * n)
    for i in range(n):
        row_str = f"{i + 1:>2}|" + "".join([f"{matrix[i][j]:>3}" if matrix[i][j] > 0 else "  ." for j in range(n)])
        print(row_str)


# =========================================================================
# ТОЧКА ВХОДА (ВЫПОЛНЕНИЕ ЛАБОРАТОРНОЙ РАБОТЫ)
# =========================================================================
if __name__ == "__main__":
    n, matrix = get_variant_3_matrix()

    print("=" * 75)
    print("ЛАБОРАТОРНАЯ РАБОТА №1 (УБС) — ВАРИАНТ №3 (БРИГАДА №3)")
    print("=" * 75)

    # 1. Вывод исходной матрицы
    print_table(n, matrix, "Исходная матрица смежности и цен дуг (Вариант 3)")

    # 2. Поуровневое упорядочивание
    levels, old_to_new, new_to_old = compute_levels(n, matrix)

    print("\n--- 1. РАСПРЕДЕЛЕНИЕ ВЕРШИН ПО УРОВНЯМ (ЯРУСАМ) ---")
    for idx, lvl in enumerate(levels):
        new_nums = [f"#{old_to_new[v]}" for v in lvl]
        print(f"  Уровень N_{idx}: исходные номера {lvl:<18} -> новые номера {new_nums}")

    # 3. Расчет минимального пути методом Беллмана
    start_node = 3
    target_node = 15

    print(f"\n--- 2. ПОШАГОВЫЙ РАСЧЕТ МЕТОДОМ БЕЛЛМАНА (ПУТЬ {start_node} -> {target_node}) ---")
    min_len, min_paths, min_log = bellman_solve(
        n, matrix, new_to_old, old_to_new, start=start_node, target=target_node, find_max=False
    )
    for line in min_log:
        print(" ", line)

    print(f"\nРЕЗУЛЬТАТ: Минимальное время подготовки (длина пути) L_min = {min_len}")
    print(f"Количество оптимальных маршрутов: {len(min_paths)}")
    for idx, p in enumerate(min_paths, 1):
        print(f"  Маршрут №{idx}: {' -> '.join(map(str, p))}")

    # 4. Расчет критического (максимального) пути
    max_len, max_paths, _ = bellman_solve(
        n, matrix, new_to_old, old_to_new, start=start_node, target=target_node, find_max=True
    )
    print(f"\n--- 3. КРИТИЧЕСКИЙ (МАКСИМАЛЬНЫЙ) ПУТЬ ---")
    print(f"Длина критического пути L_max = {max_len}")
    for idx, p in enumerate(max_paths, 1):
        print(f"  Критический маршрут №{idx}: {' -> '.join(map(str, p))}")
    print("=" * 75)