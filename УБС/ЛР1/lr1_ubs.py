"""
Лабораторная работа №1 по дисциплине УБС.
Тема: Структурный анализ больших систем управления.
Бригада: Грудинин, Турханов А.К. (Вариант 3)
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor


def get_variant_3():
    n = 17
    mat = [[0] * n for _ in range(n)]
    # Базовые дуги (1-based: u, v, w)
    base_edges = [
        (1, 9, 2), (2, 17, 2), (3, 5, 4), (3, 7, 5), (3, 11, 3),
        (4, 9, 2), (5, 4, 3), (5, 6, 4), (6, 12, 3), (7, 1, 2),
        (7, 4, 2), (8, 14, 2), (9, 16, 3), (10, 15, 1), (11, 1, 3),
        (11, 2, 4), (12, 16, 2), (13, 15, 2), (14, 15, 2), (16, 10, 3),
        (16, 13, 3), (17, 8, 2),
        # Дополнительные дуги Варианта 3
        (11, 4, 5), (6, 17, 5)
    ]
    for u, v, w in base_edges:
        mat[u - 1][v - 1] = w
    return n, mat


def compute_levels(n, mat):
    in_edges = {v: set() for v in range(1, n + 1)}
    for i in range(n):
        for j in range(n):
            if mat[i][j] > 0:
                in_edges[j + 1].add(i + 1)

    levels = []
    assigned = set()

    while len(assigned) < n:
        current_level = []
        for v in range(1, n + 1):
            if v not in assigned and in_edges[v].issubset(assigned):
                current_level.append(v)

        if not current_level:
            unassigned = [v for v in range(1, n + 1) if v not in assigned]
            raise ValueError(f"В графе обнаружен контур (цикл)! Невозможно упорядочить вершины: {unassigned}")

        current_level.sort()
        levels.append(current_level)
        assigned.update(current_level)

    # Сквозная нумерация
    old_to_new = {}
    new_to_old = {}
    new_idx = 1
    for lvl in levels:
        for old_v in lvl:
            old_to_new[old_v] = new_idx
            new_to_old[new_idx] = old_v
            new_idx += 1

    # Построение упорядоченной матрицы
    ordered_mat = [[0] * n for _ in range(n)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            ordered_mat[i - 1][j - 1] = mat[new_to_old[i] - 1][new_to_old[j] - 1]

    return levels, old_to_new, new_to_old, ordered_mat


def compute_bellman_path(n, ordered_mat, new_to_old, old_to_new, start_old=3, end_old=15):
    start_new = old_to_new.get(start_old, 1)
    end_new = old_to_new.get(end_old, n)

    f_min = {i: float('inf') for i in range(1, n + 1)}
    parent = {i: [] for i in range(1, n + 1)}
    f_min[start_new] = 0

    steps_log = []

    for j in range(start_new, n + 1):
        incoming = []
        for i in range(1, j):
            w = ordered_mat[i - 1][j - 1]
            if w > 0 and f_min[i] != float('inf'):
                incoming.append((i, w, f_min[i] + w))

        if incoming:
            best_val = min(item[2] for item in incoming)
            f_min[j] = best_val
            parent[j] = [item[0] for item in incoming if item[2] == best_val]

            old_j = new_to_old[j]
            variants_str = ", ".join([f"q({new_to_old[i]},{old_j})+F({new_to_old[i]})={w}+{f_min[i]}={cand}" for i, w, cand in incoming])
            steps_log.append(f"Вершина #{j} (исх. {old_j}): min({variants_str}) = {best_val}")
        elif j == start_new:
            steps_log.append(f"Вершина #{j} (исх. {new_to_old[j]}): источник, F({new_to_old[j]}) = 0")

    if f_min[end_new] == float('inf'):
        return None, [], [], steps_log

    paths_new = []
    def backtrack(curr, path):
        if curr == start_new:
            paths_new.append(list(reversed(path)))
            return
        for p in parent[curr]:
            backtrack(p, path + [p])

    backtrack(end_new, [end_new])
    paths_old = [[new_to_old[v] for v in p] for p in paths_new]

    return f_min[end_new], paths_old, paths_new, steps_log


class UBSApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("УБС — Лабораторная работа №1 (Вариант 3)")
        self.resize(1000, 720)

        self.n_vertices = 17
        self.matrix = [[0] * self.n_vertices for _ in range(self.n_vertices)]

        self._init_ui()
        self.load_variant_3()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Панель управления
        toolbar = QHBoxLayout()
        btn_v3 = QPushButton("Загрузить вариант 3")
        btn_v3.clicked.connect(self.load_variant_3)
        toolbar.addWidget(btn_v3)

        btn_clear = QPushButton("Очистить")
        btn_clear.clicked.connect(self.clear_matrix)
        toolbar.addWidget(btn_clear)

        toolbar.addSpacing(16)
        toolbar.addWidget(QLabel("Число вершин:"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(2, 35)
        self.spin_n.setValue(self.n_vertices)
        toolbar.addWidget(self.spin_n)

        btn_size = QPushButton("Применить размер")
        btn_size.clicked.connect(self.apply_size)
        toolbar.addWidget(btn_size)

        toolbar.addSpacing(16)
        btn_run = QPushButton("Выполнить расчет")
        btn_run.setStyleSheet("font-weight: bold; background-color: #E3F2FD;")
        btn_run.clicked.connect(self.run_analysis)
        toolbar.addWidget(btn_run)

        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # Вкладки
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Вкладка 1: Исходная матрица
        tab_mat = QWidget()
        lay_mat = QVBoxLayout(tab_mat)
        lay_mat.addWidget(QLabel("Матрица смежности и цен графа (0 — отсутствие дуги, число > 0 — вес дуги):"))
        self.tbl_input = QTableWidget()
        lay_mat.addWidget(self.tbl_input)
        self.tabs.addTab(tab_mat, "1. Исходная матрица цен")

        # Вкладка 2: Упорядочение
        tab_lev = QWidget()
        lay_lev = QVBoxLayout(tab_lev)
        lay_lev.addWidget(QLabel("Разбиение на уровни N_k и сквозная топологическая перенумерация:"))
        self.txt_levels = QTextEdit()
        self.txt_levels.setReadOnly(True)
        self.txt_levels.setFixedHeight(140)
        self.txt_levels.setFont(QFont("Consolas", 10))
        lay_lev.addWidget(self.txt_levels)

        lay_lev.addWidget(QLabel("Упорядоченная верхнетреугольная матрица смежности/цен (a_ij = 0 при i >= j):"))
        self.tbl_ordered = QTableWidget()
        lay_lev.addWidget(self.tbl_ordered)
        self.tabs.addTab(tab_lev, "2. Упорядочение по уровням")

        # Вкладка 3: Минимальный путь
        tab_path = QWidget()
        lay_path = QVBoxLayout(tab_path)
        path_params = QHBoxLayout()
        path_params.addWidget(QLabel("Начальная вершина (источник):"))
        self.spin_src = QSpinBox()
        self.spin_src.setRange(1, 35)
        self.spin_src.setValue(3)
        path_params.addWidget(self.spin_src)

        path_params.addSpacing(20)
        path_params.addWidget(QLabel("Конечная вершина (сток):"))
        self.spin_dst = QSpinBox()
        self.spin_dst.setRange(1, 35)
        self.spin_dst.setValue(15)
        path_params.addWidget(self.spin_dst)
        path_params.addStretch()
        lay_path.addLayout(path_params)

        lay_path.addWidget(QLabel("Протокол расчета методом динамического программирования (уравнение Беллмана):"))
        self.txt_path = QTextEdit()
        self.txt_path.setReadOnly(True)
        self.txt_path.setFont(QFont("Consolas", 10))
        lay_path.addWidget(self.txt_path)
        self.tabs.addTab(tab_path, "3. Минимальный путь (ДП)")

        self.statusBar().showMessage("Готов к работе")

    def load_variant_3(self):
        self.n_vertices, self.matrix = get_variant_3()
        self.spin_n.setValue(self.n_vertices)
        self.spin_src.setValue(3)
        self.spin_dst.setValue(15)
        self._populate_input_table()
        self.statusBar().showMessage("Загружен Вариант 3: 17 вершин, 24 дуги")

    def clear_matrix(self):
        self.matrix = [[0] * self.n_vertices for _ in range(self.n_vertices)]
        self._populate_input_table()
        self.statusBar().showMessage("Матрица очищена")

    def apply_size(self):
        new_n = self.spin_n.value()
        new_mat = [[0] * new_n for _ in range(new_n)]
        min_dim = min(self.n_vertices, new_n)
        for i in range(min_dim):
            for j in range(min_dim):
                new_mat[i][j] = self.matrix[i][j]
        self.n_vertices = new_n
        self.matrix = new_mat
        self.spin_src.setMaximum(new_n)
        self.spin_dst.setMaximum(new_n)
        self._populate_input_table()
        self.statusBar().showMessage(f"Размер матрицы изменен на {new_n}x{new_n}")

    def _populate_input_table(self):
        n = self.n_vertices
        self.tbl_input.setRowCount(n)
        self.tbl_input.setColumnCount(n)
        labels = [str(i + 1) for i in range(n)]
        self.tbl_input.setHorizontalHeaderLabels(labels)
        self.tbl_input.setVerticalHeaderLabels(labels)

        for i in range(n):
            self.tbl_input.setRowHeight(i, 28)
            self.tbl_input.setColumnWidth(i, 36)
            for j in range(n):
                val = self.matrix[i][j]
                item = QTableWidgetItem(str(val) if val > 0 else "")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if i == j:
                    item.setBackground(QColor("#ECEFF1"))
                self.tbl_input.setItem(i, j, item)

    def _read_input_table(self):
        n = self.n_vertices
        for i in range(n):
            for j in range(n):
                item = self.tbl_input.item(i, j)
                txt = item.text().strip() if item else ""
                if not txt or txt == "0":
                    self.matrix[i][j] = 0
                else:
                    try:
                        v = int(txt)
                        if v < 0:
                            raise ValueError
                        self.matrix[i][j] = v
                    except ValueError:
                        QMessageBox.warning(self, "Ошибка ввода", f"Некорректный вес в ячейке ({i+1}, {j+1}): '{txt}'. Ожидается неотрицательное целое число.")
                        return False
        return True

    def run_analysis(self):
        if not self._read_input_table():
            return

        try:
            levels, old_to_new, new_to_old, ordered_mat = compute_levels(self.n_vertices, self.matrix)
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка упорядочения графа", str(e))
            self.statusBar().showMessage("Ошибка: в графе обнаружен контур")
            return

        # Заполнение вкладки уровней
        self.txt_levels.clear()
        self.txt_levels.append("Распределение вершин по уровням порядковой функции:")
        for idx, lvl in enumerate(levels):
            old_str = ", ".join(map(str, lvl))
            new_str = ", ".join([f"#{old_to_new[v]}" for v in lvl])
            self.txt_levels.append(f"  Уровень N_{idx}: исх. {{{old_str}}}  -->  новые номера {{{new_str}}}")

        mapping_str = ", ".join([f"#{i}(исх.{new_to_old[i]})" for i in range(1, self.n_vertices + 1)])
        self.txt_levels.append(f"\nСквозное соответствие нумерации:\n  {mapping_str}")

        # Отображение упорядоченной матрицы
        n = self.n_vertices
        self.tbl_ordered.setRowCount(n)
        self.tbl_ordered.setColumnCount(n)
        h_labels = [f"#{j}" for j in range(1, n + 1)]
        v_labels = [f"#{i} ({new_to_old[i]})" for i in range(1, n + 1)]
        self.tbl_ordered.setHorizontalHeaderLabels(h_labels)
        self.tbl_ordered.setVerticalHeaderLabels(v_labels)

        for i in range(n):
            self.tbl_ordered.setRowHeight(i, 28)
            self.tbl_ordered.setColumnWidth(i, 40)
            for j in range(n):
                w = ordered_mat[i][j]
                item = QTableWidgetItem(str(w) if w > 0 else "0" if i == j else "")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if w > 0:
                    item.setBackground(QColor("#FFF9C4"))
                elif i >= j:
                    item.setBackground(QColor("#F5F5F5"))
                self.tbl_ordered.setItem(i, j, item)

        # Расчет минимального пути
        s = self.spin_src.value()
        t = self.spin_dst.value()
        min_cost, paths_old, paths_new, log = compute_bellman_path(n, ordered_mat, new_to_old, old_to_new, s, t)

        self.txt_path.clear()
        self.txt_path.append("ПОШАГОВЫЙ РАСЧЕТ МЕТОДОМ ДИНАМИЧЕСКОГО ПРОГРАММИРОВАНИЯ (БЕЛЛМАН):")
        self.txt_path.append("=" * 80)
        for line in log:
            self.txt_path.append(line)
        self.txt_path.append("=" * 80)

        if min_cost is None:
            self.txt_path.append(f"Маршрут из вершины {s} в вершину {t} отсутствует.")
            self.statusBar().showMessage("Расчет выполнен: путь не существует")
        else:
            self.txt_path.append(f"РЕЗУЛЬТАТ: Минимальная длина пути L_min = {min_cost}\n")
            self.txt_path.append(f"Количество оптимальных маршрутов: {len(paths_old)}")
            for idx, (p_old, p_new) in enumerate(zip(paths_old, paths_new), start=1):
                s_old = " -> ".join(map(str, p_old))
                s_new = " -> ".join([f"#{v}" for v in p_new])
                self.txt_path.append(f"  Маршрут №{idx}:")
                self.txt_path.append(f"    В исходной нумерации: {s_old}")
                self.txt_path.append(f"    В новой нумерации:    {s_new}\n")
            self.statusBar().showMessage(f"Расчет успешно выполнен: L_min = {min_cost}, найдено путей: {len(paths_old)}")

        self.tabs.setCurrentIndex(1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UBSApp()
    window.show()
    sys.exit(app.exec())
