import os
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Константы ---
DATA_FILE           = "file.txt"
NUMERATOR_TEST      = "3"
DENOMINATOR_TEST    = "1,0.6,1.2"
STEP                = 0.01
# --- Функции ---

def read_data_from_file(file_path):
    """Читает данные из файла и возвращает словарь временных значений и значений функции,
    а также верхний и нижний пределы для времени из файла.
    """
    try:
        with open(file_path, "r") as file:
            data = file.readlines()
    except FileNotFoundError:
        messagebox.showerror("Ошибка", f"Файл '{file_path}' не найден.")
        return None, None
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при чтении файла: {e}")
        return None, None

    time_values = []
    function_values = []
    for line in data:
        try:
            time, value = map(float, line.strip().split())
            time_values.append(time)
            function_values.append(value)
        except ValueError:
            messagebox.showerror("Ошибка", f"Неверный формат данных в строке: '{line.strip()}'")
            return None, None

    # Получение нижнего предела времени из первой строки файла
    first_line = data[0].strip()
    try:
        t_min = float(first_line.split()[0])
    except ValueError:
        messagebox.showerror("Ошибка", f"Неверный формат данных в первой строке файла: '{last_line}'")
        return None, None

    # Получение верхнего предела времени из последней строки файла
    last_line = data[-1].strip()
    try:
        t_max = float(last_line.split()[0])
    except ValueError:
        messagebox.showerror("Ошибка", f"Неверный формат данных в последней строке файла: '{last_line}'")
        return None, None

    return dict(zip(time_values, function_values)), t_min, t_max

def calculate_and_plot(numerator_str, denominator_str, t_min, t_max):
    """Рассчитывает шаг системы и выводит график."""
    try:
        numerator = list(map(float, numerator_str.split(',')))
        denominator = list(map(float, denominator_str.split(',')))
    except ValueError:
        messagebox.showerror("Ошибка", "Пожалуйста, введите числитель и знаменатель в виде чисел, разделенных запятой.")
        return

    system = signal.TransferFunction(numerator, denominator)
    t = np.arange(t_min, t_max+STEP, STEP)
    t = [round(val, 10) for val in t]
    t, response = signal.step(system, T=t)
    analytic_method = dict(zip(tuple(t), tuple(response)))

    # --- Расчет MSE ---
    sim_values = list(simintech_data.values())
    sim_times = list(simintech_data.keys())
    mse = 0.0
    for i in range(len(sim_values)):
        mse += (sim_values[i] - analytic_method[sim_times[i]]) ** 2
    mse = (mse / i) ** 0.5
    # --- Отображение MSE ---
    mse_label.config(text=f"MSE: {mse}")

    # --- Построение графика ---
    plt.clf() # Очистка предыдущего графика
    plt.plot(list(simintech_data.keys()), list(simintech_data.values()), label="Решение SimInTech")
    plt.plot(t, response, label='Решение Python')
    plt.xlabel("Время")
    plt.ylabel("Значение функции")
    plt.title("Переходная функция")
    plt.legend()
    canvas.draw()

# --- Обработка событий ---

def on_execute():
    """Выполняет расчет и построение графика при нажатии на кнопку "Выполнить"."""
    numerator_str = numerator_entry.get()
    denominator_str = denominator_entry.get()
    calculate_and_plot(numerator_str, denominator_str, t_min, t_max)

def on_test():
    """Заполняет поля ввода тестовыми значениями и выполняет расчет."""
    numerator_entry.delete(0, tk.END)
    denominator_entry.delete(0, tk.END)
    numerator_entry.insert(0, NUMERATOR_TEST)
    denominator_entry.insert(0, DENOMINATOR_TEST)
    calculate_and_plot(NUMERATOR_TEST, DENOMINATOR_TEST, t_min, t_max)

# --- Главный код ---

# --- Чтение данных из файла ---
current_directory = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_directory, DATA_FILE)
simintech_data, t_min, t_max = read_data_from_file(data_file_path)
if simintech_data is None:
    exit() # Выход из программы, если чтение данных не удалось

# --- Создание графического интерфейса ---
root = tk.Tk()
root.title("Введите коэффициенты числителя и знаменателя через запятую по убыванию степени")

# --- Поля ввода ---
tk.Label(root, text="Числитель:").pack()
numerator_entry = tk.Entry(root)
numerator_entry.pack()

tk.Label(root, text="Знаменатель:").pack()
denominator_entry = tk.Entry(root)
denominator_entry.pack()

# --- Кнопки ---
execute_button = tk.Button(root, text="Выполнить", command=on_execute)
execute_button.pack()

test_button = tk.Button(root, text="Тест", command=on_test)
test_button.pack()

# --- Метка для вывода MSE ---
mse_label = tk.Label(root, text="MSE: ")
mse_label.pack()

# --- График ---
figure = plt.figure()
canvas = FigureCanvasTkAgg(figure, root)
canvas.get_tk_widget().pack()

# --- Запуск основного цикла Tkinter ---
root.mainloop()
