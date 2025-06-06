import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

SIZE_POPULATION = 6
FILTERS_OPTIONS = [16, 32, 64]
LR_RANGE = (1e-5, 1e-1)
NUM_GENERATIONS = 3
EPOCHS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# --------------------------
# Данные
# --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
val_data = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)

# --------------------------
# Инициализация
# --------------------------
def initialize_population():
    population = []
    for _ in range(SIZE_POPULATION):
        filters = random.choice(FILTERS_OPTIONS)
        lr = 10 ** np.random.uniform(np.log10(LR_RANGE[0]), np.log10(LR_RANGE[1]))
        individual = [filters, lr]
        population.append(individual)
    return population

# --------------------------
# Архитектура
# --------------------------
def build_cnn(filters):
    model = nn.Sequential(

        nn.Conv2d(1, filters, kernel_size=3, padding=1),
        nn.BatchNorm2d(filters),
        nn.ReLU(),
        nn.MaxPool2d(2),


        nn.Conv2d(filters, filters * 2, kernel_size=3, padding=1),
        nn.BatchNorm2d(filters * 2),
        nn.ReLU(),
        nn.MaxPool2d(2),


        nn.Flatten(),
        nn.Linear(filters * 2 * 7 * 7, 10)
    )
    return model.to(device)

# --------------------------
# Оценка индивида
# --------------------------
def evaluate(individual):
    filters, lr = individual
    model = build_cnn(filters)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Обучение
    model.train()
    for epoch in range(EPOCHS):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Валидация
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total

    # Чистка
    del model
    del optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return accuracy

# --------------------------
# Оценка
# --------------------------
def evaluate_population(population):
    fitnesses = []
    for i, individual in enumerate(population):
        print(f"Оценивается особь {i+1}/{len(population)}: {individual}")
        acc = evaluate(individual)
        print(f"Точность: {acc:.4f}")
        fitnesses.append(acc)
    return fitnesses

# --------------------------
# Турнир
# --------------------------
def tournament_selection(population, fitnesses, k=3):
    indices = np.random.choice(len(population), size=k)
    best_index = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[best_index]

# --------------------------
# Кроссовер
# --------------------------
def crossover(parent1, parent2):
    f1, lr1 = parent1
    f2, lr2 = parent2

    # Learning rate: арифметический кроссовер
    alpha = np.random.rand()
    child_lr = alpha * lr1 + (1 - alpha) * lr2

    # Filters: усреднение до ближайшего допустимого
    avg = (f1 + f2) / 2
    child_filters = min(FILTERS_OPTIONS, key=lambda x: abs(x - avg))

    return [child_filters, child_lr]

# --------------------------
# Мутации
# --------------------------
def mutate(individual, sigma=0.1, filter_prob=0.3):
    filters, lr = individual

    # Learning rate mutation
    lr *= 10 ** (np.random.randn() * sigma)
    lr = np.clip(lr, LR_RANGE[0], LR_RANGE[1])

    # Filter mutation
    if np.random.rand() < filter_prob:
        idx = FILTERS_OPTIONS.index(filters)
        idx += np.random.choice([-1, 1])
        idx = max(0, min(len(FILTERS_OPTIONS) - 1, idx))
        filters = FILTERS_OPTIONS[idx]

    return [filters, lr]

# --------------------------
# Создание нового поколения
# --------------------------
def create_new_population(population, fitnesses):
    new_population = []

    # лучший мальчик
    best_idx = np.argmax(fitnesses)
    new_population.append(population[best_idx])

    # Генерируем остальных
    while len(new_population) < SIZE_POPULATION:
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)

    return new_population

# --------------------------
# Основной циклочек
# --------------------------
population = initialize_population()

for generation in range(NUM_GENERATIONS):
    print(f"\n=== Поколение {generation + 1} ===")
    fitnesses = evaluate_population(population)
    best_fitness = max(fitnesses)
    best_individual = population[np.argmax(fitnesses)]
    print(f"Лучшая точность: {best_fitness:.4f}, параметры: {best_individual}")

    population = create_new_population(population, fitnesses)