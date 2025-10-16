import numpy as np

# Função de ativação sigmoid (converte valores em faixa 0–1)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da sigmoid (usada para atualizar os pesos)
def sigmoid_derivative(x):
    return x * (1 - x)

# --- Dados de treino ---
# Cada linha: [número de pessoas, dias de consumo]
# Saída: consumo médio de água (normalizado entre 0 e 1)
entradas = np.array([
    [1, 10],   # casa pequena, 1 pessoa, 10 dias
    [2, 10],   # casa média
    [3, 15],   # casa maior
    [4, 20],   # casa grande
    [5, 25]    # família grande
])

# Consumo estimado em metros cúbicos (normalizado)
# Exemplo: 5, 10, 15, 20, 25 m³ -> valores normalizados 0.2, 0.4, 0.6, 0.8, 1.0
saidas = np.array([[0.2], [0.4], [0.6], [0.8], [1.0]])

# Inicializar pesos aleatórios
np.random.seed(42)
pesos = 2 * np.random.random((2, 1)) - 1

# Taxa de aprendizagem
alpha = 0.2

# --- Treinar a rede ---
for i in range(10000):
    # Passo 1: propagação direta
    soma = np.dot(entradas, pesos)
    saida = sigmoid(soma)

    # Passo 2: calcular erro
    erro = saidas - saida

    # Passo 3: ajustar pesos (retropropagação)
    ajustes = erro * sigmoid_derivative(saida)
    pesos += np.dot(entradas.T, ajustes) * alpha

# --- Resultados ---
print("Pesos finais da rede neural:")
print(pesos)

print("\nPrevisoes apos treino:")
print(saida.round(3))

# --- Testar com novos dados ---
# Exemplo: casa com 3 pessoas e 20 dias de consumo
nova_casa = np.array([3, 20])
resultado = sigmoid(np.dot(nova_casa, pesos))

# Converter valor normalizado para metros cúbicos (escala de 0 a 25 m³)
consumo_estimado = resultado[0] * 25
print(f"\nPrevisão para uma casa com 3 pessoas e 20 dias:")
print(f"Consumo estimado: {consumo_estimado:.2f} m³")

