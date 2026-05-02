# 🧠 Descrição e Configuração do Ambiente

Este projeto é uma avaliação da disciplina de Inteligência Artificial Computacional. O objetivo foi implementar do absoluto zero (utilizando apenas a matemática vetorial do **NumPy**) algoritmos clássicos de Machine Learning para tarefas de **Classificação** e **Regressão**.

---

## 📂 Estrutura do Projeto

O projeto foi modularizado utilizando as melhores práticas de Clean Code:

| Arquivo / Pasta | Descrição |
|---|---|
| `main.py` | Script principal que orquestra a execução, carrega os dados, treina os modelos usando Validação Cruzada (K-Fold) e exibe as métricas e tempos de execução. |
| `data/` | Pasta contendo os bancos de dados nos formatos originais (ARFF). |
| `data/dataset_ANIME` | Dataset para Regressão. |
| `data/dataset_STUDENTS_DROPOUT_AND_ACADEMIC_SUCCESS` | Dataset para Classificação. |
| `models/knn.py` | K-Nearest Neighbors (Classificador e Regressor com distâncias Euclidiana e Manhattan). |
| `models/naive_bayes.py` | Classificador Bayesiano Univariado. |
| `models/multivariate_bayes.py` | Classificador Bayesiano Multivariado (usando Matriz de Covariância). |
| `models/linear_regression.py` | Regressão Linear Múltipla via equação normal (Closed-Form). |
| `utils/loader.py` | Leitura manual e parsing de arquivos ARFF irregulares. |
| `utils/encoder.py` | Tratamento de valores nulos (Imputação de médias) e codificação de categorias de texto em numéricas (Label Encoding). |
| `utils/splitter.py` | Lógica matemática de divisão dos dados para Validação Cruzada (K-Fold). |
| `utils/metrics.py` | Fórmulas puras de avaliação (Acurácia, F1-Score, Precisão, Recall, R² e R² Ajustado). |

---

## 🐍 Pré-requisitos

Antes de começar, certifique-se de que você possui o Python instalado:

```bash
python --version
# ou
python3 --version
```

---

## 🔧 Criando e ativando o ambiente virtual

Para evitar conflitos com outros projetos, é recomendado usar um ambiente virtual (`venv`).

### 🐧 Linux e 🍎 macOS

**Crie o ambiente virtual:**
```bash
python3 -m venv venv
```

**Ative o ambiente virtual:**
```bash
source venv/bin/activate
```

### 🪟 Windows

**Crie o ambiente virtual:**
```bash
python -m venv venv
```

**Ative o ambiente virtual:**

No Prompt de Comando (CMD):
```bash
venv\Scripts\activate
```

No PowerShell:
```powershell
venv\Scripts\Activate.ps1
```

> Caso ocorra erro de permissão no PowerShell, execute:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## 📥 Instalando as dependências

Com o ambiente virtual ativado, instale as bibliotecas necessárias (essencialmente o NumPy):

```bash
pip install -r requirements.txt
```

---

## ▶️ Executando o projeto

Antes de executar, certifique-se de que os caminhos dos arquivos de dados estão corretos para o seu computador no arquivo `main.py`.

Abra o arquivo `main.py` e, no topo do bloco `if __name__ == "__main__":`, ajuste as variáveis que apontam para a pasta `data`:

```python
# Exemplo (caminho relativo, com o terminal aberto na pasta do projeto):
ANIME = r"data\dataset_ANIME"
STUDENTS = r"data\dataset_STUDENTS_DROPOUT_AND_ACADEMIC_SUCCESS"
```

Depois de ajustar os caminhos, execute a bateria completa de testes:

```bash
python main.py
```

---

## ❌ Desativando o ambiente virtual

Quando terminar, você pode sair do ambiente virtual com:

```bash
deactivate
```

---

## 💡 Lembre-se

- Sempre ative o ambiente virtual antes de rodar o projeto.
- O projeto leva alguns segundos para rodar as tarefas de Classificação devido aos cálculos pesados de matrizes da Validação Cruzada do KNN em datasets com mais de 30 features.
- Execute o comando sempre a partir da pasta raiz do projeto.

