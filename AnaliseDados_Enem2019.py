import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

#1.Carrega os dados do arquivo Enem 2019
#sep=";" → informa ao pandas que os dados estão separados por ponto e vírgula, não por vírgula.
#encoding="latin1" → funciona melhor com acentos em arquivos do Windows/Excel.
dados_enem =  pd.read_csv ('C:\\Users\\clsoares\\Desktop\\Projeto_Dados_Enem2019_Youth2025\\microdados_enem_2019\\DADOS\\MICRODADOS_ENEM_2019.csv',sep=";", encoding="latin1")
dados_enem = dados_enem.drop(columns=["NU_ANO","TX_RESPOSTAS_CN", "TX_RESPOSTAS_CH","TX_RESPOSTAS_LC","TX_RESPOSTAS_MT","TX_GABARITO_CN", "TX_GABARITO_LC", "TX_GABARITO_MT"])
print (dados_enem)
qtdebrasil= len (dados_enem)
print (f" O número de inscritos no Enem 2019: {qtdebrasil} inscritos")
print(dados_enem.columns)

#2.Cria novo arquivo Enem Ceará 2019, somente com os dados do Ceará
enem_ceara = dados_enem[dados_enem["SG_UF_PROVA"] == "CE"]
enem_ceara.to_csv("Novo.csv", index=False, encoding="utf8")  
print (enem_ceara)
qtdeceara= len (enem_ceara)
print (f" O número de inscritos no Enem 2019: {qtdeceara} inscritos do Ceará")

#3.cria uma coluna sobre o Status do Inscrito na prova do Enem no Ceará 
#Regras:
# 0 → se o candidato faltou todas as provas
# 1 → se o candidato estiver presente em todas as provas
# 2 → se o candidato fez pelo menos uma prova, mas não todas
#A nova coluna Status_Inscrito vai conter:
#5 provas feitas =  presente = 1
#0 faltou todas as provas = 0
# 1 a 4 fez parcialmente = 2


colunas_notas = [
    "NU_NOTA_CN",       # Ciências da Natureza
    "NU_NOTA_CH",       # Ciências Humanas
    "NU_NOTA_LC",       # Linguagens e Códigos
    "NU_NOTA_MT",       # Matemática
    "NU_NOTA_REDACAO"   # Redação
]

# Conta quantas provas o candidato fez (notas NÃO nulas)
enem_ceara["Provas_Feitas"] = enem_ceara[colunas_notas].notnull().sum(axis=1)

# Define o Status_Inscrito conforme a regra
enem_ceara["Status_Inscrito"] = enem_ceara["Provas_Feitas"].apply(
    lambda x: 1 if x == 5 else (0 if x == 0 else 2)
)

# Mostra os totais por status
totais = enem_ceara["Status_Inscrito"].value_counts().sort_index()
print("Totais por Status_Inscrito:")
print(f"0 - Faltou todas as provas:       {totais.get(0, 0)}")
print(f"1 - Presente em todas as provas: {totais.get(1, 0)}")
print(f"2 - Presente em parte das provas: {totais.get(2, 0)}")

print(enem_ceara.columns)

filtros1 = enem_ceara['Status_Inscrito'] == 0
num_faltantes = filtros1.sum()
print(f"Número de participante que faltaram no Ceará: {num_faltantes}")
profalt= (num_faltantes/qtdeceara)*100
print ("Percentagem de faltantes(%) :", profalt)
filtros2 = enem_ceara['Status_Inscrito'] == 1
num_presentes=filtros2.sum()
print(f"Número de participantes que compareceram no Ceará: {num_presentes}")
propres= (num_presentes/qtdeceara)*100
print ("Percentagem de presentes em todas as provas (%):", propres)

plt.figure(figsize=(10, 6))
sns.histplot(enem_ceara["NU_NOTA_REDACAO"], kde=True, bins=30, color='skyblue')
plt.title("Distribuição da Nota de Redação - Presentes")
plt.xlabel("Nota de Redação")
plt.ylabel("Frequência")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(enem_ceara["NU_NOTA_MT"], kde=True, bins=30, color='salmon')
plt.title("Distribuição da Nota de Matemática - Presentes")
plt.xlabel("Nota de Matemática")
plt.ylabel("Frequência")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=enem_ceara, x='TP_SEXO', y='NU_NOTA_MT')
plt.title("Nota de Matemática por Sexo")
plt.xlabel("Sexo")
plt.ylabel("Nota de Matemática")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=enem_ceara, x='TP_COR_RACA', y='NU_NOTA_MT')
plt.title("Nota de Matemática por Raça/Cor")
plt.xlabel("Raça/Cor")
plt.ylabel("Nota de Matemática")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=enem_ceara, x='TP_ESCOLA', y='NU_NOTA_MT')
plt.title("Desempenho em Matemática por Tipo de Escola")
plt.xlabel("Tipo de Escola")
plt.ylabel("Nota de Matemática")
plt.grid(True)
plt.show()

renda_vs_nota = enem_ceara.groupby('Q006')[['NU_NOTA_MT', 'NU_NOTA_REDACAO']].mean().sort_index()
print(renda_vs_nota)
plt.figure(figsize=(12, 6))
renda_vs_nota['NU_NOTA_MT'].plot(marker='o')
plt.title("Nota Média de Matemática por Faixa de Renda (Q006)")
plt.xlabel("Faixa de Renda (Q006)")
plt.ylabel("Nota Média em Matemática")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=enem_ceara, x='TP_FAIXA_ETARIA', y='NU_NOTA_MT', alpha=0.3)
plt.title('Idade vs Nota de Matemática')
plt.xlabel('Idade')
plt.ylabel('Nota de Matemática')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 12))
sns.heatmap(enem_ceara.select_dtypes(include='number').corr(), cmap='coolwarm', annot=False)
plt.title('Matriz de Correlação das Variáveis Numéricas')
plt.show()



#label encoder para variáveis ordinais
le = LabelEncoder()
enem_ceara["Q001_codificada"] = le.fit_transform(enem_ceara["Q001"])

# Codifica Q002 (mãe)
enem_ceara["Q002_codificada"] = le.fit_transform(enem_ceara["Q002"])

# Codifica Q006 (renda)
enem_ceara["Q006_codificada"] = le.fit_transform(enem_ceara["Q006"])

# Mostra os primeiros valores
print(enem_ceara[["Q001", "Q001_codificada", "Q002", "Q002_codificada", "Q006", "Q006_codificada"]].head())


#OneHotEncoder
sexo = enem_ceara[["TP_SEXO"]]
encoder = OneHotEncoder(sparse_output=False, drop='first') 
sexo_encoded = encoder.fit_transform(sexo)
enem_ceara["Sexo_One"] = sexo_encoded.astype(int)
print(enem_ceara[["TP_SEXO", "Sexo_One"]].head())

#normalizar variáveis numéricas
colunas_numericas = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
scaler = MinMaxScaler() 
df_normalizado = enem_ceara.copy()
df_normalizado[colunas_numericas] = scaler.fit_transform(enem_ceara[colunas_numericas])
print(df_normalizado[colunas_numericas].head())

#prever faltante
variaveis = ['TP_SEXO', 'Q001', 'Q002', 'Q006', 'TP_COR_RACA', 'TP_ESCOLA']


# Cria a variável-alvo: 1 se faltou tudo, 0 se foi a pelo menos uma prova
colunas_notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
enem_ceara['Faltou'] = enem_ceara[colunas_notas].isnull().all(axis=1).astype(int)

# Seleciona colunas e remove ausentes
variaveis = ['TP_SEXO', 'Q001', 'Q002', 'Q006']
df_modelo = enem_ceara[variaveis + ['Faltou']].dropna()

# Codifica com LabelEncoder
encoders = {}
for col in variaveis:
    le = LabelEncoder()
    df_modelo[col] = le.fit_transform(df_modelo[col])
    encoders[col] = le  # salva para usar depois

# Treina o modelo
X = df_modelo[variaveis]
y = df_modelo['Faltou']

modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo.fit(X, y)

# Salva o modelo e os encoders
joblib.dump(modelo, "modelo_faltou.pkl")
joblib.dump(encoders, "encoders.pkl")

# Carrega o modelo e os encoders
modelo = joblib.load("modelo_faltou.pkl")
encoders = joblib.load("encoders.pkl")

# Define as variáveis de entrada
variaveis = ['TP_SEXO', 'Q001', 'Q002', 'Q006']
entrada = {}

print("Digite os valores para prever se o candidato irá faltar (use letras A, B, C...):")
print("TP_SEXO: M = Masculino | F = Feminino")
print("Q001 (Escolaridade do pai): A-Nunca estudou; B-Não completou a 4ª série/5º ano do Ensino Fundamental. C-Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental. D-	Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio. E-Completou o Ensino Médio, mas não completou a Faculdade.F-	Completou a Faculdade, mas não completou a Pós-graduação.G-Completou a Pós-graduação.H-Não sei")
print ("Q002 (Escolaridade do mãe): A-Nunca estudou; B-Não completou a 4ª série/5º ano do Ensino Fundamental. C-Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental. D-	Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio. E-Completou o Ensino Médio, mas não completou a Faculdade.F-	Completou a Faculdade, mas não completou a Pós-graduação.G-Completou a Pós-graduação.H-Não sei")
print ("Q006 (renda mensal de sua família): A-Nenhuma renda. B-Até R$ 998,00. C-De R$ 998,01 até R$ 1.497,00. D-De R$ 1.497,01 até R$ 1.996,00. E-	De R$ 1.996,01 até R$ 2.495,00. F-	De R$ 2.495,01 até R$ 2.994,00. G-	De R$ 2.994,01 até R$ 3.992,00. H-	De R$ 3.992,01 até R$ 4.990,00.I-	De R$ 4.990,01 até R$ 5.988,00. J-	De R$ 5.988,01 até R$ 6.986,00. K	De R$ 6.986,01 até R$ 7.984,00. L-	De R$ 7.984,01 até R$ 8.982,00. M-	De R$ 8.982,01 até R$ 9.980,00. N-	De R$ 9.980,01 até R$ 11.976,00. O-De R$ 11.976,01 até R$ 14.970,00.P-	De R$ 14.970,01 até R$ 19.960,00.Q-Mais de R$ 19.960,00.")
for var in variaveis:
    valor = input(f"{var}: ").strip().upper()
    # Codifica com o LabelEncoder correspondente
    if var in encoders:
        valor_codificado = encoders[var].transform([valor])[0]
        entrada[var] = valor_codificado

# Cria DataFrame com entrada
df_entrada = pd.DataFrame([entrada])

# Faz a previsão
predicao = modelo.predict(df_entrada)[0]

# Exibe resultado
print("\n✅ Resultado da previsão:")
print("👉 O candidato provavelmente vai FALTAR." if predicao == 1 else "✅ O candidato provavelmente VAI COMPARECER.")

#prever nota do aluno
target = 'NU_NOTA_REDACAO'
variaveis = ['TP_SEXO', 'Q001', 'Q002', 'Q006', 'TP_ESCOLA', 'TP_COR_RACA']
df_modelo = enem_ceara[variaveis + [target]].dropna()
print(enem_ceara['Q001'].unique())
print(enem_ceara['Q002'].unique())
print(enem_ceara['Q006'].unique())
print(enem_ceara['TP_ESCOLA'].unique())
print(enem_ceara['TP_COR_RACA'].unique())
print(enem_ceara['TP_SEXO'].unique())

# Codifica variáveis categóricas
encoders = {}
for col in variaveis:
    le = LabelEncoder()
    df_modelo[col] = le.fit_transform(df_modelo[col])
    encoders[col] = le

X = df_modelo[variaveis]
y = df_modelo[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Previsão
y_pred = modelo.predict(X_test)

# Avaliação
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) 
print("RMSE:", rmse)
print("R² (coef. de determinação):", r2_score(y_test, y_pred))

entrada = {}

print("\nDigite os dados para prever a nota de redação:")
for var in variaveis:
    valor = input(f"{var}: ").strip().upper()
    valor_cod = encoders[var].transform([valor])[0]
    entrada[var] = valor_cod

df_entrada = pd.DataFrame([entrada])
nota_prevista = modelo.predict(df_entrada)[0]

print(f"\n📝 Nota de redação prevista: {nota_prevista:.2f}")