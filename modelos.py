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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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

features = [
    'TP_SEXO', 'TP_COR_RACA', 'TP_ESTADO_CIVIL', 'TP_ESCOLA', 'TP_ENSINO',
    'IN_TREINEIRO', 'TP_DEPENDENCIA_ADM_ESC', 'TP_LOCALIZACAO_ESC'
]

# Filtrar apenas essas colunas + alvo
df_model = enem_ceara[features + ['Status_Inscrito']].dropna()

# 4. Codificar variáveis categóricas (one-hot ou label encoding)
df_model = pd.get_dummies(df_model, columns=features)

# Separar em X (entradas) e y (saída)
X = df_model.drop('Status_Inscrito', axis=1)
y = df_model['Status_Inscrito']

# 5. Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Treinar o modelo SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# 8. Avaliar o modelo
y_pred = svm.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#df = df.sample(frac=0.1, random_state=42)  # usa 10% dos dados

novo_dado = pd.DataFrame([{
    'TP_SEXO': 'M',
    'TP_COR_RACA': 1,
    'TP_ESTADO_CIVIL': 1,
    'TP_ESCOLA': 1,
    'TP_ENSINO': 1,
    'IN_TREINEIRO': 0,
    'TP_DEPENDENCIA_ADM_ESC': 2,
    'TP_LOCALIZACAO_ESC': 1
}])

# Aplicar o mesmo get_dummies
novo_dado_encoded = pd.get_dummies(novo_dado)

# Garantir que todas as colunas usadas no treino estão presentes
for col in X.columns:
    if col not in novo_dado_encoded:
        novo_dado_encoded[col] = 0  # adiciona coluna faltante com 0

# Reordenar colunas para bater com X do treino
novo_dado_encoded = novo_dado_encoded[X.columns]

# Escalar os dados com o mesmo scaler
novo_dado_scaled = scaler.transform(novo_dado_encoded)

# Fazer a previsão
predicao = svm.predict(novo_dado_scaled)

# Interpretar o resultado
print("Presente" if predicao[0] == 1 else "Faltou")


