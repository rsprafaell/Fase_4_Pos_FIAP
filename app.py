
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================

# CONFIG

# ============================================================

st.set_page_config(
page_title="Sistema de Classificação de Obesidade",
layout="wide"
)

st.title("🔬 Sistema Inteligente de Classificação de Obesidade")

st.markdown(
"Este sistema utiliza Machine Learning (Random Forest) "
"para prever a classe de obesidade do paciente."
)

# ============================================================

# LOAD MODEL

# ============================================================

@st.cache_resource
def carregar_modelo():
return joblib.load("modelo_obesidade.pkl")

modelo = carregar_modelo()

# ============================================================

# SIDEBAR

# ============================================================

st.sidebar.header("📋 Dados do Paciente")

idade = st.sidebar.slider("Idade", 1, 100, 25)

altura = st.sidebar.number_input(
"Altura (m)",
min_value=1.00,
max_value=2.50,
value=1.70,
step=0.01
)

peso = st.sidebar.number_input(
"Peso (kg)",
min_value=20.0,
max_value=300.0,
value=70.0,
step=0.1
)

genero = st.sidebar.selectbox(
"Gênero",
["Masculino", "Feminino"]
)

historico = st.sidebar.selectbox(
"Histórico Familiar de Obesidade",
["Sim", "Nao"]
)

alta_caloria = st.sidebar.selectbox(
"Consumo de alimentos calóricos",
["Sim", "Nao"]
)

fumante = st.sidebar.selectbox(
"Fumante",
["Sim", "Nao"]
)

monitora = st.sidebar.selectbox(
"Monitora calorias",
["Sim", "Nao"]
)

transporte = st.sidebar.selectbox(
"Meio de Transporte",
[
"Carro",
"Moto",
"Bicicleta",
"Transporte_Publico",
"A_pe"
]
)

# ============================================================

# VARIÁVEIS ORDINAIS

# ============================================================

vegetais = st.sidebar.slider(
"Consumo de Vegetais",
1, 3, 2
)

refeicoes = st.sidebar.slider(
"Refeições por dia",
1, 4, 3
)

agua = st.sidebar.slider(
"Consumo de Água",
1, 3, 2
)

atividade = st.sidebar.slider(
"Frequência de Atividade Física",
0, 3, 1
)

tempo_exercicio = st.sidebar.slider(
"Tempo de Exercício",
0, 2, 1
)

# ============================================================

# FEATURE ENGINEERING

# ============================================================

imc = peso / (altura ** 2)

score_atividade = atividade * tempo_exercicio

# ============================================================

# DATAFRAME

# ============================================================

input_df = pd.DataFrame([{

```
"Genero": genero,
"Idade": idade,
"Historico_Familiar": historico,
"Consumo_Alta_Caloria": alta_caloria,
"Consumo_Vegetais": vegetais,
"Refeicoes_Dia": refeicoes,
"Fumante": fumante,
"Monitora_Calorias": monitora,
"Consumo_Agua": agua,
"Frequencia_Ativ_Fisica": atividade,
"Tempo_Exercicio": tempo_exercicio,
"Meio_Transporte": transporte,
"IMC": imc,
"Score_Atividade": score_atividade
```

}])

# ============================================================

# PREVISÃO

# ============================================================

if st.button("🚀 Gerar Diagnóstico"):

```
predicao = modelo.predict(input_df)[0]

probabilidades = modelo.predict_proba(input_df)[0]

confianca = np.max(probabilidades)

classes = modelo.classes_

# --------------------------------------------------------
# RESULTADO
# --------------------------------------------------------

st.subheader("📌 Resultado da Predição")

st.success(f"Classe prevista: {predicao}")

st.info(f"Confiança do modelo: {confianca:.2%}")

# --------------------------------------------------------
# IMC
# --------------------------------------------------------

st.subheader("📊 Indicadores Clínicos")

st.metric("IMC", f"{imc:.2f}")

# --------------------------------------------------------
# PROBABILIDADES
# --------------------------------------------------------

st.subheader("📈 Probabilidades por Classe")

prob_df = pd.DataFrame({
    "Classe": classes,
    "Probabilidade": probabilidades
})

prob_df = prob_df.sort_values(
    by="Probabilidade",
    ascending=False
)

st.bar_chart(
    prob_df.set_index("Classe")
)

st.dataframe(prob_df)
```

# ============================================================

# FOOTER

# ============================================================

st.markdown("---")

st.caption(
"Modelo Random Forest + Explainable AI (SHAP)"
)
