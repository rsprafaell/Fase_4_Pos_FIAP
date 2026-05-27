import streamlit as st
import pandas as pd
import numpy as np
import joblib

# CONFIG
st.set_page_config(
    page_title="Sistema de Classificação de Obesidade",
    layout="wide"
)

st.title("🔬 Sistema Inteligente de Classificação de Obesidade")

st.markdown(
    "Este sistema utiliza Machine Learning (Random Forest) "
    "para prever a classe de obesidade do paciente."
)

# LOAD MODEL
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_obesidade.pkl")

modelo = carregar_modelo()

# SIDEBAR
st.sidebar.header("📋 Dados do Paciente")

idade = st.sidebar.slider("Idade", 1, 100, 25)

altura = st.sidebar.number_input("Altura (m)", 1.0, 2.5, 1.7)
peso = st.sidebar.number_input("Peso (kg)", 20.0, 300.0, 70.0)

genero = st.sidebar.selectbox("Gênero", ["Masculino", "Feminino"])
historico = st.sidebar.selectbox("Histórico Familiar", ["Sim", "Nao"])
alta_caloria = st.sidebar.selectbox("Alta Caloria", ["Sim", "Nao"])
fumante = st.sidebar.selectbox("Fumante", ["Sim", "Nao"])
monitora = st.sidebar.selectbox("Monitora calorias", ["Sim", "Nao"])

transporte = st.sidebar.selectbox(
    "Transporte",
    ["Carro", "Moto", "Bicicleta", "Transporte_Publico", "A_pe"]
)

vegetais = st.sidebar.slider("Vegetais", 1, 3, 2)
refeicoes = st.sidebar.slider("Refeições", 1, 4, 3)
agua = st.sidebar.slider("Água", 1, 3, 2)
atividade = st.sidebar.slider("Atividade física", 0, 3, 1)
tempo_exercicio = st.sidebar.slider("Tempo exercício", 0, 2, 1)

# FEATURE ENGINEERING
imc = peso / (altura ** 2)
score_atividade = atividade * tempo_exercicio

# DATAFRAME
input_df = pd.DataFrame([{
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
}])

# PREVISÃO
if st.button("🚀 Gerar Diagnóstico"):

    predicao = modelo.predict(input_df)[0]
    probabilidades = modelo.predict_proba(input_df)[0]
    confianca = np.max(probabilidades)

    st.subheader("Resultado")
    st.success(f"Classe prevista: {predicao}")
    st.info(f"Confiança: {confianca:.2%}")

    st.metric("IMC", f"{imc:.2f}")

    prob_df = pd.DataFrame({
        "Classe": modelo.classes_,
        "Probabilidade": probabilidades
    }).sort_values(by="Probabilidade", ascending=False)

    st.bar_chart(prob_df.set_index("Classe"))
    st.dataframe(prob_df)

# FOOTER
st.markdown("---")
st.caption("Random Forest + Streamlit")
