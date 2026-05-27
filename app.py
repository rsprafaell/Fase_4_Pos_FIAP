import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# CONFIG (SEMPRE PRIMEIRO)
# ============================================================
st.set_page_config(
    page_title="Sistema de Classificação de Obesidade",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🔬 Sistema Inteligente de Classificação de Obesidade")

st.markdown(
    "Modelo de Machine Learning (Random Forest) para prever classe de obesidade."
)

# ============================================================
# CARREGAMENTO DO MODELO (SEGURO)
# ============================================================
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_obesidade.pkl")

try:
    modelo = carregar_modelo()
    st.success("Modelo carregado com sucesso ✔")
except Exception as e:
    st.error("Erro ao carregar modelo")
    st.exception(e)
    st.stop()

# ============================================================
# INPUTS (SEM DEPENDER DE SIDEBAR - MAIS ESTÁVEL)
# ============================================================
st.subheader("📋 Dados do Paciente")

col1, col2 = st.columns(2)

with col1:
    idade = st.slider("Idade", 1, 100, 25)
    altura = st.number_input("Altura (m)", 1.0, 2.5, 1.70)
    peso = st.number_input("Peso (kg)", 20.0, 300.0, 70.0)
    genero = st.selectbox("Gênero", ["Masculino", "Feminino"])
    historico = st.selectbox("Tem Histórico Familiar de Obesidade", ["Sim", "Nao"])
    alta_caloria = st.selectbox("Condome Alta Caloria", ["Sim", "Nao"])
    fumante = st.selectbox("Fumante", ["Sim", "Nao"])

with col2:
    monitora = st.selectbox("Monitora calorias", ["Sim", "Nao"])
    transporte = st.selectbox(
        "Tipo de Transporte mais Utilizado",
        ["Carro", "Moto", "Bicicleta", "Transporte_Publico", "A_pe"]
    )
    vegetais = st.slider("Frequencia que consome Vegetais", 1, 3, 2)
    refeicoes = st.slider("Quantidade Refeições/dia", 1, 4, 3)
    agua = st.slider("Consumo Água em litros", 1, 3, 2)
    atividade = st.slider("Atividade física", 0, 3, 1)
    tempo_exercicio = st.slider("Tempo exercício", 0, 2, 1)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
imc = peso / (altura ** 2)
score_atividade = atividade * tempo_exercicio

# ============================================================
# DATAFRAME
# ============================================================
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

# ============================================================
# BOTÃO DE PREVISÃO
# ============================================================
if st.button("🚀 Gerar Diagnóstico"):

    try:
        predicao = modelo.predict(input_df)[0]
        probabilidades = modelo.predict_proba(input_df)[0]
        confianca = np.max(probabilidades)

        st.subheader("📌 Resultado")
        st.success(f"Classe prevista: {predicao}")
        st.info(f"Confiança: {confianca:.2%}")

        st.subheader("📊 IMC")
        st.metric("IMC", f"{imc:.2f}")

        st.subheader("📈 Probabilidades")

        prob_df = pd.DataFrame({
            "Classe": modelo.classes_,
            "Probabilidade": probabilidades
        }).sort_values(by="Probabilidade", ascending=False)

        st.bar_chart(prob_df.set_index("Classe"))
        st.dataframe(prob_df)

    except Exception as e:
        st.error("Erro na predição")
        st.exception(e)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Random Forest + Streamlit ML App")
