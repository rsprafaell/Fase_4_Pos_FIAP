import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Sistema Clínico de Obesidade",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_obesidade.pkl")

modelo = carregar_modelo()

# ============================================================
# HEADER
# ============================================================

st.title("🩺 Sistema Inteligente de Avaliação de Obesidade")

st.markdown("""
Este sistema utiliza **Machine Learning (Random Forest)**  
para auxiliar na classificação clínica do paciente.
""")

st.markdown("---")

# ============================================================
# LAYOUT
# ============================================================

col1, col2 = st.columns(2)

# ============================================================
# COLUNA 1 — PERFIL
# ============================================================

with col1:

    st.subheader("👤 Perfil do Paciente")

    idade = st.slider(
        "Idade",
        1, 100, 25
    )

    altura = st.number_input(
        "Altura (m)",
        1.00,
        2.50,
        1.70,
        step=0.01
    )

    peso = st.number_input(
        "Peso (kg)",
        20.0,
        300.0,
        70.0,
        step=0.1
    )

    genero = st.selectbox(
        "Gênero",
        ["Masculino", "Feminino"]
    )

    historico = st.selectbox(
        "Histórico familiar de obesidade",
        ["Sim", "Nao"]
    )

    fumante = st.selectbox(
        "Fumante",
        ["Sim", "Nao"]
    )

# ============================================================
# COLUNA 2 — ESTILO DE VIDA
# ============================================================

with col2:

    st.subheader("🍎 Hábitos e Estilo de Vida")

    alta_caloria = st.selectbox(
        "Consumo frequente de alimentos calóricos",
        ["Sim", "Nao"]
    )

    monitora = st.selectbox(
        "Monitora ingestão calórica",
        ["Sim", "Nao"]
    )

    transporte = st.selectbox(
        "Meio de transporte principal",
        [
            "Carro",
            "Moto",
            "Bicicleta",
            "Transporte_Publico",
            "A_pe"
        ]
    )

    vegetais_txt = st.selectbox(
        "Consumo de vegetais",
        [
            "Raramente",
            "Às vezes",
            "Sempre"
        ]
    )

    refeicoes_txt = st.selectbox(
        "Quantidade de refeições por dia",
        [
            "1 refeição",
            "2 refeições",
            "3 refeições",
            "4 ou mais"
        ]
    )

    agua_txt = st.selectbox(
        "Consumo diário de água",
        [
            "< 1 litro",
            "1-2 litros",
            "> 2 litros"
        ]
    )

    atividade_txt = st.selectbox(
        "Frequência de atividade física",
        [
            "Nenhuma",
            "1-2 dias/semana",
            "3-4 dias/semana",
            "5+ dias/semana"
        ]
    )

    tempo_txt = st.selectbox(
        "Tempo diário de exercício",
        [
            "0-2 horas",
            "3-5 horas",
            "> 5 horas"
        ]
    )

# ============================================================
# MAPEAMENTOS
# ============================================================

map_vegetais = {
    "Raramente": 1,
    "Às vezes": 2,
    "Sempre": 3
}

map_refeicoes = {
    "1 refeição": 1,
    "2 refeições": 2,
    "3 refeições": 3,
    "4 ou mais": 4
}

map_agua = {
    "< 1 litro": 1,
    "1-2 litros": 2,
    "> 2 litros": 3
}

map_atividade = {
    "Nenhuma": 0,
    "1-2 dias/semana": 1,
    "3-4 dias/semana": 2,
    "5+ dias/semana": 3
}

map_tempo = {
    "0-2 horas": 0,
    "3-5 horas": 1,
    "> 5 horas": 2
}

vegetais = map_vegetais[vegetais_txt]
refeicoes = map_refeicoes[refeicoes_txt]
agua = map_agua[agua_txt]
atividade = map_atividade[atividade_txt]
tempo_exercicio = map_tempo[tempo_txt]

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
# BOTÃO
# ============================================================

st.markdown("---")

if st.button("🧠 Gerar Avaliação Clínica"):

    predicao = modelo.predict(input_df)[0]

    probabilidades = modelo.predict_proba(input_df)[0]

    confianca = np.max(probabilidades)

    # ========================================================
    # RESULTADO
    # ========================================================

    st.subheader("📋 Resultado Clínico")

    st.success(f"Classificação prevista: {predicao}")

    st.info(f"Confiança do modelo: {confianca:.2%}")

    # ========================================================
    # IMC
    # ========================================================

    st.subheader("📊 Indicadores Corporais")

    st.metric("IMC", f"{imc:.2f}")

    if imc < 18.5:
        st.warning("Paciente abaixo do peso.")
    elif imc < 25:
        st.success("Faixa de peso normal.")
    elif imc < 30:
        st.warning("Paciente com sobrepeso.")
    else:
        st.error("Paciente com obesidade.")

    # ========================================================
    # PROBABILIDADES
    # ========================================================

    st.subheader("📈 Distribuição das Probabilidades")

    prob_df = pd.DataFrame({
        "Classe": modelo.classes_,
        "Probabilidade": probabilidades
    })

    prob_df = prob_df.sort_values(
        by="Probabilidade",
        ascending=False
    )

    st.bar_chart(
        prob_df.set_index("Classe")
    )

    st.dataframe(
        prob_df,
        use_container_width=True
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Sistema desenvolvido com Random Forest + Streamlit"
)
