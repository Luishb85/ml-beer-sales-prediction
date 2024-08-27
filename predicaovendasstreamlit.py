import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# Carregar o modelo treinado
model = xgb.XGBRegressor()
model.load_model("xgboost_model.json")

# Função para realizar a previsão
def predict_sales(dia, dia_semana, mes, temperatura, vendas):
    input_data = pd.DataFrame({
        'Temperatura': [temperatura],
        'Dia da Semana': [dia_semana],
        'Mês': [mes],
        'Vendas': [vendas],
        'Temp_DiaSemana_Interaction': [dia_semana * temperatura],
        'Semana_Mes_Interaction': [dia_semana * mes]
    })

    model_feature_names = [
        'Temperatura', 'Dia da Semana', 'Mês', 'Vendas',
        'Temp_DiaSemana_Interaction', 'Semana_Mes_Interaction'
    ]
    input_data = input_data[model_feature_names]
    predicted_sales = model.predict(input_data)
    return predicted_sales[0]

# Interface do Streamlit
st.title("Previsão de Vendas de Cerveja")

# Mensagens iniciais ao usuário
st.write("""
### Instruções:
- Utilize a barra lateral para inserir os dados da previsão de vendas.
- Os dias devem estar entre 1 e 31.
- A temperatura deve estar entre 0°C e 50°C.
- Selecione o mês, o dia da semana e informe as vendas atuais.
""")

# Entradas do usuário
dia = st.sidebar.number_input("Dia do Mês", min_value=1, max_value=31, step=1)
dia_semana = st.sidebar.selectbox("Dia da Semana", ['segunda', 'terça', 'quarta', 'quinta', 'sexta', 'sábado', 'domingo'])
dia_semana_num = {'segunda': 1, 'terça': 2, 'quarta': 3, 'quinta': 4, 'sexta': 5, 'sábado': 6, 'domingo': 7}[dia_semana]
mes = st.sidebar.selectbox("Mês", list(range(1, 13)))
temperatura = st.sidebar.number_input("Temperatura esperada (°C)", min_value=0.0, max_value=50.0, value=18.0, step=0.1)
vendas = st.sidebar.number_input("Vendas Atuais", min_value=0, step=1)

# Exibir previsão antes dos gráficos
if st.sidebar.button("Calcular"):
    if not (1 <= dia <= 31):
        st.error("Por favor, insira um número válido para o dia (1-31).")
    else:
        resultado_vendas = predict_sales(dia, dia_semana_num, mes, temperatura, vendas)
        
        # Mostrar o resultado da previsão em destaque
        st.markdown(f"## Previsão de Vendas: **R$ {resultado_vendas:.2f}**")

# Carregar dados históricos do arquivo CSV
df = pd.read_csv("DadosVendasBM.csv", delimiter=';')
df['Ticket Médio'] = df['Ticket Médio'].str.replace('R$', '').str.replace('.', '').str.replace(',', '.').astype(float)
df['Valor Total'] = df['Valor Total'].str.replace('R$', '').str.replace('.', '').str.replace(',', '.').astype(float)

# Gráfico de Linha: Vendas Médias por Semana
st.subheader("Vendas Médias por Semana")
vendas_medias_semana = df.groupby("Dia da Semana")["Vendas"].mean()
plt.figure(figsize=(10, 6))
plt.plot(vendas_medias_semana.index, vendas_medias_semana.values, marker='o')
plt.title("Vendas Médias por Semana")
plt.xlabel("Dia da Semana")
plt.ylabel("Média de Vendas")
st.pyplot(plt)

# Gráfico de Linha: Vendas Médias por Mês
st.subheader("Vendas Médias por Mês")
vendas_medias_mes = df.groupby("Mês")["Vendas"].mean()
plt.figure(figsize=(10, 6))
plt.plot(vendas_medias_mes.index, vendas_medias_mes.values, marker='o')
plt.title("Vendas Médias por Mês")
plt.xlabel("Mês")
plt.ylabel("Média de Vendas")
st.pyplot(plt)
