import altair as alt
import joblib
import pandas as pd
import streamlit as st

# 1. Configuração da página
st.set_page_config(page_title="Classificação de países — HELP International", layout="wide")


# 2. Carregamento do modelo
@st.cache_resource
def carregar_modelo():
    artefato = joblib.load("modelo_paises.pkl")
    return (artefato["modelo"], artefato["colunas"], artefato["nomes"],
            artefato["perfil"], artefato["referencia"], artefato["info"])


modelo, colunas, nomes, perfil, referencia, info = carregar_modelo()


# 3. Descrição dos campos (rótulo em português, valor inicial = mediana da base, formato)
CAMPOS = {
    "child_mort": ("Mortalidade infantil (mortes de menores de 5 anos por 1.000 nascidos vivos)", 19.30, "%.1f"),
    "exports":    ("Exportações (% do PIB per capita)", 35.00, "%.1f"),
    "health":     ("Gasto com saúde (% do PIB per capita)", 6.32, "%.2f"),
    "imports":    ("Importações (% do PIB per capita)", 43.30, "%.1f"),
    "income":     ("Renda por pessoa (US$)", 9960.00, "%.0f"),
    "inflation":  ("Inflação (% ao ano)", 5.39, "%.2f"),
    "life_expec": ("Expectativa de vida (anos)", 73.10, "%.1f"),
    "total_fer":  ("Filhos por mulher", 2.41, "%.2f"),
    "gdpp":       ("PIB per capita (US$)", 4660.00, "%.0f"),
}
ROTULO_CURTO = {
    "child_mort": "Mortalidade infantil", "exports": "Exportações (% PIB)", "health": "Gasto com saúde (% PIB)",
    "imports": "Importações (% PIB)", "income": "Renda por pessoa (US$)", "inflation": "Inflação (%)",
    "life_expec": "Expectativa de vida (anos)", "total_fer": "Filhos por mulher", "gdpp": "PIB per capita (US$)",
}
CORES = {0: "#2e7d32", 1: "#ef8a17", 2: "#c62828"}  # baixa, média e alta prioridade


# 4. Interface
st.title("Classificação de países por prioridade de ajuda humanitária")
st.markdown(
    "Informe os indicadores de um país para saber seu nível de prioridade de ajuda.  \n"
    "**Autor:** João Pedro Balduino"
)

with st.form("indicadores"):
    col1, col2 = st.columns(2)
    valores = {}
    for i, coluna in enumerate(colunas):
        rotulo, padrao, formato = CAMPOS[coluna]
        with (col1 if i % 2 == 0 else col2):
            valores[coluna] = st.number_input(rotulo, value=float(padrao), format=formato, key=coluna)
    classificar = st.form_submit_button("Classificar", type="primary")


# 5. Classificação
if classificar:
    entrada = pd.DataFrame([valores])[colunas]          # valores na escala original, na ordem do treino
    grupo = int(modelo.predict(entrada)[0])
    nome_grupo = nomes[grupo]

    st.subheader("Resultado")
    mensagem = f"**Grupo {grupo}: {nome_grupo}**"
    if grupo == max(nomes, key=lambda g: perfil.loc[g, "child_mort"]):
        st.error(mensagem)
    elif grupo == min(nomes, key=lambda g: perfil.loc[g, "child_mort"]):
        st.success(mensagem)
    else:
        st.warning(mensagem)

    # Valores informados pelo usuário
    st.markdown("##### Valores informados")
    st.dataframe(
        pd.DataFrame({"Indicador": [ROTULO_CURTO[c] for c in colunas], "Valor": [valores[c] for c in colunas]}),
        hide_index=True, width="stretch",
    )

    # Comparação com o perfil típico do grupo
    st.markdown(f"##### Comparação com o perfil médio do grupo ({int(perfil.loc[grupo, 'n_paises'])} países)")
    comparacao = pd.DataFrame({
        "Indicador": [ROTULO_CURTO[c] for c in colunas],
        "País informado": [valores[c] for c in colunas],
        "Média do grupo": [perfil.loc[grupo, c] for c in colunas],
    })
    comparacao["Diferença"] = comparacao["País informado"] - comparacao["Média do grupo"]
    st.dataframe(comparacao.round(2), hide_index=True, width="stretch")

    with st.expander("Médias dos indicadores em todos os grupos"):
        tabela_perfil = perfil[colunas + ["n_paises"]].rename(columns={**ROTULO_CURTO, "n_paises": "Nº de países"})
        tabela_perfil.index = [f"Grupo {g}: {nomes[g]}" for g in tabela_perfil.index]
        st.dataframe(tabela_perfil.round(2), width="stretch")

    # País informado no plano PC1 x PC2, usando a padronização e o PCA do próprio pipeline
    st.markdown("##### Posição do país no plano PC1 × PC2")
    pcs = modelo[:-1].transform(entrada)[0]              # 5 componentes; apenas os 2 primeiros são exibidos
    var1, var2 = info["variancia_pc1_pc2"]

    base = referencia.assign(Grupo=referencia["grupo"].map(lambda g: f"Grupo {g}: {nomes[g]}"))
    dominio = [f"Grupo {g}: {nomes[g]}" for g in sorted(nomes)]
    escala = alt.Scale(domain=dominio, range=[CORES.get(g, "#888") for g in sorted(nomes)])
    eixo_x = alt.X("PC1:Q", title=f"PC1 — desenvolvimento socioeconômico ({var1:.1%} da variância)")
    eixo_y = alt.Y("PC2:Q", title=f"PC2 — abertura comercial ({var2:.1%} da variância)")

    paises = alt.Chart(base).mark_circle(size=55, opacity=0.6).encode(
        x=eixo_x, y=eixo_y,
        color=alt.Color("Grupo:N", scale=escala, legend=alt.Legend(orient="bottom", title=None, labelLimit=400, columns=1)),
        tooltip=[alt.Tooltip("country:N", title="País"), "Grupo:N",
                 alt.Tooltip("PC1:Q", format=".2f"), alt.Tooltip("PC2:Q", format=".2f")],
    )
    ponto = pd.DataFrame({"PC1": [pcs[0]], "PC2": [pcs[1]], "rotulo": ["País informado"]})
    destaque = alt.Chart(ponto).mark_point(shape="diamond", size=380, filled=True, fill="#1e88e5", stroke="white",
                                           strokeWidth=2, opacity=1).encode(
        x="PC1:Q", y="PC2:Q", tooltip=[alt.Tooltip("PC1:Q", format=".2f"), alt.Tooltip("PC2:Q", format=".2f")])
    texto = alt.Chart(ponto).mark_text(align="left", dx=14, dy=-12, fontWeight="bold", fontSize=14,
                                       color="#1e88e5").encode(
        x="PC1:Q", y="PC2:Q", text="rotulo:N")
    st.altair_chart((paises + destaque + texto).properties(height=480).interactive(), width="stretch")
    st.caption(f"Coordenadas do país informado: PC1 = {pcs[0]:.2f}, PC2 = {pcs[1]:.2f}. "
               f"O PCA do pipeline usa {info['n_componentes']} componentes; o gráfico mostra apenas os dois primeiros.")

    # Aviso sobre limitações e uso responsável
    st.caption(
        "**Uso responsável:** resultado de triagem inicial, baseado em dados antigos e médias nacionais. "
        "Países perto da fronteira entre grupos podem mudar de classificação. Confirme com dados atualizados "
        "e análise de especialistas antes de decidir."
    )


# Sobre o modelo (versão inicial do model card)
st.divider()
with st.expander("Sobre o modelo", expanded=False):
    st.markdown(f"""
| Item | Descrição |
|---|---|
| **Finalidade** | Classificar países em três níveis de prioridade para a alocação de recursos humanitários da HELP International |
| **Origem dos dados** | {info['fonte']}; {info['n_paises']} países e {len(colunas)} indicadores socioeconômicos e de saúde |
| **Pipeline** | `StandardScaler` → `PCA(n_components=0.9)` → `KMeans(n_clusters={info['n_grupos']}, n_init=10, random_state=42)` |
| **Componentes do PCA** | {info['n_componentes']} componentes, preservando {info['variancia_explicada']:.1%} da variância |
| **Número de grupos** | {info['n_grupos']} — {"; ".join(f"Grupo {g}: {nomes[g]} ({int(perfil.loc[g, 'n_paises'])} países)" for g in sorted(nomes))} |
| **Entradas** | {", ".join(ROTULO_CURTO[c] for c in colunas)} (escala original) |
| **Versão do scikit-learn** | {info['versao_sklearn']} |
| **Limitações** | Dados desatualizados e de procedência pouco documentada; silhueta moderada (≈ 0,31), indicando grupos pouco separados; não considera população, desigualdade interna nem conflitos |
| **Uso recomendado** | Apoio à triagem inicial, sempre complementado por dados atuais e análise de especialistas |
""")
