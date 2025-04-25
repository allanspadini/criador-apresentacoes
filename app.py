import streamlit as st
import os
import tempfile
import zipfile
import io
import re
import time
from pathlib import Path
import yaml
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Bibliotecas necessárias do código original
import nest_asyncio
from pydantic import BaseModel
from typing import List, Literal
from docling.document_converter import DocumentConverter
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.gemini import GeminiModel
from google import genai
from google.genai import types

# Aplicar nest_asyncio para permitir execução de loops aninhados
nest_asyncio.apply()


#Pega as chaves
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY


modelo = GeminiModel('gemini-2.0-flash', provider='google-gla')
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Classes do modelo
class Formatacao(BaseModel):
    title: str
    author: str
    format: str
    theme: str
    incremental: bool

class Imagem(BaseModel):
    slide: int
    markdown: str


class Imagens(BaseModel):
    imagens: List[Imagem]

def formatar_para_yaml(result_data):
    """
    Formata os dados de result.data no formato YAML esperado.
    """
    # Constrói a estrutura do dicionário com base nos dados fornecidos
    yaml_data = {
        "title": result_data.title,
        "author": result_data.author,
        "format": {
            result_data.format: {
                "theme": result_data.theme,
                "incremental": result_data.incremental,
            }
        },
    }

    # Converte para YAML
    import yaml
    return yaml.dump(
        yaml_data,
        sort_keys=False,  # Mantém a ordem dos campos
        default_flow_style=False,  # Gera o YAML no estilo de múltiplas linhas
    )


agente_formatador = Agent(
    'google-gla:gemini-2.0-flash',
    system_prompt='''Você é um criador de apresentações que retorna a formatação para apresentação. A formatação tem o seguinte formato:
---
title: "Presentation"
author: "John Doe"
format:
  revealjs:
    theme: dark
    incremental: true
---

As opções de tema são:
beige
blood
dark
default
league
moon
night
serif
simple
sky
solarized

Você deve inferir quais as propriedades da formatação a partir do prompt.
''',result_type=Formatacao
)


agente_roteirista = Agent(
    model=modelo,
    system_prompt='''Você é um criador de apresentações que retorna o roteiro de um slide.
    A apresentação deve responder a perguntas relacionadas no título dos slides e o roteiro de cada slide
    deve ficar entre as tags ::: {.notes} roteiro ::: . O roteiro deve ser um texto corrido sem bullets e ser elaborado com
    base no resultado da consulta à ferramenta de nome consulta. Nunca escreva sem antes consultar a ferramenta consulta, mas sempre coloque algo
    nas notas para a pessoa ter o que falar no slide.

    Exemplo de resultado:

    ## título do slide

    ::: {.notes}
    roteiro do slide
    :::

'''
)

@agente_roteirista.tool_plain(docstring_format='google', require_parameter_descriptions=True,retries=2)
def consulta(pergunta: str):
    """Realiza uma consulta em uma coleção de relatórios com base na pergunta fornecida.

    Args:
        pergunta (str): A pergunta ou consulta que será usada para gerar o embedding e buscar o relatório correspondente.
    Returns:
        str: O conteúdo do relatório mais relevante encontrado com base na pergunta.

    Exemplo:
        >>> consulta("Qual é o relatório mais recente sobre vendas?")
        "Relatório de vendas do trimestre Q1 2025..."
    """

    
    embedding_model = TextEmbedding()
    query_embedding = embedding_model.embed(pergunta)
    query_embedding = list(query_embedding)

    client = QdrantClient(path="qdrant_db")
    search_result = client.query_points(
        collection_name="relatórios",
        query=query_embedding[0],
        limit=1,
    )
    return search_result.points[0].payload['source']

def funcao_agente_designer(conteudo):
      """Gera uma imagem com base na descrição fornecida.

      Args:
          conteudo (str): Conteúdo gerado pelo agente roteirista.

      Returns:
          Resultado: O conteúdo da apresentação complementado.

      Exemplo:
          >>> funcao_agente_designer(conteudo)
          "Resultado da apresentação"
      """
      agente_designer = Agent(
          model=modelo,
          system_prompt=f'''
      Você é um criador de apresentações que recebe o roteiro: {conteudo}.
      Seu trabalho é transformar esse roteiro em uma apresentação no estilo Quarto usando markdown.

      Cada slide começa com ## Título do Slide.
      Você deve:
      - Adicionar até 3 bullets, imagem ou código python
      em markdown que permita a construção de um gráfico com o que é observado nas notas.
      - Você deve escolher apenas um dos três por slide, bullets, imagens ou gráficos.
      - Manter as notas com o que será falado, usando o campo `notes:`.
      - Adicionar uma imagem em alguns dos slides. A imagem deve estar no formato markdown e estar relacionada ao número do slide.
      - Retornar a **apresentação completa** com todos os slides, incluindo aqueles com imagem gerada.

      Exemplo de chamada de imagem:
      `![Descrição de um gato que deve aparecer no slide](3.png)`

      Notas devem aparecer assim:
      ::: {{.notes}}
      Texto das notas aqui
      :::

      Não diga mais nada além do markdown final.
      '''
      )
      result = agente_designer.run_sync('Complemente a apresentação')
      Resultado = result.data
      return Resultado

agente_revisor = Agent(
          model=modelo,
          system_prompt=f'''
          Você é o agente revisor de apresentação.
          Você deve verificar o texto passado via prompt e conferir se não existem informações como
          - [Preencha com os resultados da consulta]. Nesse caso você deve substituir esse conteúdo por
          um texto que faça sentido de acordo com as notas do slide em questão.

          Quando você observar a presença de uma imagem como por exemplo:
          ![Um gato brincando com um brinquedo](3.png)
          Você deve gerar o arquivo da imagem correspondente usando a ferramenta gera_imagem.

          Ao final você deve retornar o texto corrigido.

      '''
      )

agente_artista = Agent(
          model=modelo,
          system_prompt=f'''
          Você é um agente que deve analisar o texto do prompt e retornar uma lista
das imagens em markdown (ex: ![alt](url)) que aparecerem, junto com o número do slide
em que cada imagem aparece.

Retorne no formato:
{{"imagens": [{{"slide": 1, "markdown": "![descrição](link)"}}, ...]}}

Se nenhuma imagem for encontrada, retorne {{"imagens": []}}
''',result_type=Imagens
)

def extrai_descricao(markdown: str) -> str:
    """Extrai a descrição da imagem a partir da sintaxe markdown."""
    match = re.match(r'!\[(.*?)\]\(.*?\)', markdown)
    return match.group(1) if match else "Imagem sem descrição"

def gera_imagem(descricao: str, n_slide: str) -> str:
    """Gera uma imagem com base na descrição fornecida.

    Args:
        descricao (str): A descrição ou prompt que será usado para gerar a imagem.
        n_slide (str): O número do slide onde a imagem será inserida.

    Returns:
        Image_path: O caminho onde a imagem foi salva

    Exemplo:
        >>> consulta("Gere uma imagem de um gato em ambiente de escritório")
        "slide1.png"
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
      model="gemini-2.0-flash-exp-image-generation",
      contents=descricao,
      config=types.GenerateContentConfig(
        response_modalities=['Text', 'Image']
      )
    )
    for part in response.candidates[0].content.parts:
      if part.text is not None:
        print(part.text)
      elif part.inline_data is not None:
        image = Image.open(BytesIO((part.inline_data.data)))
        image.save(n_slide+'.png')
    Image_path = n_slide+'.png'
    return Image_path


# Configuração da página
st.set_page_config(
    page_title="Gerador de Apresentações com IA",
    page_icon="📊",
    layout="wide",
)


# Título e introdução
st.title("🐱 Gerador de Apresentações com IA")
st.markdown("""
Esta aplicação gera apresentações em formato markdown a partir de documentos (PDF/CSV).
Carregue seus arquivos, configure as informações da apresentação e receba um documento pronto para usar com Quarto.
""")



if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

perguntas = [
    "Qual o resumo executivo do relatório?",
    "Qual o faturamento com rações na vendas por categoria?",
    "Qual o faturamento com arranhadores na vendas por categoria?",
    "Qual o faturamento com brinquedos na vendas por categoria?",
    "Quais produtos de previsão de reposição urgente na análise de estoque?",
    "Como está o desempenho das vendas online comparado à loja física?",
    "Quais são os diferenciais competitivos frente à concorrência?",
    "Quais ações de RH impactaram o desempenho das equipes?",
    "Os indicadores financeitos estão alinhados com as metas?",
    "Quais tendências de consumo devem ser exploradas nos próximos meses?",
    "Qual a reação mais comum do Gato?"
]

# Função para salvar arquivos temporários
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

# Upload de documentos

col1, col2 = st.columns(2)

with col1:
    pdf_file = st.file_uploader("Carregue o relatório em PDF", type=["pdf"])
    csv_file = st.file_uploader("Carregue os dados complementares (CSV)", type=["csv"])

    if st.button("Processar Documentos"):
        if pdf_file is None or csv_file is None:
            st.error("Por favor, carregue um arquivo PDF para continuar.")
        else:
            with st.spinner("Processando documentos..."):
                #Inicializar a construção da base vetorial

                # Processar PDF
                pdf_path = save_uploaded_file(pdf_file)
                converter = DocumentConverter()
                result = converter.convert(pdf_path)
                csv_path = save_uploaded_file(csv_file)
                result_csv = converter.convert(csv_path)

                # Embedding
                texto = result.document.export_to_markdown()

                # Defina o tamanho médio de uma página (ajuste conforme necessário)
                tamanho_pagina = 1800

                # Divide o texto em chunks
                chunks = {}
                for i in range(0, len(texto), tamanho_pagina):
                    chave = f"page_{(i // tamanho_pagina) + 1}"
                    chunk = texto[i:i + tamanho_pagina]
                    chunks[chave] = chunk.strip()

                document = list(chunks.values()) + [result_csv.document.export_to_markdown()]

                embedding_model = TextEmbedding()
                embeddings_generator = embedding_model.embed(document)
                embeddings_list = list(embeddings_generator)

                #Criando a base vetorial
                client = QdrantClient(path="qdrant_db")
                texto_csv = result_csv.document.export_to_markdown()
                metadata = [{"source": chunks[f'page_{i+1}']} for i in range(7)]
                metadata.append({"source": texto_csv})
                ids = list(range(8))

                points = [
                    models.PointStruct(id=id, vector=vector, payload=payload)
                    for id, (vector, payload) in zip(ids, zip(embeddings_list, metadata))
                ]

                # Criar coleção e inserir pontos
                client.create_collection(
                    collection_name="relatórios",
                    vectors_config={
                        "size": 384,
                        "distance": "Cosine"
                    }
                )

                client.upsert(
                    collection_name="relatórios",
                    wait=True,
                    points=points
                )
                client = QdrantClient(path="qdrant_db_temp")
                st.session_state.documents_processed = True

with col2:
    if st.session_state.documents_processed: 
        st.success("Documentos processados com sucesso!")

        prompt = st.text_input('Diga como você quer a apresentação')
        if st.button('Configurar a apresentação'):
            saida = agente_formatador.run_sync(prompt)
            yaml_formatado = formatar_para_yaml(saida.data)

            st.code(yaml_formatado, language='yaml')

        if st.button('Gerar apresentação'):
            roteiro_completo = ""

            # Processamento das perguntas
            for pergunta in perguntas:
                time.sleep(10)  # Adicione um atraso de 1 segundo entre as perguntas
                resultado = agente_roteirista.run_sync(pergunta)
                roteiro_completo += resultado.data + "\n\n"  # Adiciona quebra de linha entre seções

            time.sleep(10)
            Resultado = funcao_agente_designer(roteiro_completo)
            time.sleep(10)
            apresentacao_final = agente_revisor.run_sync(Resultado)
            st.code(apresentacao_final.data, language='markdown')

            result = agente_artista.run_sync(apresentacao_final.data)
            resultado = result.data

            # Criar um arquivo zip na memória
            st.session_state.imagens_geradas = []  # Armazena os caminhos

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for img in resultado.imagens:
                    descricao = extrai_descricao(img.markdown)
                    nome_arquivo = f"{img.slide}"
                    caminho_imagem = gera_imagem(descricao, nome_arquivo)
                    st.image(caminho_imagem, caption=f"Slide {img.slide}")

                    zip_file.write(caminho_imagem, arcname=Path(caminho_imagem).name)
                    st.session_state.imagens_geradas.append(caminho_imagem)

            zip_buffer.seek(0)
            st.download_button(
                label="Baixar todas as imagens (.zip)",
                data=zip_buffer,
                file_name="slides.zip",
                mime="application/zip"
            )