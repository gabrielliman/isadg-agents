import os
import json
import base64
import fitz
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
# LangChain & NVIDIA NIM
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# NOTE: the system prompts, the JSON schemas requested from the models and the
# default division labels are kept in Portuguese on purpose: the corpus is in
# (19th-century) Portuguese and these are the exact prompts used in the paper.

# ---------------------------------------------------------
# 1. Configuration, Global Variables and Logging
# ---------------------------------------------------------
# Credentials are read from the environment (see .env.example)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise SystemExit("Set the NVIDIA_API_KEY environment variable (see .env.example).")

# Paths relative to the repository root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT, "data", "editions")
RESULTS_DIR = os.path.join(ROOT, "results")
ISADG_DIR = os.path.join(RESULTS_DIR, "isadg_records")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "transcription_checkpoints")
RAW_DIR = os.path.join(RESULTS_DIR, "raw_texts")
NORM_DIR = os.path.join(RESULTS_DIR, "normalized_texts")
VECTOR_DB_DIR = os.path.join(RESULTS_DIR, "vector_db")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
MAX_BUFFER_SIZE = 40000

# Create output folders
for d in [ISADG_DIR, CHECKPOINT_DIR, RAW_DIR, NORM_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# Logging and metrics setup
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO, # INFO captures latency metrics, ERROR captures failures
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Avoid flooding the log with every HTTP request
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------
# 2. Agent Initialization (NVIDIA NIM)
# ---------------------------------------------------------
print("Initializing agents and models...")

llm_vision = ChatNVIDIA(
    model="meta/llama-3.2-90b-vision-instruct",
    api_key=NVIDIA_API_KEY, temperature=0.1, timeout=600, max_completion_tokens=2048
).with_retry(stop_after_attempt=3)

llm_structurer = ChatNVIDIA(
    model="meta/llama-4-maverick-17b-128e-instruct",
    api_key=NVIDIA_API_KEY, temperature=0.0, timeout=300, max_completion_tokens=4096
).with_retry(stop_after_attempt=5)

llm_normalizer = ChatNVIDIA(
    model="qwen/qwen3-next-80b-a3b-instruct",
    api_key=NVIDIA_API_KEY, temperature=0.1, timeout=300, max_completion_tokens=4096
).with_retry(stop_after_attempt=5)

llm_archivist = ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    api_key=NVIDIA_API_KEY, temperature=0.1, timeout=300, max_completion_tokens=2048
).with_retry(stop_after_attempt=5)

json_parser = JsonOutputParser()

# ---------------------------------------------------------
# 3. Vector Store Initialization (ChromaDB)
# ---------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings,
    collection_name="acervo_bndigital"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
)

# ---------------------------------------------------------
# 4. System Prompts
# ---------------------------------------------------------
structurer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um analisador de layout de periódicos antigos. Sua tarefa é avaliar a página transcrita e decidir se uma NOVA divisão lógica começa nela. "
     "Isso inclui: seções, capítulos, anexos e estudos\n"
     "Títulos geralmente estão em letras maiúsculas, centralizados, ou compõem a maior parte da página.\n"
     "Retorne EXCLUSIVAMENTE um objeto JSON válido. Use aspas duplas.\n"
     "{{\n"
     "  \"contem_novo_artigo\": true/false,\n"
     "  \"titulo_artigo\": \"Nome da divisão/capítulo se houver, ou null\",\n"
     "  \"autor\": \"Nome do autor se houver, ou null\"\n"
     "}}\n"
     "{format_instructions}"),

    # Example 1: page 8
    ("user", "PÁGINA TRANSCRITA:\nESTUDO CHIMICO E GEOLOGICO\n\nDAS\n\nROCHAS DO CENTRO DA PROVINCIA DE MINAS GERAES\n\nPRIMEIRA PARTE\n\nArredores de Ouro Preto\n\nA maior parte do centro da provincia de Minas Geraes é constituido especialmente pela grande cadeia..."),
    ("assistant", "{{\n  \"contem_novo_artigo\": true,\n  \"titulo_artigo\": \"ESTUDO CHIMICO E GEOLOGICO DAS ROCHAS DO CENTRO DA PROVINCIA DE MINAS GERAES\",\n  \"autor\": null\n}}"),

    # Example 2: page 42
    ("user", "PÁGINA TRANSCRITA:\nEXPLORAÇÃO\n\nDAS\n\nMINAS DE GALENA DO RIBEIRÃO DO CHUMBO\n\nAFFLUENTE DO ABAETÉ\n\nE ESTUDO DA ZONA PERCORRIDA DE OURO PRETO ATÉ ESSE LOGAR\n\nAspecto geral do terreno – sua divisão\n\nA zona compreendida entre a cidade de Ouro Preto..."),
    ("assistant", "{{\n  \"contem_novo_artigo\": true,\n  \"titulo_artigo\": \"EXPLORAÇÃO DAS MINAS DE GALENA DO RIBEIRÃO DO CHUMBO AFFLUENTE DO ABAETÉ\",\n  \"autor\": null\n}}"),

    # Example 3: page 158
    ("user", "PÁGINA TRANSCRITA:\nESTATISTICA\n\nDA\n\nProdução do ouro na Província de Minas Geraes no anno de 1879\n\nE' facil estabelecer-se essa estatistica para as grandes companhias de mineração da provincia..."),
    ("assistant", "{{\n  \"contem_novo_artigo\": true,\n  \"titulo_artigo\": \"ESTATISTICA DA Produção do ouro na Província de Minas Geraes no anno de 1879\",\n  \"autor\": null\n}}"),

    # Example 4: generic continuation page (negative)
    ("user", "PÁGINA TRANSCRITA:\nEmfim de pequenos depositos de rochas anormaes, contendo turmalinas, disthenio e outros mineraes, cujo conjuncto é muito menos importante do que as primeiras.\nO estudo detalhado das rochas schistosas, que formam a maior parte da serra do Espinhaço..."),
    ("assistant", "{{\n  \"contem_novo_artigo\": false,\n  \"titulo_artigo\": null,\n  \"autor\": null\n}}"),

    # The actual call
    ("user", "PÁGINA TRANSCRITA:\n{texto_pagina}")
]).partial(format_instructions=json_parser.get_format_instructions())

normalizer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Sua tarefa é normalizar a ortografia de um artigo histórico para o português contemporâneo. "
     "REGRA 1: Preserve EXATAMENTE a estrutura de parágrafos, o tamanho e o conteúdo do texto original. É estritamente PROIBIDO resumir, omitir trechos ou reescrever frases.\n"
     "REGRA 2: Se encontrar marcações como [TABELA: ...] ou [IMAGEM: ...], preserve-as intactas.\n"
     "Retorne APENAS o texto normalizado, sem markdown e sem comentários adicionais."),
    ("user", "Texto bruto do OCR:\n{texto_bruto}")
])

archivist_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Arquivista especialista na norma ISAD(G). "
     "REGRA ABSOLUTA: Você deve retornar ÚNICA E EXCLUSIVAMENTE um objeto JSON válido. "
     "NÃO escreva nenhum processo de pensamento, justificativa, introdução ou conclusão. NÃO pense em voz alta. "
     "O seu output DEVE começar obrigatoriamente com a chave '{{' e terminar com a chave '}}'.\n\n"
     "Esquema obrigatório:\n"
     "{{\n"
     "  \"area_identificacao\": {{\"titulo\": \"\", \"data\": \"\"}},\n"
     "  \"area_contextualizacao\": {{\"nome_produtor\": \"\"}},\n"
     "  \"area_conteudo_estrutura\": {{\"ambito_e_conteudo\": \"\"}},\n"
     "  \"area_pontos_acesso\": {{\"pessoas\": [], \"lugares\": []}}\n"
     "}}\n"
     "Preencha com 'Não identificado' caso não exista. Resolva ambiguidades usando o CONTEXTO HISTÓRICO.\n"
     "{format_instructions}"),
    ("user", "CONTEXTO HISTÓRICO RECUPERADO (RAG):\n{contexto_historico}\n\nTEXTO NORMALIZADO:\n{texto_normalizado}")
]).partial(format_instructions=json_parser.get_format_instructions())

structurer_chain = structurer_prompt | llm_structurer | json_parser
normalizer_chain = normalizer_prompt | llm_normalizer | StrOutputParser()
archivist_chain = archivist_prompt | llm_archivist | json_parser

# ---------------------------------------------------------
# 5. Helper Functions
# ---------------------------------------------------------

def invoke_with_retry(chain, params, max_attempts=5, task_name="Agent"):
    """Wraps chain execution in a retry loop for parsing errors and hallucinations."""
    for attempt in range(1, max_attempts + 1):
        try:
            return chain.invoke(params)
        except Exception as e:
            # Keep only the first 200 characters of the error to avoid flooding the log with tokens
            error_msg = str(e).replace('\n', ' ')[:200]

            if attempt < max_attempts:
                logging.warning(f"[Retry {attempt}/{max_attempts}] {task_name} failed: {error_msg}. Retrying in 3s...")
                print(f"      ! {task_name} failed (attempt {attempt}/{max_attempts}). Retrying...")
                time.sleep(3)
            else:
                logging.error(f"[Retries exhausted] {task_name} failed after {max_attempts} attempts: {error_msg}")
                raise e # Propagate so the main try/except skips the article

def vision_api_call_thread(page_num, b64_image):
    t_start = time.time()

    vision_instruction = (
        "Transcreva fielmente o texto desta imagem. "
        "REGRAS ESTRITAS: IGNORE completamente o conteúdo interno de tabelas, quadros numéricos e imagens/ilustrações. "
        "Onde houver uma tabela ou imagem, NÃO tente transcrever os dados. Substitua todo o elemento escrevendo apenas a tag correspondente: "
        "[TABELA: título ou descrição] ou [IMAGEM: descrição]. "
        "Responda APENAS com a transcrição bruta e as tags, sem introduções."
    )

    msg = HumanMessage(content=[
        {"type": "text", "text": vision_instruction},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
    ])

    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            response = llm_vision.invoke([msg]).content

            # Check for an empty response
            if not response.strip():
                logging.warning(f"Vision [Page {page_num}]: empty response on attempt {attempt+1}. Retrying...")
                time.sleep(2) # Short pause to avoid rate limiting
                continue

            t_end = time.time()
            logging.info(f"Latency Vision [Page {page_num}]: {t_end - t_start:.2f}s")
            return page_num, response

        except Exception as e:
            logging.error(f"Vision error [Page {page_num}]: {e}")

    # Both attempts failed: return empty text (fixed later by self-healing)
    return page_num, ""

def process_finished_article(raw_article_text, article_metadata, pdf_name):
    if len(raw_article_text.strip()) < 50:
        return

    start_page = article_metadata.get("pagina_inicio", "?")
    end_page = article_metadata.get("pagina_fim", "?")
    safe_title = str(article_metadata.get('titulo_artigo', f'artigo_pag_{start_page}')).replace(" ", "_").replace("/", "")[:50]
    base_filename = f"{pdf_name}_{safe_title}"

    # ARTIFACT: raw text (VLM)
    raw_path = os.path.join(RAW_DIR, f"{base_filename}_raw.txt")
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(raw_article_text)

    print(f"      -> Normalizing article: {safe_title}...")
    t_start_norm = time.time()
    try:
        clean_text = invoke_with_retry(
            chain=normalizer_chain,
            params={"texto_bruto": raw_article_text[:MAX_BUFFER_SIZE]},
            task_name="Normalizer"
        )
        logging.info(f"Latency Normalization [{safe_title}]: {time.time() - t_start_norm:.2f}s")

        # ARTIFACT: normalized text
        norm_path = os.path.join(NORM_DIR, f"{base_filename}_norm.txt")
        with open(norm_path, 'w', encoding='utf-8') as f:
            f.write(clean_text)

    except Exception as e:
        logging.error(f"Normalization failed for {safe_title}: {e}")
        clean_text = raw_article_text

    print("      -> Adding normalized article to vector memory (RAG)...")
    chunks = text_splitter.split_text(clean_text)
    if chunks:
        ids = [f"{base_filename}_{i}" for i in range(len(chunks))]
        meta = [{"origem": pdf_name, "titulo": safe_title, "autor": str(article_metadata.get('autor'))} for _ in chunks]
        vectorstore.add_texts(texts=chunks, metadatas=meta, ids=ids)

    print("      -> Generating ISAD(G) record...")
    retrieved_docs = retriever.invoke(clean_text)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    t_start_isadg = time.time()
    try:
        isadg_record = invoke_with_retry(
            chain=archivist_chain,
            params={"contexto_historico": context, "texto_normalizado": clean_text},
            task_name="Archivist"
        )

        # DETERMINISTIC INJECTION (page range and reference code)
        isadg_record["pagina_inicio"] = start_page
        isadg_record["pagina_fim"] = end_page

        # Simulated reference code: country + institution + file
        reference_code = f"BR MG-EMOP {base_filename}"

        # Guard against the LLM not producing the nested dictionary
        if "area_identificacao" not in isadg_record:
            isadg_record["area_identificacao"] = {}

        # Force the reference code into the first position of the identity area
        isadg_record["area_identificacao"] = {
            "codigo_referencia": reference_code,
            **isadg_record["area_identificacao"]
        }

        logging.info(f"Latency ISAD(G) [{safe_title}]: {time.time() - t_start_isadg:.2f}s")

        json_path = os.path.join(ISADG_DIR, f"{base_filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(isadg_record, f, indent=4, ensure_ascii=False)
        print(f"      -> Success! Saved to: {json_path}")
    except Exception as e:
        logging.error(f"Error generating ISAD(G) for {safe_title}: {e}")
        print(f"      -> Error generating ISAD(G). Check the log file.")

# ---------------------------------------------------------
# 6. Orchestration with Parallelism and Checkpoints
# ---------------------------------------------------------
def render_pages_b64(doc, page_nums):
    """Renders the given pages (1-based) as base64-encoded PNGs at 200 dpi."""
    b64_dict = {}
    for num in page_nums:
        pix = doc.load_page(num - 1).get_pixmap(dpi=200)
        b64_dict[num] = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    return b64_dict

def run_vision(b64_dict, page_texts, checkpoint_path, label):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(vision_api_call_thread, num, b64) for num, b64 in b64_dict.items()]
        for future in as_completed(futures):
            num, text = future.result()
            page_texts[num] = text
            print(f"    - {label}: Page {num}")

            # IMMEDIATE SAVE: update the checkpoint after every page
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(page_texts, f, ensure_ascii=False, indent=4)

def process_pdf_edition(pdf_path, pdf_name):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{pdf_name}_transcribed.json")
    page_texts = {}

    # 1. Visual extraction phase (checkpoint, parallelism and self-healing)
    if os.path.exists(checkpoint_path):
        print("  -> Loading transcriptions from checkpoint...")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            page_texts = {int(k): v for k, v in json.load(f).items()}

        # Self-healing: find empty pages or pages with fewer than 10 characters (noise)
        failed_pages = [num for num, text in page_texts.items() if len(text.strip()) < 10]

        if failed_pages:
            print(f"  -> Found {len(failed_pages)} empty/failed pages in the checkpoint. Re-extracting...")
            doc = fitz.open(pdf_path)
            b64_dict = render_pages_b64(doc, failed_pages)
            doc.close()
            run_vision(b64_dict, page_texts, checkpoint_path, "Vision done (self-healing)")
    else:
        print("  -> Starting parallel visual extraction of all pages...")
        doc = fitz.open(pdf_path)
        b64_dict = render_pages_b64(doc, range(1, len(doc) + 1))
        doc.close()
        run_vision(b64_dict, page_texts, checkpoint_path, "Vision done")

    # 2. Agentic routing phase
    print("\n  -> Starting agentic routing and cataloging...")
    active_buffer = ""
    active_metadata = None  # Initialized on the first valid page

    sorted_pages = sorted(page_texts.keys())

    for page_num in sorted_pages:
        page_text = page_texts[page_num]
        print(f"  [Router - Page {page_num}] Analyzing semantic structure...")

        # Skip fully blank pages to save tokens
        if len(page_text.strip()) < 20:
            active_buffer += "\n" + page_text
            continue

        t_start_router = time.time()
        try:
            analysis = invoke_with_retry(
                chain=structurer_chain,
                params={"texto_pagina": page_text},
                task_name=f"Structurer [Page {page_num}]"
            )
            logging.info(f"Latency Structurer [Page {page_num}]: {time.time() - t_start_router:.2f}s")

            # --- FIRST VALID PAGE ---
            if active_metadata is None:
                title = analysis.get("titulo_artigo")
                active_metadata = {
                    "titulo_artigo": title if title else "Elementos_Pre_Textuais",
                    "autor": analysis.get("autor") or "Não identificado",
                    "pagina_inicio": page_num
                }
                active_buffer = page_text
                print(f"  [Document start] Initial block set to: {active_metadata['titulo_artigo']}")
                continue

            # --- REMAINING PAGES ---
            if analysis.get("contem_novo_artigo"):
                found_title = analysis.get("titulo_artigo") or "Divisao_sem_Titulo"
                print(f"  [Split] New division detected: {found_title}")

                # 1. Close the previous article on the previous page (page_num - 1)
                active_metadata["pagina_fim"] = page_num - 1
                process_finished_article(active_buffer, active_metadata, pdf_name)

                # 2. Start the new article on the current page
                active_buffer = page_text
                active_metadata = {
                    "titulo_artigo": found_title,
                    "autor": analysis.get("autor") or "Não identificado",
                    "pagina_inicio": page_num
                }
            else:
                # Not a new article: just accumulate the text
                active_buffer += "\n" + page_text

        except Exception as e:
            logging.error(f"Structurer error [Page {page_num}]: {e}")
            active_buffer += "\n" + page_text

            # Safety fallback in case the request fails on the very first page
            if active_metadata is None:
                active_metadata = {
                    "titulo_artigo": "Elementos_Pre_Textuais",
                    "autor": "Não identificado",
                    "pagina_inicio": page_num
                }

    # Flush whatever is left in the buffer at the end of the PDF
    if active_buffer.strip() and active_metadata is not None:
        active_metadata["pagina_fim"] = sorted_pages[-1] if sorted_pages else 1
        process_finished_article(active_buffer, active_metadata, pdf_name)

def process_collection():
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}.")
        return

    print(f"\nStarting the pipeline for {len(pdf_files)} editions...\n")
    for pdf_name in sorted(pdf_files):
        print(f"\n==================================================")
        print(f"Processing edition: {pdf_name}")
        print(f"==================================================")
        process_pdf_edition(os.path.join(INPUT_DIR, pdf_name), pdf_name.replace(".pdf", ""))

if __name__ == "__main__":
    process_collection()
