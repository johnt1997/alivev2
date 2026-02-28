
from packages.globals import (
    EMBEDDINGS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEARCH_KWARGS_NUM,
    TEMPERATURE,
    MODEL_NAME,
)

import os
import json
from flask import Flask, request, jsonify, abort, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import openai

# Importiere deine neuen, überarbeiteten Klassen
from packages.document_processing import DocumentProcessing, SplitterType
from packages.vector_store_handler import VectorStoreHandler, HybridRetriever
from packages.llm_config import LLMConfig
from packages.bm25_retriever import BM25Retriever
from packages.init_chain import InitializeQuesionAnsweringChain
from packages.person import Person
from langchain.schema.document import Document
from packages.globals import EMBEDDINGS, CHUNK_SIZE, CHUNK_OVERLAP, TEMPERATURE, SEARCH_KWARGS_NUM
from helper_functions import clear_dir, delete_dir, string_to_bool, get_document_name
from typing import List, Tuple, Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Cache für geladene Chains – verhindert Neu-Laden bei jedem Request
_chain_cache: Dict = {}



MIN_SENT_CHUNK = 200  # wie im Frontend, nur für sentence_transformer

def normalize_chunk_params(splitter: str, chunk_size: int, chunk_overlap: int) -> tuple[int, int]:
    # Defaults
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        chunk_size = CHUNK_SIZE
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        chunk_overlap = CHUNK_OVERLAP

    # 🔒 semantic: vollständig automatisch, pfadstabil via (0,0)
    if splitter == "semantic":
        if (chunk_size, chunk_overlap) != (0, 0):
            print(f"[INFO] normalize (semantic): forcing chunk_size=0, overlap=0 (auto mode)")
        return 0, 0

    # sentence_transformer: Frontend hat bereits ÷3 gemacht → nur Mindestgröße prüfen
    if splitter == "sentence_transformer":
        eff = max(MIN_SENT_CHUNK, chunk_size)
        if eff != chunk_size:
            print(f"[INFO] normalize (sentence_transformer): {chunk_size} -> {eff} (MIN enforced)")
        chunk_size = eff

    # Overlap-Guard
    if chunk_overlap >= chunk_size:
        new_overlap = max(0, chunk_size // 2)
        print(f"[WARN] chunk_overlap >= chunk_size; adjusting {chunk_overlap} -> {new_overlap}")
        chunk_overlap = new_overlap

    return chunk_size, chunk_overlap


@app.route("/has-openai-api-key", methods=["GET"])
def has_openai_api_key():
    try:
        openai_key = os.environ["OPENAI_API_KEY"]
        if openai_key == "":
            return (
                jsonify(False),
                200,
                {"Access-Control-Allow-Origin": "http://localhost:3000"},
            )

        openai.api_key = openai_key
        return (
            jsonify(True),
            200,
            {"Access-Control-Allow-Origin": "http://localhost:3000"},
        )
    except KeyError:
        return (
            jsonify(False),
            200,
            {"Access-Control-Allow-Origin": "http://localhost:3000"},
        )

def _get_embedding_instance(use_openai: bool):
    """Gibt die korrekte Embedding-Instanz zurück."""
    if use_openai:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            abort(500, "OpenAI API key not found in .env file.")
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(openai_api_key=openai_api_key)
        except ImportError:
            from langchain_community.embeddings import OpenAIEmbeddings
            return OpenAIEmbeddings(openai_api_key=openai_api_key)
    return EMBEDDINGS

def _get_qa_chain_and_person(params: Dict):
    """
    Kapselt die gesamte Logik zum Laden oder Erstellen der Indizes
    und zur Initialisierung der QA-Chain.
    Ergebnisse werden gecacht – bei identischer Konfiguration wird die
    Chain nicht neu gebaut.
    """
    cache_key = (
        params["person_name"],
        params["splitter_type"],
        params["chunk_size"],
        params["chunk_overlap"],
        params["retrieval_mode"],
        params["use_reranker"],
        params["use_openai"],
        params["search_kwargs_num"],
        params["temperature"],
        params["language"],
        params["eyewitness_mode"],
        params["llm_model"],
    )
    if cache_key in _chain_cache:
        print(f"[CACHE] Returning cached chain for key: {cache_key}")
        return _chain_cache[cache_key]

    person = Person(name=params["person_name"], voice=params["voice"])
    embeddings_instance = _get_embedding_instance(params["use_openai"])
    
    doc_processor = DocumentProcessing(embeddings=embeddings_instance)
    vectorstore_handler = VectorStoreHandler(embeddings=embeddings_instance, search_kwargs_num=params["search_kwargs_num"])

    # Versuche, die DB zu laden.
    retriever, db = vectorstore_handler.get_db_and_retriever(
        vector_store_name=person.name,
        splitter_type=params["splitter_type"],
        chunk_size=params["chunk_size"],
        chunk_overlap=params["chunk_overlap"],
    )

    # Wenn DB nicht existiert, erstelle sie neu.
    if not db:
        print(f"[INFO] Vector store for config '{params['splitter_type']}' not found. Creating new one...")
        split_documents = doc_processor.get_chunked_documents(
            directory_path=f"./data/{person.name}",
            splitter_type=params["splitter_type"],
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
        )
        if not split_documents:
            abort(500, f"No documents could be processed for person '{person.name}'.")

        _, db = vectorstore_handler.create_db_and_retriever(
            chunked_documents=split_documents,
            vector_store_name=person.name,
            splitter_type=params["splitter_type"],
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
        )
        if not db:
            abort(500, f"Failed to create vector store for person '{person.name}'.")

    # Erstelle den passenden Retriever basierend auf dem Modus.
    if params["retrieval_mode"] == 'hybrid':
        print("[INFO] Using HYBRID Retriever.")
        bm25_index_dir = f"./bm25_indexes/{person.name}_{params['splitter_type']}"
        if not os.path.isdir(bm25_index_dir):
             abort(500, f"BM25 index not found at {bm25_index_dir}. Please pre-build it.")
        bm25_retriever = BM25Retriever(bm25_index_dir)
        retriever = HybridRetriever(db=db, bm25_retriever=bm25_retriever, k=params["search_kwargs_num"])
    else: # Default ist 'dense'
        print("[INFO] Using DENSE Retriever.")
        retriever = db.as_retriever(search_kwargs={"k": params["search_kwargs_num"]})
    
    # Initialisiere die QA-Chain.
    llm_config = LLMConfig(temperature=params["temperature"], model_name=params["llm_model"])
    llm = llm_config.get_local_llm(use_openai=params["use_openai"])
    
    qa_chain = InitializeQuesionAnsweringChain(
        llm=llm,
        retriever=retriever,
        db=db,
        person=person,
        search_kwargs_num=params["search_kwargs_num"],
        language=params["language"],
        use_reranker=params["use_reranker"],
        eyewitness_mode=params["eyewitness_mode"]
    )
    
    _chain_cache[cache_key] = (qa_chain, person)
    return qa_chain, person

@app.route("/upload-person-data", methods=["POST"])
def upload_person_data():
    name = request.args.get("name", default="", type=str)
    if not name:
        abort(400, "No name specified")
    
    files = request.files.getlist("file")
    if not files:
        abort(400, "No file part")

    # Alten Daten-Ordner leeren, um Konsistenz zu wahren.
    # Die zugehörigen Indizes sollten separat via /delete-person gelöscht werden.
    data_path = f"./data/{name}"
    clear_dir(data_path)
    os.makedirs(data_path, exist_ok=True)

    for file in files:
        if file.filename and (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
            file.save(os.path.join(data_path, file.filename))
        else:
            abort(400, "Invalid file type. Only PDF and TXT files are allowed.")
            
    return jsonify({"message": f"Data for {name} uploaded. Please delete old indices if necessary."}), 200

@app.route("/upload-person-voice", methods=["POST"])
def upload_person_voice():
    if "file" not in request.files:
        abort(400, "No file part")

    name = request.args.get("name", default="", type=str)
    if name == "":
        abort(400, "No name specified")

    file = request.files["file"]

    if file.filename == "":
        abort(400, "No selected file")

    if not (file.filename.endswith(".wav")):
        abort(400, "Only .wav")

    # Check if person already exists and delete it
    # this is easier than constantly updating the vector store(s)
    # clear_dir(f"./data/{name}")
    # clear_dir("./database/" + OpenAIEmbeddings.__name__ + "_" + name)
    # clear_dir("./database/" + HuggingFaceEmbeddings.__name__ + "_" + name)

    if not os.path.exists(f"./voices/{name}"):
        os.makedirs(f"./voices/{name}")

    file.save(f"./voices/{name}/" + name + ".wav")

    return (
        jsonify("File succesfully uploaded"),
        200,
        {"Access-Control-Allow-Origin": "http://localhost:3000"},
    )

@app.route("/upload-background", methods=["POST"])
def upload_background():
    if "file" not in request.files:
        abort(400, "No file part")

    name = request.args.get("name", default="", type=str)
    if name == "":
        abort(400, "No name specified")

    file = request.files["file"]

    if file.filename == "":
        abort(400, "No selected file")

    file_name = file.filename.lower()
    if not (file_name.endswith(".jpg") or file_name.endswith(".jpeg")):
        abort(400, "Only .jpg")

    if not os.path.exists(f"./backgrounds/{name}"):
        os.makedirs(f"./backgrounds/{name}")

    file.save(f"./backgrounds/{name}/" + name + ".jpg")

    return (
        jsonify("File succesfully uploaded"),
        200,
        {"Access-Control-Allow-Origin": "http://localhost:3000"},
    )

@app.route("/get-background", methods=["GET"])
def get_background():
    person_name = request.args.get("person", default="", type=str)

    if person_name == "":
        abort(400, "No person specified")

    file_path = f"./backgrounds/{person_name}/{person_name}.jpg"
    if os.path.exists(file_path):
        # mimetype = "glb"
        return send_file(file_path, mimetype=None)

    else:
        return (
            jsonify("No File Found"),
            200,
            {"Access-Control-Allow-Origin": "http://localhost:3000"},
        )

@app.route("/upload-virtual-character", methods=["POST"])
def upload_virtual_character():
    if "file" not in request.files:
        abort(400, "No file part")

    name = request.args.get("name", default="", type=str)
    if name == "":
        abort(400, "No name specified")

    file = request.files["file"]

    if file.filename == "":
        abort(400, "No selected file")

    if not (file.filename.endswith(".glb")):
        abort(400, "Only .glb allowed")

    if not os.path.exists(f"./virtual_characters/{name}"):
        os.makedirs(f"./virtual_characters/{name}")

    file.save(f"./virtual_characters/{name}/" + name + ".glb")

    return (
        jsonify("File succesfully uploaded"),
        200,
        {"Access-Control-Allow-Origin": "http://localhost:3000"},
    )

@app.route("/get-virtual-character", methods=["GET"])
def get_virtual_character():
    person_name = request.args.get("person", default="", type=str)

    if person_name == "":
        abort(400, "No person specified")

    file_path = f"./virtual_characters/{person_name}/{person_name}.glb"
    if os.path.exists(file_path):
        # mimetype = "glb"
        return send_file(file_path, mimetype=None)

    else:
        return (
            jsonify("No File Found"),
            200,
            {"Access-Control-Allow-Origin": "http://localhost:3000"},
        )

@app.route("/get-audio-response", methods=["GET"])
def get_audio_response():
    person_name = request.args.get("person", default="", type=str)
    chat_index = request.args.get("chat_index", default=None, type=str)
    text = request.args.get("text", default=None, type=str)
    voice = request.args.get("voice", default="Male", type=str)
    language = request.args.get("language", default="en", type=str)

    if person_name == "":
        abort(400, "No person specified")
    if chat_index is None:
        abort(400, "No chat index specified")
    if text is None:
        abort(400, "No text specified")

    person = Person(name=person_name, voice=voice)

    wav_file_path = f"./tts-outputs/{person.name}/{language + voice + chat_index}.wav"
    if os.path.exists(wav_file_path):
        mimetype = "audio/wav"
        return send_file(wav_file_path, mimetype=mimetype)
    else:
        new_file_path = person.text_to_speach(
            text=text, index=chat_index, lang=language
        )
        if new_file_path is not None:
            mimetype = "audio/wav"
            return send_file(new_file_path, mimetype=mimetype)
        else:
            abort(400, "Something went wrong")

@app.route("/person-has-voice", methods=["GET"])
def person_has_voice():
    person_name = request.args.get("person", default="", type=str)
    if person_name == "":
        abort(400, "No person specified")

    if os.path.exists(f"./voices/{person_name}/{person_name}.wav"):
        return (
            jsonify(True),
            200,
            {"Access-Control-Allow-Origin": "http://localhost:3000"},
        )
    else:
        return (
            jsonify(False),
            200,
            {"Access-Control-Allow-Origin": "http://localhost:3000"},
        )
    
@app.route("/create-person", methods=["POST"])
def create_person():
    """
    This route creates the json file for a person, adding the name (obligaotry) and optionally the personality.
    The Person class creates the json file.
    """
    data = json.loads(request.data)
    name = ""
    personality = None
    try:
        name = data["name"]
    except KeyError:
        abort(400, "Name can't be empty!")

    try:
        personality = data["personality"]
    except:
        pass

    # Delete old person if ever existed
    file_path = f"./persons/{name}.json"
    if os.path.exists(file_path):
        # Delete the file
        os.remove(file_path)

    Person(name=name, personality=personality)

    return (
        jsonify("Succesfully created person"),
        200,
        {"Access-Control-Allow-Origin": "http://localhost:3000"},
    )

@app.route("/get-person-data", methods=["GET"])
def get_person_data():
    person_name = request.args.get("person", default="", type=str)
    if person_name == "":
        abort(400, "No person specified")

    file_path = f"./persons/{person_name}.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
            return jsonify(data)
    else:
        return (
            jsonify("No data found"),
            400,
            {"Access-Control-Allow-Origin": "http://localhost:3000"},
        )

@app.route("/get-person-voice", methods=["GET"])
def get_person_voice():
    person_name = request.args.get("person", default="", type=str)
    if person_name == "":
        abort(400, "No person specified")

    file_path = f"./voices/{person_name}/{person_name}.wav"
    if os.path.exists(file_path):
        mimetype = "audio/wav"
        return send_file(file_path, mimetype=mimetype)

    else:
        return (
            jsonify("No voice found"),
            200,
            {"Access-Control-Allow-Origin": "http://localhost:3000"},
        )

@app.route("/delete-person", methods=["DELETE"])
def delete_person():
    person_name = request.args.get("person", default="", type=str)
    if not person_name:
        abort(400, "No person specified")

    print(f"[INFO] Deleting all data and indices for person: {person_name}")
    
    # Daten, JSON, und andere Assets löschen
    delete_dir(f"./data/{person_name}")
    delete_dir(f"./voices/{person_name}")
    delete_dir(f"./backgrounds/{person_name}")
    delete_dir(f"./virtual_characters/{person_name}")
    if os.path.exists(f"./persons/{person_name}.json"):
        os.remove(f"./persons/{person_name}.json")

    # Alle möglichen Vektor- und BM25-Indizes löschen
    vectorstore_handler = VectorStoreHandler()
    configs_to_delete = [
        {"splitter": "semantic", "size": 0, "overlap": 0},
        {"splitter": "sentence_transformer", "size": 256, "overlap": 30},
        {"splitter": "recursive", "size": 500, "overlap": 50}
    ]

    for config in configs_to_delete:
        # Vektor-Indizes löschen
        vectorstore_handler.delete_db(
            vector_store_name=person_name, # name wird im Pfad nicht mehr verwendet, aber als Argument behalten
            splitter_type=config["splitter"],
            chunk_size=config["size"],
            chunk_overlap=config["overlap"]
        )
        # BM25-Indizes löschen
        delete_dir(f"./bm25_indexes/{person_name}_{config['splitter']}")
        delete_dir(f"./bm25_jsonl/{person_name}_{config['splitter']}")

    return jsonify({"message": f"Successfully deleted all data and indices for {person_name}"}), 200

@app.route("/ask-question", methods=["GET"])
def ask_question():
    # Parameter aus dem Request holen
    params = {
        "person_name": request.args.get("person", default="", type=str),
        "question": request.args.get("question", default="", type=str),
        "voice": request.args.get("voice", default="Male", type=str),
        "chunk_size": request.args.get("chunk_size", default=-1, type=int),
        "chunk_overlap": request.args.get("chunk_overlap", default=-1, type=int),
        "temperature": request.args.get("temperature", default=TEMPERATURE, type=float),
        "search_kwargs_num": request.args.get("search_kwargs_num", default=SEARCH_KWARGS_NUM, type=int),
        "use_openai": string_to_bool(request.args.get("use_openai", default="False", type=str)),
        "language": request.args.get("language", default="en", type=str),
        "retrieval_mode": request.args.get("retrieval_mode", default="dense", type=str),
        "use_reranker": string_to_bool(request.args.get("use_reranker", default="False", type=str)),
        "splitter_type": request.args.get("splitter_type", default="recursive", type=str),
        "eyewitness_mode": string_to_bool(request.args.get("eyewitness_mode", default="True", type=str)),
        "previous_question": request.args.get("previous_question", default="", type=str),
        "previous_answer": request.args.get("previous_answer", default="", type=str),
        "llm_model": request.args.get("llm_model", default=MODEL_NAME, type=str),
    }
    print(params["use_reranker"])
    if not params["person_name"] or not params["question"]:
        abort(400, "Person and question must be specified.")

    # Validierung des Splitter-Typs
    allowed_splitters = ["recursive", "sentence_transformer", "semantic"]
    if params["splitter_type"] not in allowed_splitters:
        print(f"[WARN] Invalid splitter type '{params['splitter_type']}', defaulting to 'semantic'.")
        params["splitter_type"] = "semantic"

    # 🔒 EINHEITLICHE SERVERSEITIGE NORMALISIERUNG
    params["chunk_size"], params["chunk_overlap"] = normalize_chunk_params(
        params["splitter_type"], params["chunk_size"], params["chunk_overlap"]
    )

    print(f"[INFO] Request for '{params['person_name']}', splitter='{params['splitter_type']}', reranker={params['use_reranker']}")

    # QA-Chain über die Hilfsfunktion holen
    qa_chain, _ = _get_qa_chain_and_person(params)

    # Korrekte `answer`-Methode aufrufen
    answer, final_docs, metadata = qa_chain.answer(
        query=params["question"],
        previous_question=params["previous_question"],
        previous_answer=params["previous_answer"],
    )

    # Antwort aufbereiten und zurückgeben
    response_data = {
        "answer": answer,
        "metadata": metadata, # Enthält jetzt factuality_score, k_init, etc.
        "relevant_docs": process_relevant_docs(final_docs),
    }

    return jsonify(response_data), 200

@app.route("/get-all-persons", methods=["GET"])
def get_all_persons():
    directory_path = "./persons"
    json_files = [f for f in os.listdir(directory_path) if f.endswith(".json")]
    persons = []
    # Open and read each JSON file
    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        with open(file_path, "r", encoding='utf-8') as file:
            data = json.load(file)
            if "name" in data:
                persons.append(data["name"])
            else:
                print(f"[WARN] 'name' key not found in {json_file}")

    return (
        jsonify(sorted(persons)),
        200,
        {"Access-Control-Allow-Origin": "http://localhost:3000"},
    )

@app.route("/get-person-documents-names", methods=["GET"])
def get_person_documents_names():
    person_name = request.args.get("person", default="", type=str)
    if person_name == "":
        abort(400, "No person specified")

    directory_path = f"./data/{person_name}"
    if not os.path.exists(directory_path):
        abort(400, "No data found")

    files = [
        f
        for f in os.listdir(directory_path)
        if f.endswith(".pdf") or f.endswith(".txt")
    ]
    return (
        jsonify(files),
        200,
        {"Access-Control-Allow-Origin": "http://localhost:3000"},
    )

@app.route("/get-document", methods=["GET"])
def get_document():
    person_name = request.args.get("person", default="", type=str)
    document_name = request.args.get("document", default="", type=str)
    if person_name == "":
        abort(400, "No person specified")
    if document_name == "":
        abort(400, "No document specified")

    file_path = f"./data/{person_name}/{document_name}"
    if os.path.exists(file_path):
        return send_file(file_path, mimetype=None)
    else:
        return (
            jsonify("No File Found"),
            200,
            {"Access-Control-Allow-Origin": "http://localhost:3000"},
        )

import math 

def process_relevant_docs(relevant_docs: list[tuple]):
    """Prepares the documents for the JSON response and handles NaN scores."""
    processed_docs = []
    for doc, score in relevant_docs:
        processed_score = float(score)

        # Check for NaN and replace it with 0.0 instead of None
        if math.isnan(processed_score):
            processed_score = 0.0 # Send 0.0, which is JSON serializable
            print(f"[WARN] NaN score found for document from source: {doc.metadata.get('source', 'N/A')}. Replacing with 0.0.")

        processed_docs.append({
            "source": get_document_name(doc.metadata.get("source", "")),
            "page": doc.metadata.get("page", ""),
            "page_content": doc.page_content,
            "score": processed_score
        })
    return processed_docs

def clean_up_audio_files():
    clear_dir("./tts-outputs")

if __name__ == "__main__":
    clean_up_audio_files()
    app.run("0.0.0.0", port=8080, debug=True)
