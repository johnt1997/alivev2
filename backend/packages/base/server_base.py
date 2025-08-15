from flask import Flask, g, request, json, abort, jsonify, send_file
import os
from packages.document_processing import DocumentProcessing
from packages.vector_store_handler import VectorStoreHandler
from packages.llm_config import LLMConfig
from packages.init_chain import InitializeQuesionAnsweringChain
from helper_functions import clear_dir, delete_dir, string_to_bool, get_document_name
from packages.globals import (
    EMBEDDINGS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEARCH_KWARGS_NUM,
    TEMPERATURE,
)
from dotenv import load_dotenv
from datetime import datetime
from packages.person import Person
from typing import List, Tuple
from langchain.schema.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from flask_cors import CORS
import openai

os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])


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


@app.route("/upload-person-data", methods=["POST"])
def upload_person_data():
    """
    This function creates a new person with the given name and the given files.
    If the person already exists, it will delete it and create a new one with the data.
    Accepts .txt and .pdf files.

    NOTE: upload-person-data and create-person MUST be called together when creating a new person ->
          keep in mind when implementing front-end
    """

    if "file" not in request.files:
        abort(400, "No file part")

    name = request.args.get("name", default="", type=str)
    if name == "":
        abort(400, "No name specified")

    files = request.files.getlist("file")

    for file in files:
        if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
            abort(400, "Only PDF and TXT files are allowed")

    # Check if person already exists and delete it
    # this is easier than constantly updating the vector store(s)
    clear_dir(f"./data/{name}")
    clear_dir("./database/" + OpenAIEmbeddings.__name__ + "_" + name)
    clear_dir("./database/" + HuggingFaceEmbeddings.__name__ + "_" + name)

    if not os.path.exists(f"./data/{name}"):
        os.makedirs(f"./data/{name}")

    for file in files:
        if file.filename.endswith(".pdf") or file.filename.endswith(".txt"):
            file.save(f"./data/{name}/" + file.filename)

    # return f"{name} Data Successfully Created"
    return (
        jsonify("File succesfully uploaded"),
        200,
        {"Access-Control-Allow-Origin": "http://localhost:3000"},
    )


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
    if person_name == "":
        abort(400, "No person specified")
    delete_dir("./data/" + person_name)
    delete_dir("./database/" + OpenAIEmbeddings.__name__ + "_" + person_name)
    delete_dir("./database/" + HuggingFaceEmbeddings.__name__ + "_" + person_name)
    delete_dir("./virtual_characters/" + person_name)
    delete_dir("./voices/" + person_name)
    delete_dir("./backgrounds/" + person_name)

    if os.path.exists("./persons/" + person_name + ".json"):
        os.remove("./persons/" + person_name + ".json")
    return (
        jsonify(f"Delete Succesfully: {person_name}"),
        200,
        {"Access-Control-Allow-Origin": "http://localhost:3000"},
    )


@app.route("/ask-question", methods=["GET"])
def ask_question():
    person_name = request.args.get("person", default="", type=str)
    question = request.args.get("question", default="", type=str)
    voice = request.args.get("voice", default="Male", type=str)
    chunk_size = request.args.get("chunk_size", default=CHUNK_SIZE, type=int)
    chunk_overlap = request.args.get("chunk_overlap", default=CHUNK_OVERLAP, type=int)
    temperature = request.args.get("temperature", default=TEMPERATURE, type=float)
    search_kwargs_num = request.args.get(
        "search_kwargs_num", default=SEARCH_KWARGS_NUM, type=int
    )
    use_openai = request.args.get("use_openai", default="False", type=str)
    use_openai = string_to_bool(use_openai)
    language = request.args.get("language", default="en", type=str)

    previous_question = request.args.get("previous_question", default="", type=str)
    previous_answer = request.args.get("previous_answer", default="", type=str)

    if person_name == "":
        abort(400, "No person specified")

    person = Person(name=person_name, voice=voice)

    embeddings = EMBEDDINGS
    if use_openai:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    vectorstore_handler = VectorStoreHandler(
        embeddings=embeddings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        search_kwargs_num=search_kwargs_num,
    )

    if not os.path.exists(
        f"./database/{vectorstore_handler._get_embedding_name()}_{person.name}/{chunk_size}_{chunk_overlap}"
    ):
        split_documents = DocumentProcessing().get_chunked_documents(
            directory_path=f"./data/{person.name}",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        retriever, db = vectorstore_handler.create_db_and_retriever(
            documents=split_documents,
            vector_store_name=person.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        retriever, db = vectorstore_handler.get_db_and_retriever(
            vector_store_name=person.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    llm_config = LLMConfig(temperature=temperature)
    llm = llm_config.get_local_llm(use_openai=use_openai)
    qa_chain = InitializeQuesionAnsweringChain(
        llm=llm,
        retriever=retriever,
        db=db,
        person=person,
        search_kwargs_num=search_kwargs_num,
        language=language,
    )

    answer, relevant_docs = qa_chain.get_answer_with_docs_and_scores(
        query=question,
        previous_question=previous_question,
        previous_answer=previous_answer,
    )

    response_data = {
        "answer": answer[0],
        "transformation_steps": answer[1],
        "relevant_docs": process_relevant_docs(relevant_docs),
    }

    """
    print("###############################################")
    print("Asking:")
    print(f"Person: {person.name}")
    print(f"Chunk Size: {chunk_size}")
    print(f"Chunk Overlap: {chunk_overlap}")
    print(f"Temperature: {temperature}")
    print(f"Search Kwargs Num: {search_kwargs_num}")
    print(f"Use OpenAI: {use_openai}")
    print(f"Voice: {voice}")
    print(f"Language: {language}")
    print(f"Question: {question}")
    print("###############################################")
    """
    return (
        jsonify(response_data),
        200,
        {"Access-Control-Allow-Origin": "http://localhost:3000"},
    )


@app.route("/get-all-persons", methods=["GET"])
def get_all_persons():
    directory_path = "./persons"
    json_files = [f for f in os.listdir(directory_path) if f.endswith(".json")]
    persons = []
    # Open and read each JSON file
    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        with open(file_path, "r") as file:
            data = json.load(file)
            persons.append(data["name"])

    return (
        jsonify(persons),
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


def process_relevant_docs(relevant_docs: List[Tuple[Document, float]]):
    processed_docs = []
    for doc, score in relevant_docs:
        processed_doc = {
            "source": get_document_name(doc.metadata.get("source", "")),
            "page": doc.metadata.get("page", ""),
            "page_content": doc.page_content,
            "score": score,
        }
        processed_docs.append(processed_doc)
    return processed_docs


def clean_up_audio_files():
    clear_dir("./tts-outputs")


if __name__ == "__main__":
    clean_up_audio_files()
    app.run("0.0.0.0", port=8080, debug=True)
