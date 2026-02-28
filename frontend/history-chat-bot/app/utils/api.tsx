// src/services/api.js
const BASE_URL = "http://127.0.0.1:8080";
import type { SplitterType } from "./types";

export async function getAudioResponse(
  person: string,
  chat_index: string,
  text: string,
  voice: string,
  language: string
) {
  try {
    const response = await fetch(
      `${BASE_URL}/get-audio-response?person=${person}&chat_index=${chat_index}&text=${text}&voice=${voice}&language=${language}`
    );
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function checkIfPersonHasVoice(person: string) {
  try {
    const response = await fetch(
      `${BASE_URL}/person-has-voice?person=${person}`
    );
    // const data = await response.json();
    // return data;
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function getAllPersons() {
  try {
    const response = await fetch(`${BASE_URL}/get-all-persons`, {
      method: "GET",
      mode: "cors",
      headers: {
        "Content-Type": "application/json",
      },
    });
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function getVirtualCharacter(person: string) {
  try {
    const response = await fetch(
      `${BASE_URL}/get-virtual-character?person=${person}`
    );
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function get_person_documents_names(person: string) {
  try {
    const response = await fetch(
      `${BASE_URL}/get-person-documents-names?person=${person}`
    );
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function get_document(person: string, document: string) {
  try {
    const response = await fetch(
      `${BASE_URL}/get-document?person=${person}&document=${document}`
    );
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function ask_question(
  person: string,
  question: string,
  voice: string = "Male",
  chunk_size: string = "1000",
  chunk_overlap: string = "0",
  temperature: string = "0.4",
  search_kwargs_num: string = "3",
  use_openai: string = "False",
  language: string = "en",
  previous_question: string = "",
  previous_answer: string = "",
  eyewitness_mode: string = "False",
  use_reranker = "False",
  retrieval_mode: "dense" | "hybrid" = "dense",
  splitter_type: SplitterType = "semantic",
  llm_model: string = "Phi-3-mini-4k-instruct-q4.gguf"
) {
  try {
    const response = await fetch(
      `${BASE_URL}/ask-question?question=${question}&person=${person}&voice=${voice}&chunk_size=${chunk_size}&chunk_overlap=${chunk_overlap}&temperature=${temperature}&search_kwargs_num=${search_kwargs_num}&use_openai=${use_openai}&language=${language}&previous_question=${previous_question}&previous_answer=${previous_answer}&eyewitness_mode=${eyewitness_mode}&retrieval_mode=${retrieval_mode}&splitter_type=${splitter_type}&use_reranker=${use_reranker}&llm_model=${llm_model}`
    );
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function change_person(
  person: string,
  chunk_size: string = "1000",
  chunk_overlap: string = "0",
  search_kwargs_num: string = "3",
  temperature: string = "0.4",
  use_openai: string = "False",
  voice: string = "Male"
) {
  try {
    const response = await fetch(
      `${BASE_URL}/change-person?person=${person}&chunk_size=${chunk_size}&chunk_overlap=${chunk_overlap}&search_kwargs_num=${search_kwargs_num}&temperature=${temperature}&use_openai=${use_openai}&voice=${voice}`,
      {
        method: "POST",
      }
    );
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function create_person(name: string, personality: any = null) {
  try {
    const response = await fetch(`${BASE_URL}/create-person`, {
      method: "POST",
      body: JSON.stringify({ name, personality }),
    });
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function upload_person_data(name: string, files: File[]) {
  try {
    const formData = new FormData();
    files.forEach((file, index) => {
      formData.append(`file`, file);
    });
    const response = await fetch(
      `${BASE_URL}/upload-person-data?name=${name}`,
      {
        method: "POST",
        body: formData,
      }
    );
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function upload_person_voice(name: string, file: File) {
  try {
    const formData = new FormData();
    formData.append(`file`, file);
    const response = await fetch(
      `${BASE_URL}/upload-person-voice?name=${name}`,
      {
        method: "POST",
        body: formData,
      }
    );

    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function upload_person_character(name: string, file: File) {
  try {
    const formData = new FormData();
    formData.append(`file`, file);
    const response = await fetch(
      `${BASE_URL}/upload-virtual-character?name=${name}`,
      {
        method: "POST",
        body: formData,
      }
    );

    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function delete_person(name: string) {
  try {
    const response = await fetch(`${BASE_URL}/delete-person?person=${name}`, {
      method: "DELETE",
    });
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function get_person_data(name: string) {
  try {
    const response = await fetch(`${BASE_URL}/get-person-data?person=${name}`);
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function person_voice(name: string) {
  try {
    const response = await fetch(`${BASE_URL}/get-person-voice?person=${name}`);
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function has_openai() {
  try {
    const response = await fetch(`${BASE_URL}/has-openai-api-key`);
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function get_background(name: string) {
  try {
    const response = await fetch(`${BASE_URL}/get-background?person=${name}`);
    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}

export async function upload_background(name: string, file: File) {
  try {
    const formData = new FormData();
    formData.append(`file`, file);
    const response = await fetch(`${BASE_URL}/upload-background?name=${name}`, {
      method: "POST",
      body: formData,
    });

    return response;
  } catch (error) {
    console.log(error);
    return undefined;
  }
}
