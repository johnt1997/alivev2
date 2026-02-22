import torch
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from typing import List, Tuple, Dict # Dict hinzugefügt für Typsicherheit
from datetime import datetime
import os
from langchain.llms.llamacpp import LlamaCpp
from langchain.schema.vectorstore import VectorStoreRetriever
from typing import List, Tuple
from langchain.schema.document import Document
from packages.factual_verifier import FactualVerifier
import torch, numpy as np
from sentence_transformers import CrossEncoder, __version__ as st_ver
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.faiss import FAISS
from packages.globals import CHAIN_TYPE, SEARCH_KWARGS_NUM
from prompts.qa_chain_prompt import STRICT_QA_PROMPT, STRICT_QA_PROMPT_GERMAN
from prompts.translation_prompt import TRANSLATE, TRANSLATE_OPENAI
from prompts.impersonation_prompt import IMPERSONATION_PROMPT
from prompts.impersonation_prompt_with_personality import (
    IMPERSONATION_PROMPT_WITH_PERSONALITY,
)
from prompts.query_transformation_prompt import (
    QUERY_TRANSFORMATION,
    QUERY_TRANSFORMATION_GERMAN,
)
from packages.person import Person
from prompts.chat_history import CONDENSE_QUESTION_PROMPT


class InitializeQuesionAnsweringChain:
    def __init__(
        self,
        llm,
        retriever,
        db,
        person,
        chain_type: str = CHAIN_TYPE,
        search_kwargs_num = 3,
        language: str = "en",
        #factuality_threshold: float = 0.4,
        use_reranker: bool = True,
        eyewitness_mode: bool= True
        
    ):
        self.llm = llm
        self.chain_type = chain_type
        self.retriever = retriever
        self.db = db
        self.eyewitness_mode = eyewitness_mode
        self.use_reranker = use_reranker
        self.cross_encoder = None
        self.search_kwargs_num = search_kwargs_num
        self.language = language
        #self.factuality_threshold = factuality_threshold
        self.verifier = FactualVerifier()
        self.person = person
        #self.memory = self._init_memory()
        if use_reranker:
            self._init_reranker()
    
    # NEU: Methode zum Initialisieren des Rerankers
    def _init_reranker(self):
        """Initializes the CrossEncoder model for re-ranking."""
        # Modell Wahl: ms-marco-MiniLM-L-6-v2 ist gut für Englisch. Für Deutsch ggf. anderes Modell nötig!
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        # PrüfeGPU verfügbar ist (CUDA oder Metal) und setzt Device entsprechend
        # Hinweis: Auf M2 Mac sollte Metal automatisch erkannt werden, wenn llama-cpp-python korrekt installiert ist.
        # Auf Windows mit Iris Xe wird es wahrscheinlich 'cpu' sein.
        # Eine robustere Prüfung wäre hier sinnvoll, aber für den Start reichts:
        try:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():  # Mac Silicon
                device = 'mps'
            else:
                device = 'cpu'
        except AttributeError:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_kwargs = {'device': device}
        print(f"[INFO] Initializing CrossEncoder '{model_name}' on device: {device}")
        print(f"[RERANK] model={model_name} device={device} "
        f"torch={torch.__version__} sbert={st_ver} "
          f"np={np.__version__}")
        try:
            self.cross_encoder = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs=model_kwargs)
            print("[INFO] CrossEncoder initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Error initializing HuggingFaceCrossEncoder: {e}. Reranking might not work.")
            self.cross_encoder = None # Auf None setzen, damit wir später prüfen können

    
    def _init_memory(self):####
        """Initializes the memory for the conversational chain. Allows to ask follow up questions during conversation."""
        memory = ConversationBufferMemory(memory_key="chat_history")
        return memory
    
    def _debug_pair_stats(self, pairs, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_name)
            lens = [len(tok.encode(q, t, truncation=True, max_length=384))
                    for (q, t) in pairs]
            import numpy as np
            if not lens:
                print("[RERANK] pairs=0"); return
            a = np.array(lens)
            print(f"[RERANK] pairs={len(pairs)} token_len min/avg/max = "
                f"{a.min()}/{a.mean():.1f}/{a.max()}")
            # 2–3 Beispiele
            for i, (q,t) in enumerate(pairs[:3]):
                print(f"[RERANK] EX{i+1} q='{q[:60]}' t[:80]='{t[:80].replace(chr(10),' ')}'")
        except Exception as e:
            print(f"[RERANK] token length debug failed: {e}")


    def _transform_question(self, query: str) -> str:
        """Transforms a user-facing question into a third-person query for retrieval."""
        prompt = QUERY_TRANSFORMATION
        if self.language == "de":
            prompt = QUERY_TRANSFORMATION_GERMAN
            
        print(f"[INFO] Transforming question for person: {self.person.name}")
        transformed_query = self.llm.invoke(prompt.format(person=self.person.name, question=query))
        print(f"[INFO] Original query: '{query}'")
        print(f"[INFO] Transformed query: '{transformed_query}'")
        return transformed_query

    def answer(
        self,
        query: str,
        previous_question: str = "",
        previous_answer: str = "",
    ) -> tuple[str, list[tuple], dict]:
        """
        Hauptmethode für Question Answering mit korrekter Evaluation.
        Gibt rerankte Dokumente zurück für konsistente RAGAS-Metriken.
        Unterstützt optionale Chat-Historie via previous_question/previous_answer.
        """
        top_n = self.search_kwargs_num
        k_init = max(10, top_n)  # Mehr initiale Kandidaten für sinnvolles Reranking

        # Chat-Historie: Follow-up-Frage zu Standalone-Frage umformulieren
        if previous_question and previous_answer:
            print(f"[INFO] Condensing follow-up question with chat history...")
            chat_history = f"Human: {previous_question}\nAI: {previous_answer}"
            query = self.llm.invoke(
                CONDENSE_QUESTION_PROMPT.format(
                    chat_history=chat_history,
                    question=query,
                )
            )
            print(f"[INFO] Condensed standalone question: '{query}'")

        print(f"[INFO] Starting QA with k_init={k_init}, top_n={top_n}, reranker={self.use_reranker}")
        retrieval_query = self._transform_question(query)
        
        # 1) Temporär k erhöhen für mehr initiale Kandidaten
        orig_k = getattr(self.retriever, "k", top_n)
        orig_dense = getattr(self.retriever, "dense_k", None)
        orig_bm25  = getattr(self.retriever, "bm25_k", None)

        docs_with_scores: list[tuple] = []   # <— NEU: safe default
        try:
        # Kandidatenpool hochdrehen
            if hasattr(self.retriever, "set_k_init"):
                self.retriever.set_k_init(k_init)   # setzt k, dense_k, bm25_k
            else:
                if hasattr(self.retriever, "k"):       self.retriever.k = k_init
                if hasattr(self.retriever, "dense_k"): self.retriever.dense_k = max(2*k_init, k_init)
                if hasattr(self.retriever, "bm25_k"):  self.retriever.bm25_k  = max(3*k_init, k_init)

            print(f"[INFO] Temporarily set retriever.k to {k_init}")

        # 2) Dokumente abrufen
            docs = self.retriever.get_relevant_documents(retrieval_query)
            if docs and isinstance(docs[0], tuple):
                docs_with_scores = docs[:k_init]  # Schneide auf k_init ab
            else:
                # Fallback: wenn retriever List[Document] liefert
                docs_with_scores = [(d, 0.0) for d in docs[:k_init]]

        except Exception as e:                # <— NEU: fallback
            print(f"[ERROR] Retrieval failed: {e}")
            docs_with_scores = []
        
        finally:
            if orig_k     is not None and hasattr(self.retriever, "k"):       self.retriever.k = orig_k
            if orig_dense is not None and hasattr(self.retriever, "dense_k"): self.retriever.dense_k = orig_dense
            if orig_bm25  is not None and hasattr(self.retriever, "bm25_k"):  self.retriever.bm25_k  = orig_bm25
            
        # 3) Reranking (oder Top-N Selection)
        if self.use_reranker and self.cross_encoder and docs_with_scores:
            reranked_docs_with_scores = self._rerank_documents(retrieval_query, docs_with_scores, top_n=top_n)
            used_reranker = True
        else:
            # Fallback: Sortiere nach ursprünglichen Scores und nimm Top-N
            sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
            reranked_docs_with_scores = sorted_docs[:top_n]
            used_reranker = False
            print(f"[INFO] No reranking. Using top {top_n} from {len(docs_with_scores)} initial docs.")
        
        # 4) Context für LLM erstellen (aus reranked docs)
        if reranked_docs_with_scores:
            context = "\n\n".join(doc.page_content for doc, _ in reranked_docs_with_scores)
        else:
            context = ""
            print("[WARN] No documents available for context.")
        
        # 5) LLM Antwort generieren
        prompt = STRICT_QA_PROMPT 
        if context:
            factual_answer = self.llm.invoke(prompt.format(question=query, context=context))
        else:
            factual_answer = "I don't have enough information to answer this question."

        final_answer = factual_answer # Default to the factual answer


        if self.person.personality is not None:
            print("[INFO] Impersonating answer with personality...")
            personality_prompt = IMPERSONATION_PROMPT_WITH_PERSONALITY
             #final_answer = self.llm.invoke(
            #personality_prompt.format(
                #person=self.person.name,
                #answer=factual_answer,
                #personality=self.person.format_personality_prompt(),
            #)
            prompt_text = personality_prompt.format(
                person=self.person.name,
                answer=factual_answer,
                personality=self.person.format_personality_prompt(),
        )
        else:
            # Use the simple impersonation prompt
            print("[INFO] Impersonating answer without personality...")
            impersonation_prompt = IMPERSONATION_PROMPT
            #final_answer = self.llm.invoke(
                #mpersonation_prompt.format(person=self.person.name, answer=factual_answer)
            #)
            prompt_text = impersonation_prompt.format(
                person=self.person.name,
                answer=factual_answer
            
            )
        if self.eyewitness_mode:
            eyewitness_suffix = (
                "\n\n[Eyewitness] Ergänze am Ende 1–2 sehr kurze Sätze aus heutiger "
                "Zeitzeug*innen-Perspektive. Keine neuen Fakten. Kennzeichne mit 'Eyewitness:'."
            if self.language == "de" else
                "\n\n[Eyewitness] Append 1–2 very short sentences from a present-day "
                "eyewitness perspective at the end that directly address the question from today's perspective. Do not add new facts. Prefix with 'Eyewitness:'."
            )
        
            prompt_text += eyewitness_suffix

        final_answer = self.llm.invoke(prompt_text)
        print(f"[INFO] Factual Answer: {factual_answer}")
        print(f"[INFO] Final Impersonated Answer: {final_answer}")
        
        # 6) Factuality Score berechnen
        factuality_score = self.verifier.score(context, factual_answer) if context else 0.0
        
        # 7) Metadata zusammenstellen
        metadata = {
            "factuality_score": factuality_score,
            "k_init": k_init,
            "top_n": top_n,
            "used_reranker": used_reranker,
            "num_initial_docs": len(docs_with_scores),
            "num_final_docs": len(reranked_docs_with_scores)
        }
        
        print(f"[INFO] QA completed. Factuality: {factuality_score:.3f}, Used reranker: {used_reranker}")
        
        # WICHTIG: Gibt die rerankten Dokumente zurück (nicht die initialen!)
        return final_answer, reranked_docs_with_scores, metadata
    
    # In init_chain.py - _rerank_documents() Methode ersetzen
    def _rerank_documents(
        self, query: str, docs_with_scores: List[Tuple[Document, float]], top_n: int
    ) -> List[Tuple[Document, float]]:  # ÄNDERUNG: Gibt jetzt Tupel zurück!
        """Re-ranks the initially retrieved documents using the CrossEncoder."""
        if not self.cross_encoder:
            print("[WARN] CrossEncoder not available. Skipping re-ranking.")
            # Fallback: Gebe die Top N der ursprünglichen Liste zurück
            if docs_with_scores:
                sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
                return sorted_docs[:top_n]  # ÄNDERUNG: Tupel beibehalten
            else:
                return []
        
        if not docs_with_scores:
            print("[INFO] No documents provided for re-ranking.")
            return []

        print(f"[INFO] Re-ranking {len(docs_with_scores)} documents for query: '{query}'...")
        # Erstelle Paare aus [query, doc_content] für den CrossEncoder
        doc_contents = [doc.page_content for doc, score in docs_with_scores]

        query_doc_pairs = [[query, doc_content] for doc_content in doc_contents]
        self._debug_pair_stats(query_doc_pairs)

        # Berechne die Relevanz-Scores
        try:
            scores = self.cross_encoder.score(query_doc_pairs)
            

            n_total = scores.size
            n_nan   = int(np.isnan(scores).sum())
            n_inf   = int(np.isinf(scores).sum())
            print(f"[RERANK] score stats: total={n_total} nan={n_nan} inf={n_inf} "
                f"min={np.nanmin(scores):.4f} max={np.nanmax(scores):.4f}")

            # sanitize
            scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

            if n_nan or n_inf:
                print("[RERANK] WARNING: non-finite scores detected, sanitized to [0,1].")


        except Exception as e:
            print(f"[ERROR] Error during CrossEncoder prediction: {e}. Returning top N initial docs.")
            # Fallback bei Fehler
            sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
            return sorted_docs[:top_n]  # ÄNDERUNG: Tupel beibehalten

        # Kombiniert die ursprünglichen Dokumente mit den neuen Scores und sortiere
        original_docs = [doc for doc, score in docs_with_scores]
        docs_with_new_scores = list(zip(original_docs, scores))
        reranked_docs_with_scores = sorted(docs_with_new_scores, key=lambda x: x[1], reverse=True)

        # ÄNDERUNG: Gibt die Top N Tupel zurück (nicht nur Dokumente)
        print(f"[INFO] Re-ranking complete. Returning top {top_n} documents.")
        # Debugging: Zeige die Scores der Top-Dokumente
        for i, (doc, score) in enumerate(reranked_docs_with_scores[:top_n]):
            print(f"[DEBUG] Reranked Document {i+1}: Score={score:.4f}, Content-Snippet='{doc.page_content[:100]}'")

        return reranked_docs_with_scores[:top_n]  # ÄNDERUNG: Tupel zurückgeben
    
