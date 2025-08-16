export interface AnswerWithDocuments {
  answer: string;
  transformation_steps: TransformationSteps;
  relevant_docs: RelevantDocuments[];
}

export interface TransformationSteps {
  original_query: string;
  follow_up_question: string;
  third_person_query: string;
  original_answer: string;
  transformed_answer: string;
}

export interface RelevantDocuments {
  source: string;
  page: number; // string?
  page_content: string;
  score: number;
}

export interface Message {
  text: string;
  isHuman: boolean;
}

export type SplitterType = "recursive" | "sentence_transformer" | "semantic";
