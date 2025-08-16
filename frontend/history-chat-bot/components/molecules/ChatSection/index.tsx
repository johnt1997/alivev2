"use client";

import { ask_question } from "@/app/utils/api";
import { exportChatAsMarkdown, exportChatAsPDF } from "@/app/utils/export";
import {
  AnswerWithDocuments,
  Message,
  RelevantDocuments,
  TransformationSteps,
} from "@/app/utils/types";
import { ChatMessage } from "@/components/atoms/ChatMessage";
import {
  Button,
  Input,
  InputGroup,
  InputRightElement,
  useToast,
} from "@chakra-ui/react";
import React, { FC, useEffect, useRef, useState } from "react";
import { AccordionSection } from "../AccordionSection";
import { HStack } from "@chakra-ui/react";
import { FaFileExport } from "react-icons/fa";

interface ChatSectionProps {
  globalDisabled: number;
  setGlobalDisabled: (value: number | ((prev: number) => number)) => void;
  selected: string;
  chat: Message[];
  setChat: (chat: Message) => void;
  relevantDocs: RelevantDocuments[];
  setRelevantDocs: (relevantDocs: RelevantDocuments[]) => void;
  audioPlaying: boolean;
  setAudioPlaying: (playing: boolean) => void;
  hasVoice: boolean;
  selectedVoice: string;
  useOpenai: boolean;
  eyewitnessMode: boolean;
  useReranker: boolean;
  useHybridSearch: boolean;
  splitterType: "recursive" | "sentence_transformer" | "semantic";
  temperature: number;
  chunkSize: number;
  chunkOverlap: number;
  searchKwargsNum: number;
  selectedLanguage: string;
}

export const ChatSection: FC<ChatSectionProps> = ({
  globalDisabled,
  setGlobalDisabled,
  selected,
  chat,
  setChat,
  relevantDocs,
  setRelevantDocs,
  audioPlaying,
  setAudioPlaying,
  hasVoice,
  selectedVoice,
  eyewitnessMode,
  useReranker,
  useHybridSearch,
  splitterType,
  useOpenai,
  temperature,
  chunkSize,
  chunkOverlap,
  searchKwargsNum,
  selectedLanguage,
}) => {
  const exportChat = (format: "markdown" | "pdf") => {
    if (format === "markdown") exportChatAsMarkdown(chat);
    else exportChatAsPDF(chat);
  };
  const toast = useToast();
  const ref = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [previousQuestion, setPreviousQuestion] = useState<string>("");
  const [previousAnswer, setPreviousAnswer] = useState<string>("");
  const [transformationSteps, setTransformationSteps] =
    useState<TransformationSteps>();

  useEffect(() => {
    setPreviousAnswer("");
    setPreviousQuestion("");
  }, [selected]);

  const handleSend = async (query: string) => {
    setChat({ text: query, isHuman: true });
    setLoading(true);
    setGlobalDisabled((value) => Math.max(value + 1, 0));
    setQuery("");
    const response = await ask_question(
      selected,
      query,
      selectedVoice,
      chunkSize.toString(),
      chunkOverlap.toString(),
      temperature.toString(),
      searchKwargsNum.toString(),
      useOpenai ? "True" : "False",
      selectedLanguage,
      previousQuestion,
      previousAnswer,
      eyewitnessMode ? "True" : "False",
      useReranker ? "True" : "False",
      useHybridSearch ? "hybrid" : "dense",
      splitterType
    );
    if (response && response.ok) {
      const data: AnswerWithDocuments = await response.json();

      setTransformationSteps(data?.transformation_steps);
      setRelevantDocs(data?.relevant_docs);
      setChat({ text: data?.answer, isHuman: false });
      setPreviousQuestion(query);
      setPreviousAnswer(data?.answer);
    } else {
      setRelevantDocs([]);
      setChat({ text: "Something went wrong :/", isHuman: false });
      toast({
        title: "Ups!",
        description: "Something went wrong, please try again later.",
        status: "error",
        duration: 5000,
        isClosable: true,
        position: "top",
      });
    }

    setLoading(false);
    setGlobalDisabled((value) => Math.max(value - 1, 0));
  };

  useEffect(() => {
    if (chat.length > 0) ref.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat]);

  return (
    <div className="flex flex-col flex-1 min-w-0 h-screen min-h-0 overflow-hidden bg-theme p-5">
      {/** Chat Messages */}
      <div
        className=" flex flex-1 overflow-y-auto flex-col pb-40"
        style={{
          scrollbarWidth: "none",
        }}
      >
        {chat.map((message, index) => (
          <ChatMessage
            key={index}
            message={message}
            selected={selected}
            hasVoice={hasVoice}
            globalDisabled={globalDisabled}
            setGlobalDisabled={setGlobalDisabled}
            audioPlaying={audioPlaying}
            setAudioPlaying={setAudioPlaying}
            selectedVoice={selectedVoice}
            selectedLanguage={selectedLanguage}
          />
        ))}
        <div ref={ref} />
      </div>

      <AccordionSection
        relevantDocuments={relevantDocs}
        transformationSteps={transformationSteps}
        answer={chat[chat.length - 1]?.text}
      />
      <div className="flex flex-col">
        <HStack spacing={2} mb={2}>
          <Button
            leftIcon={<FaFileExport />}
            size="sm"
            onClick={() => exportChat("markdown")}
          >
            Export MD
          </Button>
          <Button
            leftIcon={<FaFileExport />}
            size="sm"
            onClick={() => exportChat("pdf")}
          >
            Export PDF
          </Button>
        </HStack>
      </div>

      <InputGroup className="self-end justify-end" size="md">
        <Input
          className="text-white"
          border={"0.5px solid grey"}
          rounded={"xl"}
          focusBorderColor="grey"
          type={"text"}
          placeholder="Ask Question ..."
          onChange={(e) => {
            setQuery(e.target.value);
          }}
          onKeyUp={(e) => {
            if (e.key === "Enter") handleSend(query);
          }}
          value={query}
          isDisabled={loading || globalDisabled !== 0}
        />
        <InputRightElement width="4.5rem">
          <Button
            h="1.75rem"
            size="sm"
            onClick={() => handleSend(query)}
            isLoading={loading}
            isDisabled={loading || globalDisabled !== 0 || query.length === 0}
          >
            SEND
          </Button>
        </InputRightElement>
      </InputGroup>
    </div>
  );
};
