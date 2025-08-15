"use client";
import { ChatSection } from "@/components/molecules/ChatSection";
import { CharacterSection } from "@/components/molecules/CharacterSection";
import { SideBar } from "@/components/molecules/SideBar";
import { useEffect, useState } from "react";
import {
  checkIfPersonHasVoice,
  getAllPersons,
  getVirtualCharacter,
  get_background,
} from "./utils/api";
import { Message, RelevantDocuments } from "./utils/types";
import { useToast } from "@chakra-ui/react";

export default function Home() {
  const toast = useToast();
  const loading_id = "loading-toast";
  const [chat, setChat] = useState<Message[]>([]);
  const [modelUrl, setModelUrl] = useState<string>();
  const [backgroundUrl, setBackgroundUrl] = useState<string>(
    "textures/library.jpg"
  );
  const [selected, setSelected] = useState<string>("");
  const [hasVoice, setHasVoice] = useState<boolean>(false);
  const [chunkSize, setChunkSize] = useState<number>(1000);
  const [useOpenai, setUseOpenai] = useState<boolean>(false);
  const [chunkOverlap, setChunkOverlap] = useState<number>(0);
  const [eyewitnessMode, setEyewitnessMode] = useState<boolean>(false);
  const [temperature, setTemperature] = useState<number>(0.1);
  const [globalDisabled, setGlobalDisabled] = useState<number>(0);
  const [audioPlaying, setAudioPlaying] = useState<boolean>(false);
  const [searchKwargsNum, setSearchKwargsNum] = useState<number>(3);
  const [selectedVoice, setSelectedVoice] = useState<string>("Male");
  const [relevantDocs, setRelevantDocs] = useState<RelevantDocuments[]>([]);
  const [selectedLanguage, setSelectedLanguage] = useState<string>("en");

  useEffect(() => {
    if (selectedVoice !== "Custom" && selectedLanguage == "de") {
      toast({
        title: "Warning",
        description:
          "Male and Female voices are only available in English. TTS Audio might sound weird.",
        status: "warning",
        duration: 5000,
        isClosable: true,
        position: "top",
      });
    }
  }, [selectedVoice, selectedLanguage]);

  useEffect(() => {
    if (globalDisabled > 0 && !toast.isActive(loading_id)) {
      toast({
        id: loading_id,
        title: "Loading.",
        description: "Please wait.",
        status: "loading",
        duration: null,
        isClosable: false,
        position: "top",
      });
    } else {
      toast.close(loading_id);
    }
  }, [globalDisabled]);

  const updateRelevantDocs = (relevantDocs: RelevantDocuments[]) => {
    setRelevantDocs(relevantDocs);
  };

  const updateChat = (message: Message) => {
    setChat((chat) => [...chat, message]);
  };

  const clearChat = () => {
    setChat([]);
  };

  const updateSelectedVoice = (voice: string) => {
    setSelectedVoice(voice);
  };

  useEffect(() => {
    const fetchPersons = async () => {
      setGlobalDisabled((value) => Math.max(value + 1, 0));
      const response = await getAllPersons();
      if (response && response.ok) {
        const data = await response.json();
        if (data.length > 0) {
          const sorted_persons = data.sort();
          setSelected(sorted_persons[0]);
        }
      } else {
        toast({
          title: "Ups!",
          description: "Something went wrong, please try again later.",
          status: "error",
          duration: 5000,
          isClosable: true,
          position: "top",
        });
      }

      setGlobalDisabled((value) => Math.max(value - 1, 0));
    };

    fetchPersons();
  }, []);

  useEffect(() => {
    const fetchPersonModel = async () => {
      if (selected) {
        setModelUrl(undefined);
        setGlobalDisabled((value) => Math.max(value + 1, 0));
        const response = await getVirtualCharacter(selected);
        if (response && response.ok) {
          if (
            response.headers.get("content-type") ==
              "application/octet-stream" ||
            response.headers.get("content-type") == "model/gltf-binary"
          ) {
            setModelUrl(response.url);
          }
        } else {
          toast({
            title: "Ups!",
            description: "Something went wrong, please try again later.",
            status: "error",
            duration: 5000,
            isClosable: true,
            position: "top",
          });
        }
      }
      setGlobalDisabled((value) => Math.max(value - 1, 0));
    };

    fetchPersonModel();
  }, [selected]);

  useEffect(() => {
    const fetchBackground = async () => {
      if (selected) {
        setBackgroundUrl("textures/library.jpg");
        setGlobalDisabled((value) => Math.max(value + 1, 0));
        const response = await get_background(selected);
        if (response && response.ok) {
          if (response.headers.get("content-type") !== "application/json") {
            setBackgroundUrl(response.url);
          }
        } else {
          toast({
            title: "Ups!",
            description: "Something went wrong, please try again later.",
            status: "error",
            duration: 5000,
            isClosable: true,
            position: "top",
          });
        }
      }
      setGlobalDisabled((value) => Math.max(value - 1, 0));
    };

    fetchBackground();
  }, [selected]);

  useEffect(() => {
    const fetchVoice = async () => {
      if (selected) {
        setGlobalDisabled((value) => Math.max(value + 1, 0));
        const response = await checkIfPersonHasVoice(selected);
        if (response && response.ok) {
          const data = await response.json();
          setHasVoice(data);
        }
        setGlobalDisabled((value) => Math.max(value - 1, 0));
      }
    };

    fetchVoice();
  }, [selected]);

  return (
    <main className="flex flex-1 h-screen flex-row items-center justify-between p-0">
      <SideBar
        selected={selected}
        setSelected={setSelected}
        globalDisabled={globalDisabled}
        setGlobalDisabled={setGlobalDisabled}
        clearChat={clearChat}
        setRelevantDocs={updateRelevantDocs}
        hasVoice={hasVoice}
        setHasVoice={setHasVoice}
        selectedVoice={selectedVoice}
        setSelectedVoice={updateSelectedVoice}
        useOpenai={useOpenai}
        setUseOpenai={setUseOpenai}
        useEyewitnessMode={eyewitnessMode}
        setEyewitnessMode={setEyewitnessMode}
        temperature={temperature}
        setTemperature={setTemperature}
        chunkSize={chunkSize}
        setChunkSize={setChunkSize}
        chunkOverlap={chunkOverlap}
        setChunkOverlap={setChunkOverlap}
        searchKwargsNum={searchKwargsNum}
        setSearchKwargsNum={setSearchKwargsNum}
        selectedLanguage={selectedLanguage}
        setSelectedLanguage={setSelectedLanguage}
      />
      <ChatSection
        selected={selected}
        globalDisabled={globalDisabled}
        setGlobalDisabled={setGlobalDisabled}
        chat={chat}
        setChat={updateChat}
        relevantDocs={relevantDocs}
        setRelevantDocs={updateRelevantDocs}
        audioPlaying={audioPlaying}
        setAudioPlaying={setAudioPlaying}
        hasVoice={hasVoice}
        selectedVoice={selectedVoice}
        useOpenai={useOpenai}
        eyewitnessMode={eyewitnessMode}
        temperature={temperature}
        chunkSize={chunkSize}
        chunkOverlap={chunkOverlap}
        searchKwargsNum={searchKwargsNum}
        selectedLanguage={selectedLanguage}
      />
      {modelUrl && (
        <CharacterSection
          audioPlaying={audioPlaying}
          setAudioPlaying={setAudioPlaying}
          modelUrl={modelUrl}
          backgroundUrl={backgroundUrl}
        />
      )}
    </main>
  );
}
