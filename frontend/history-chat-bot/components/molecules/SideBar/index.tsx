"use client";
import {
  change_person,
  checkIfPersonHasVoice,
  getAllPersons,
  has_openai,
} from "@/app/utils/api";
import {
  Box,
  Button,
  Checkbox,
  CheckboxGroup,
  FormControl,
  FormLabel,
  Icon,
  IconButton,
  Select,
  Slider,
  SliderFilledTrack,
  SliderMark,
  SliderThumb,
  SliderTrack,
  Stack,
  useToast,
} from "@chakra-ui/react";
import React, { FC, useEffect, useState } from "react";
import { Switch } from "@chakra-ui/react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faUser,
  faRotateLeft,
  faGear,
} from "@fortawesome/free-solid-svg-icons";
import { RelevantDocuments } from "@/app/utils/types";
import Link from "next/link";
import NextLink from "next/link";
import { Credits } from "@/components/atoms/Credits";

interface SideBarProps {
  globalDisabled: number;
  setGlobalDisabled: (value: number | ((prev: number) => number)) => void;
  selected: string;
  setSelected: (selected: string) => void;
  clearChat: () => void;
  setRelevantDocs: (relevantDocs: RelevantDocuments[]) => void;
  hasVoice: boolean;
  setHasVoice: (hasVoice: boolean) => void;
  selectedVoice: string;
  setSelectedVoice: (selectedVoice: string) => void;
  useOpenai: boolean;
  setUseOpenai: (useOpenai: boolean) => void;
  useEyewitnessMode: boolean;
  setEyewitnessMode: (enabled: boolean) => void;
  temperature: number;
  setTemperature: (temperature: number) => void;
  chunkSize: number;
  setChunkSize: (chunkSize: number) => void;
  chunkOverlap: number;
  setChunkOverlap: (chunkOverlap: number) => void;
  searchKwargsNum: number;
  setSearchKwargsNum: (searchKwargsNum: number) => void;
  selectedLanguage: string;
  setSelectedLanguage: (selectedLanguage: string) => void;
}

export const SideBar: FC<SideBarProps> = ({
  globalDisabled,
  setGlobalDisabled,
  selected,
  setSelected,
  clearChat,
  setRelevantDocs,
  hasVoice,
  setHasVoice,
  selectedVoice,
  setSelectedVoice,
  useOpenai,
  setUseOpenai,
  useEyewitnessMode,
  setEyewitnessMode,
  temperature,
  setTemperature,
  chunkSize,
  setChunkSize,
  chunkOverlap,
  setChunkOverlap,
  searchKwargsNum,
  setSearchKwargsNum,
  selectedLanguage,
  setSelectedLanguage,
}) => {
  const [persons, setPersons] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedPerson, setSelectedPerson] = useState<string>("");
  const [useOpenaiInit, setUseOpenaiInit] = useState<boolean>(false);
  const [temperatureInit, setTemperatureInit] = useState<number>(0.1);
  const [chunkSizeInit, setChunkSizeInit] = useState<number>(1000);
  const [chunkOverlapInit, setChunkOverlapInit] = useState<number>(0);
  const [searchKwargsInit, setSearchKwargsNumInit] = useState<number>(3);
  const [selectedVoiceInit, setSelectedVoiceInit] = useState<string>("Male");
  const [openAi, setHasOpenai] = useState<boolean>(false);
  const [eyewitnessModeInit, setEyewitnessModeInit] = useState<boolean>(useEyewitnessMode);
  const [selectedLanguageInit, setSelectedLanguageInit] =
    useState<string>("en");
  const toast = useToast();

  useEffect(() => {
    const checkOpenAI = async () => {
      const response = await has_openai();
      if (response && response.ok) {
        const data = await response.json();
        setHasOpenai(data);
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
    };

    checkOpenAI();
  }, []);

  const labelStyles = {
    mt: "2",
    ml: "-2.5",
    fontSize: "sm",
    color: "white",
  };

  useEffect(() => {
    const fetchVoice = async () => {
      if (selectedPerson) {
        const response = await checkIfPersonHasVoice(selectedPerson);
        if (response && response.ok) {
          const data = await response.json();
          setHasVoice(data);
        }
      }
    };

    fetchVoice();
  }, [selectedPerson]);

  useEffect(() => {
    const fetchPersons = async () => {
      setGlobalDisabled((value) => Math.max(value + 1, 0));
      const response = await getAllPersons();
      if (response && response.ok) {
        const data = await response.json();
        if (data.length > 0) {
          const sorted_persons = data.sort();
          setPersons(sorted_persons);
          setSelected(sorted_persons[0]);
          setSelectedPerson(sorted_persons[0]);
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

  const revertChanges = () => {
    setSelectedPerson(selected);
    setUseOpenai(useOpenaiInit);
    setTemperature(temperatureInit);
    setChunkSize(chunkSizeInit);
    setChunkOverlap(chunkOverlapInit);
    setSearchKwargsNum(searchKwargsInit);
    setSelectedVoiceInit(selectedVoice);
    setEyewitnessMode(eyewitnessModeInit);
    setSelectedLanguageInit(selectedLanguage);
  };

  const applyChanges = async () => {
    if (chunkOverlap > chunkSize) {
      toast({
        description: "Chunk Overlap cant be bigger than Chunk Size.",
        status: "error",
        duration: 5000,
        isClosable: true,
        position: "top",
      });

      // revertChanges();
      setChunkSize(chunkSizeInit);
      setChunkOverlap(chunkOverlapInit);
      return;
    }

    if (selectedPerson != selected) {
      clearChat();
      setRelevantDocs([]);
    }

    setSelectedVoice(selectedVoiceInit);
    setSelectedLanguage(selectedLanguageInit);
    setSelected(selectedPerson);
    setChunkOverlapInit(chunkOverlap);
    setChunkSizeInit(chunkSize);
    setTemperatureInit(temperature);
    setSearchKwargsNumInit(searchKwargsNum);
    setUseOpenaiInit(useOpenai);
    setEyewitnessModeInit(useEyewitnessMode);
};
 

  return (
    <div className="flex w-1/6 flex-col h-screen bg-dark px-5 pt-5 gap-4">
      <div
        className="flex flex-col flex-1 gap-4 overflow-y-scroll"
        style={{
          scrollbarWidth: "none",
        }}
      >
        {persons.map((person, index) => (
          <Button
            key={index}
            isDisabled={loading || globalDisabled !== 0}
            onClick={() => setSelectedPerson(person)}
            colorScheme={selectedPerson == person ? "gray" : "transparent"}
            size="sm"
            p={2}
            justifyContent={"flex-start"}
          >
            <p
              key={index}
              className={`${
                selectedPerson == person ? "text-black" : "text-white"
              }`}
            >
              {person}
            </p>
          </Button>
        ))}
      </div>

      <div className="bg-white rounded-md">
        <Select
          color={"black"}
          size={"sm"}
          rounded={"md"}
          onChange={(e) => setSelectedLanguageInit(e.target.value)}
          value={selectedLanguageInit}
          isDisabled={
            selected == undefined ||
            selected == "" ||
            globalDisabled !== 0 ||
            loading
          }
        >
          <option value="en">English</option>
          <option value="de">German</option>
        </Select>
      </div>

      <div className="bg-white rounded-md">
        <Select
          onChange={(e) => setSelectedVoiceInit(e.target.value)}
          size={"sm"}
          rounded={"md"}
          value={selectedVoiceInit}
          isDisabled={
            selected == undefined ||
            selected == "" ||
            globalDisabled !== 0 ||
            loading
          }
        >
          <option value="Male">Male (Only English TTS Supported)</option>
          <option value="Female">Female (Only English TTS Supported)</option>
          {hasVoice && <option value="Custom">Custom</option>}
        </Select>
      </div>

      <div className="flex flex-col gap-4 mt-auto">
        <div className="bg-white rounded-md">
          <FormControl className="flex flex-row items-center bg-white p-2 rounded-md gap-5">
            <Switch
              size={"sm"}
              colorScheme="green"
              isChecked={useOpenai}
              isDisabled={
                loading ||
                globalDisabled !== 0 ||
                selected == undefined ||
                selected == "" ||
                !openAi
              }
              onChange={() => {
                setUseOpenai(!useOpenai);
              }}
            />
            <p className={`${openAi ? "text-black" : "text-red-500"} text-sm`}>
              {openAi
                ? "Use Openai"
                : "You need an OpenAI API key to use OpenAI, see README.md for more information."}
            </p>
          </FormControl>
          <FormControl className="flex flex-row items-center bg-white p-2 rounded-md gap-5">
                <Switch size = "sm" 
                colorScheme = "green" 
                isDisabled={
                  loading ||
                  globalDisabled !== 0 ||
                  selected === "" ||
                  !openAi
                }
                isChecked={useEyewitnessMode}
                onChange={() => setEyewitnessMode(!useEyewitnessMode)}
              />
              <p className="text-black text-sm">Eyewitness Mode</p>
          </FormControl>

          <Box p={4} pt={0}>
            <p className="text-black self-center text-sm">
              Chunk Size {chunkSize}
            </p>

            <Slider
              size={"sm"}
              className=""
              isDisabled={
                loading ||
                globalDisabled !== 0 ||
                selected == undefined ||
                selected == ""
              }
              defaultValue={chunkSize}
              value={chunkSize}
              min={200}
              max={4000}
              step={100}
              onChange={(value) => {
                setChunkSize(value);
              }}
            >
              <SliderMark value={200} {...labelStyles}>
                0
              </SliderMark>
              <SliderMark value={2000} {...labelStyles}>
                2500
              </SliderMark>
              <SliderMark value={4000} {...labelStyles} ml={-6}>
                4000
              </SliderMark>
              <SliderTrack bg="gray.200">
                <SliderFilledTrack bg="green" />
              </SliderTrack>
              <SliderThumb boxSize={4} />
            </Slider>
          </Box>

          <Box p={4} pt={0}>
            <p className="text-black self-center text-sm">
              Chunk Overlap: {chunkOverlap}
            </p>

            <Slider
              size={"sm"}
              className=""
              isDisabled={
                loading ||
                globalDisabled !== 0 ||
                selected == undefined ||
                selected == ""
              }
              defaultValue={chunkOverlap}
              value={chunkOverlap}
              min={0}
              max={500}
              step={10}
              onChange={(value) => {
                setChunkOverlap(value);
              }}
            >
              <SliderMark value={0} {...labelStyles}>
                0
              </SliderMark>
              <SliderMark value={250} {...labelStyles}>
                250
              </SliderMark>
              <SliderMark value={500} {...labelStyles} ml={-4}>
                500
              </SliderMark>
              <SliderTrack bg="gray.200">
                <SliderFilledTrack bg="green" />
              </SliderTrack>
              <SliderThumb boxSize={4} />
            </Slider>
          </Box>

          <Box p={4} pt={0}>
            <p className="text-black self-center text-sm">
              Temperature {temperature}
            </p>

            <Slider
              className=""
              size={"sm"}
              isDisabled={
                loading ||
                globalDisabled !== 0 ||
                selected == undefined ||
                selected == ""
              }
              defaultValue={temperature}
              value={temperature}
              min={0}
              max={1}
              step={0.1}
              onChange={(value) => {
                setTemperature(value);
              }}
            >
              <SliderMark value={0} {...labelStyles}>
                0
              </SliderMark>
              <SliderMark value={0.5} {...labelStyles}>
                0.5
              </SliderMark>
              <SliderMark value={1} {...labelStyles}>
                1
              </SliderMark>
              <SliderTrack bg="gray.200">
                <SliderFilledTrack bg="green" />
              </SliderTrack>
              <SliderThumb boxSize={4} />
            </Slider>
          </Box>

          <Box p={4} pt={0}>
            <p className="text-black self-center text-sm">
              Num Documents {searchKwargsNum}
            </p>

            <Slider
              className=""
              size={"sm"}
              isDisabled={
                loading ||
                globalDisabled !== 0 ||
                selected == undefined ||
                selected == ""
              }
              defaultValue={searchKwargsNum}
              value={searchKwargsNum}
              min={1}
              max={8}
              step={1}
              onChange={(value) => {
                setSearchKwargsNum(value);
              }}
            >
              <SliderMark value={1} {...labelStyles}>
                0
              </SliderMark>
              <SliderMark value={4} {...labelStyles}>
                4
              </SliderMark>
              <SliderMark value={8} {...labelStyles}>
                8
              </SliderMark>
              <SliderTrack bg="gray.200">
                <SliderFilledTrack bg="green" />
              </SliderTrack>
              <SliderThumb boxSize={4} />
            </Slider>
          </Box>
        </div>

        <div className="flex flex-1 gap-2">
          <Button
            onClick={() => {
              applyChanges();
            }}
            size={"sm"}
            rounded={"md"}
            flex={1}
            isDisabled={
              loading ||
              globalDisabled !== 0 ||
              selected == undefined ||
              selected == "" ||
              (selectedPerson == selected &&
                useOpenai == useOpenaiInit &&
                temperature == temperatureInit &&
                chunkSize == chunkSizeInit &&
                chunkOverlap == chunkOverlapInit &&
                searchKwargsNum == searchKwargsInit &&
                selectedVoice == selectedVoiceInit &&
                selectedLanguage == selectedLanguageInit &&
                useEyewitnessMode == eyewitnessModeInit)
            }
            isLoading={loading}
            // className="mt-auto"
            colorScheme="gray"
          >
            Apply
          </Button>

          <IconButton
            //variant="outline"
            aria-label="reset"
            size={"sm"}
            icon={<FontAwesomeIcon icon={faRotateLeft} width="14" />}
            isDisabled={
              loading ||
              globalDisabled !== 0 ||
              selected == undefined ||
              selected == "" ||
              (selectedPerson == selected &&
                useOpenai == useOpenaiInit &&
                temperature == temperatureInit &&
                chunkSize == chunkSizeInit &&
                chunkOverlap == chunkOverlapInit &&
                useEyewitnessMode == eyewitnessModeInit &&
                searchKwargsNum == searchKwargsInit &&
                selectedVoice == selectedVoiceInit)
            }
            onClick={() => revertChanges()}
          />
          <NextLink href="/settings">
            <IconButton
              size={"sm"}
              aria-label="settings"
              icon={<FontAwesomeIcon icon={faGear} width="14" />}
              isDisabled={loading || globalDisabled !== 0}
            />
          </NextLink>
        </div>
        <Credits />
      </div>
    </div>
  );
};
