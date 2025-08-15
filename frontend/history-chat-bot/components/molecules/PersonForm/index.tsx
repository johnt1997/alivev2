"use client";
import {
  create_person,
  delete_person,
  getAllPersons,
  upload_background,
  upload_person_character,
  upload_person_data,
  upload_person_voice,
} from "@/app/utils/api";
import { PersonalityListElement } from "@/components/atoms/PersonalityListElement";
import {
  Button,
  FormControl,
  FormLabel,
  IconButton,
  Input,
  Textarea,
  useToast,
} from "@chakra-ui/react";
import { faUpload } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import React, { FC, use, useEffect, useState } from "react";

interface PersonFormProps {
  loading: boolean;
  setLoading: (loading: boolean) => void;
  persons: string[];
  setPersons: (persons: string[]) => void;
}

export const PersonForm: FC<PersonFormProps> = ({
  loading,
  setLoading,
  persons,
  setPersons,
}) => {
  const [name, setName] = useState<string>("");
  const [personalityList, setPersonalityList] = useState<string[][]>([]);
  const [singlePersonality, setSinglePersonality] = useState<string[]>([
    "",
    "",
  ]);

  const [voiceFile, setVoiceFile] = useState<File>();
  const [virtualAvatarFile, setVirtualAvatarFile] = useState<File>();
  const [bgFile, setBGFile] = useState<File>();
  const [documents, setDocuments] = useState<File[]>([]);
  const documentRef = React.useRef<HTMLInputElement>(null);
  const virtualAvatarRef = React.useRef<HTMLInputElement>(null);
  const voiceRef = React.useRef<HTMLInputElement>(null);
  const bgRef = React.useRef<HTMLInputElement>(null);
  const toast = useToast();
  const warningId = "warning-toast";

  const fetchPersons = async () => {
    const response = await getAllPersons();
    if (response && response.ok) {
      const data = await response.json();
      const sorted_persons = data.sort();
      setPersons(sorted_persons);
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

  useEffect(() => {
    persons.forEach((person) => {
      if (person === name && !toast.isActive(warningId)) {
        toast({
          id: warningId,
          title: "Warning",
          description:
            name +
            " already exists. Creating a new person with the same Name will overwrite the old one",
          status: "warning",
          duration: 4000,
          isClosable: true,
          position: "top",
        });
      }
    });
  }, [name]);

  const handleInputChange = (index: number, value: string) => {
    const newArray = [...singlePersonality];
    newArray[index] = value;
    setSinglePersonality(newArray);
  };

  const addArray = () => {
    setPersonalityList([...personalityList, singlePersonality]);
    setSinglePersonality(["", ""]);
  };

  const throwToast = () => {
    toast({
      title: "Ups!",
      description: "Something went wrong, please try again later.",
      status: "error",
      duration: 5000,
      isClosable: true,
      position: "top",
    });
    setLoading(false);
  };

  const createPerson = async () => {
    setLoading(true);

    if (!voiceFile || !virtualAvatarFile || !bgFile) {
      // delete person and then create again
      const response = await delete_person(name);
      if (!response?.ok) {
        throwToast();
        return;
      }
    }

    const cp_response = await create_person(
      name,
      personalityList.length ? personalityList : null
    );

    if (!cp_response?.ok) {
      throwToast();
      const response = await delete_person(name);
      return;
    }

    const upd_response = await upload_person_data(name, documents);
    if (!upd_response?.ok) {
      throwToast();
      const response = await delete_person(name);
      return;
    }

    if (voiceFile) {
      const upv_response = await upload_person_voice(name, voiceFile);
      if (!upv_response?.ok) {
        throwToast();
        const response = await delete_person(name);
        return;
      }
    }

    if (bgFile) {
      const ubg_response = await upload_background(name, bgFile);
      if (!ubg_response?.ok) {
        throwToast();
        const response = await delete_person(name);
        return;
      }
    }

    if (virtualAvatarFile) {
      const upc_response = await upload_person_character(
        name,
        virtualAvatarFile
      );
      if (!upc_response?.ok) {
        throwToast();
        const response = await delete_person(name);
        return;
      }
    }

    await fetchPersons();

    documentRef.current!.value = "";
    virtualAvatarRef.current!.value = "";
    voiceRef.current!.value = "";
    bgRef.current!.value = "";

    setName("");
    setDocuments([]);
    setVoiceFile(undefined);
    setVirtualAvatarFile(undefined);
    setLoading(false);
    setPersonalityList([]);
    setSinglePersonality(["", ""]);
    toast({
      title: "Success!",
      description: `${name} Created Successfully!`,
      status: "success",
      duration: 5000,
      isClosable: true,
      position: "top",
    });
  };

  const removePersonalityElement = (index: number) => {
    setPersonalityList((prev: string[][]) => {
      return [...prev.slice(0, index), ...prev.slice(index + 1)];
    });
  };

  return (
    <>
      <div
        className="flex flex-[1] w-1/2 self-center flex-col gap-4 overflow-y-scroll "
        style={{
          scrollbarWidth: "none",
        }}
      >
        <h1 className="text-3xl text-white">Add a Person</h1>
        <Input
          className="text-white"
          border={"0.5px solid grey"}
          rounded={"md"}
          focusBorderColor="grey"
          type={"text"}
          placeholder="Name"
          onChange={(e) => {
            setName(e.target.value);
          }}
          size="md"
          p={4}
          value={name}
          isDisabled={loading}
        />

        <FormControl>
          <FormLabel className="text-white">
            Upload Documents (.pdf, .txt)
          </FormLabel>
          <Input
            className="text-white items-center"
            border={"none"}
            padding={0}
            type={"file"}
            style={{}}
            title="Upload a file"
            accept=".pdf,.txt"
            ref={documentRef}
            multiple={true}
            onChange={(e) => {
              let files = e.target.files;
              if (files?.length) {
                for (let i = 0; i < files.length; i++) {
                  setDocuments((documents: File[]) => [
                    ...documents,
                    files![i],
                  ]);
                }
              }
            }}
            isDisabled={loading}
          />
        </FormControl>

        <FormControl>
          <FormLabel className="text-white">
            <div className="flex gap-2">
              Upload Virtual Avatar (Optionally | .glb file)
              <a
                className={"text-blue-200"}
                href={`https://readyplayer.me/`}
                target="_blank"
              >
                (Ready Player Me)
              </a>
            </div>
          </FormLabel>
          <Input
            className="text-white items-center"
            border={"none"}
            padding={0}
            type={"file"}
            title="Upload a file"
            accept=".glb"
            ref={virtualAvatarRef}
            multiple={false}
            onChange={(e) => {
              let files = e.target.files;
              if (files?.length) {
                setVirtualAvatarFile(files[0]);
              }
            }}
            isDisabled={loading}
          />
        </FormControl>

        <FormControl>
          <FormLabel className="text-white">
            <div className="flex gap-2">
              Upload Avatar Background (Optionally | .jpg)
            </div>
          </FormLabel>
          <Input
            className="text-white items-center"
            border={"none"}
            padding={0}
            type={"file"}
            title="Upload a file"
            accept=".jpg"
            ref={bgRef}
            multiple={false}
            onChange={(e) => {
              let files = e.target.files;
              if (files?.length) {
                setBGFile(files[0]);
              }
            }}
            isDisabled={loading}
          />
        </FormControl>

        <FormControl>
          <FormLabel className="text-white">
            Upload Voice (Optionally | .wav file)
          </FormLabel>
          <Input
            className="text-white items-center"
            type={"file"}
            border={"none"}
            padding={0}
            focusBorderColor="grey"
            title="Upload a file"
            ref={voiceRef}
            multiple={false}
            accept="audio/wav"
            onChange={(e) => {
              let files = e.target.files;
              if (files?.length) {
                setVoiceFile(files[0]);
              }
            }}
            isDisabled={loading}
          />
        </FormControl>

        <FormControl className="">
          <FormLabel className="text-white">
            Define Personality (optional)
          </FormLabel>
          <Textarea
            className="text-white items-center"
            border={"0.5px solid grey"}
            rounded={"md"}
            focusBorderColor="grey"
            value={singlePersonality[0] ?? ""}
            placeholder="Original Answer"
            onChange={(e) => handleInputChange(0, e.target.value)}
            isDisabled={loading}
          />

          <Textarea
            className="text-white items-center mt-2"
            value={singlePersonality[1] ?? ""}
            border={"0.5px solid grey"}
            rounded={"md"}
            focusBorderColor="grey"
            placeholder="Reformulated Answer"
            onChange={(e) => handleInputChange(1, e.target.value)}
            isDisabled={loading}
            resize="vertical"
          />
        </FormControl>

        <Button
          onClick={addArray}
          isDisabled={
            loading ||
            singlePersonality[0].length == 0 ||
            singlePersonality[1].length == 0
          }
          size="sm"
          p={3}
          style={{ width: "fit-content" }}
          // colorScheme="whatsapp"
        >
          <p>Add Personality</p>
        </Button>

        {personalityList?.length > 0 && (
          <div className="flex flex-col gap-2 mb-8">
            <h2 className="text-white">Personality List:</h2>

            {personalityList.map((arr, index) => (
              <PersonalityListElement
                key={index}
                index={index}
                personality={arr}
                removePersonalityElement={removePersonalityElement}
              />
            ))}
          </div>
        )}
      </div>

      <IconButton
        aria-label="Create Person"
        mt={"auto"}
        alignSelf={"center"}
        // style={{ width: "fit-content" }}
        width={"50%"}
        colorScheme="gray"
        isLoading={loading}
        isDisabled={loading || name == "" || documents.length == 0}
        onClick={() => {
          createPerson();
        }}
        icon={<FontAwesomeIcon icon={faUpload} width="14" />}
      />
    </>
  );
};
