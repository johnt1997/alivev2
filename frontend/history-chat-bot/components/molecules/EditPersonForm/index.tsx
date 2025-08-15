"use client";
import {
  create_person,
  delete_person,
  getAllPersons,
  getVirtualCharacter,
  get_background,
  get_document,
  get_person_data,
  get_person_documents_names,
  person_voice,
  upload_background,
  upload_person_character,
  upload_person_data,
  upload_person_voice,
} from "@/app/utils/api";
import { DocumentListElement } from "@/components/atoms/DocumentListElement";
import { FileElement } from "@/components/atoms/FileElement";
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
import { faTrash, faUpload } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import Link from "next/link";
import React, { FC, useEffect, useState } from "react";

interface EditPersonFormProps {
  name: string;
  loading: boolean;
  setLoading: (loading: boolean) => void;
  setSelectedPerson: (person: string) => void;
  setPersons: (persons: string[]) => void;
  setAdd: (add: boolean) => void;
}

export const EditPersonForm: FC<EditPersonFormProps> = ({
  name,
  loading,
  setLoading,
  setPersons,
  setSelectedPerson,
  setAdd,
}) => {
  const [personalityList, setPersonalityList] = useState<string[][]>([]);
  const [personalityListOriginal, setPersonalityListOriginal] = useState<
    string[][]
  >([]);
  const [singlePersonality, setSinglePersonality] = useState<string[]>([
    "",
    "",
  ]);

  const [voiceFile, setVoiceFile] = useState<File>();
  const [virtualAvatarFile, setVirtualAvatarFile] = useState<File>();
  const [documents, setDocuments] = useState<File[]>([]);
  const [bgFile, setBGFile] = useState<File>();
  const documentRef = React.useRef<HTMLInputElement>(null);
  const virtualAvatarRef = React.useRef<HTMLInputElement>(null);
  const bgRef = React.useRef<HTMLInputElement>(null);
  const voiceRef = React.useRef<HTMLInputElement>(null);
  const toast = useToast();
  const [canSubmit, setCanSubmit] = useState(false);

  const personalityHasChanged = () => {
    return (
      JSON.stringify(personalityList) !==
      JSON.stringify(personalityListOriginal)
    );
  };

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

  const handleInputChange = (index: number, value: string) => {
    const newArray = [...singlePersonality];
    newArray[index] = value;
    setSinglePersonality(newArray);
  };

  const addArray = () => {
    setCanSubmit(true);
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

  const updatePerson = async () => {
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

    if (documents.length) {
      const upd_response = await upload_person_data(name, documents);
      if (!upd_response?.ok) {
        throwToast();
        const response = await delete_person(name);
        return;
      }
    }

    if (voiceFile) {
      const upv_response = await upload_person_voice(name, voiceFile);
      if (!upv_response?.ok) {
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

    if (bgFile) {
      const bg_response = await upload_background(name, bgFile);

      if (!bg_response?.ok) {
        throwToast();
        const response = await delete_person(name);
        return;
      }
    }

    const fp_response = await fetchPersons();

    documentRef.current!.value = "";
    virtualAvatarRef.current!.value = "";
    voiceRef.current!.value = "";
    bgRef.current!.value = "";

    // setDocuments([]);
    // setVoiceFile(undefined);
    // setVirtualAvatarFile(undefined);
    setLoading(false);
    setSinglePersonality(["", ""]);
    setCanSubmit(false);
    toast({
      title: "Success!",
      description: `${name} Updated`,
      status: "success",
      duration: 5000,
      isClosable: true,
      position: "top",
    });
  };

  const removePersonalityElement = (index: number) => {
    setCanSubmit(true);
    setPersonalityList((prev: string[][]) => {
      return [...prev.slice(0, index), ...prev.slice(index + 1)];
    });
  };

  const removeDocumentListElement = (index: number) => {
    setCanSubmit(true);
    setDocuments((prev: File[]) => {
      return [...prev.slice(0, index), ...prev.slice(index + 1)];
    });
  };

  useEffect(() => {
    const setInitValues = async () => {
      setLoading(true);
      setDocuments([]);
      setBGFile(undefined);
      setVoiceFile(undefined);
      setVirtualAvatarFile(undefined);
      setPersonalityList([]);
      const response = await get_person_data(name);
      if (response && response.ok) {
        const person = await response.json();
        setPersonalityList(person.personality ?? []);
        setPersonalityListOriginal(person.personality ?? []);
      } else {
        throwToast();
        return;
      }

      const doc_name_response = await get_person_documents_names(name);
      if (doc_name_response && doc_name_response.ok) {
        const data: string[] = await doc_name_response.json();
        data.forEach((doc_name) => {
          const data_response = get_document(name, doc_name);
          data_response.then((response) => {
            if (response && response.ok) {
              const blob = response.blob();
              blob.then((blob) => {
                const file = new File([blob], doc_name);
                setDocuments((documents: File[]) => [...documents, file]);
              });
            } else {
              throwToast();
              return;
            }
          });
        });
      } else {
        throwToast();
        return;
      }

      const avatar_response = await getVirtualCharacter(name);
      if (avatar_response && avatar_response.ok) {
        if (
          avatar_response.headers.get("content-type") ==
            "application/octet-stream" ||
          avatar_response.headers.get("content-type") == "model/gltf-binary"
        ) {
          const blob = await avatar_response.blob();
          const file = new File([blob], name + ".glb");
          setVirtualAvatarFile(file);
        }
      } else {
        throwToast();
        return;
      }

      const voice_response = await person_voice(name);
      if (voice_response && voice_response.ok) {
        if (voice_response.headers.get("content-type") !== "application/json") {
          const blob = await voice_response.blob();
          const file = new File([blob], name + ".wav");
          setVoiceFile(file);
        }
      } else {
        throwToast();
        return;
      }

      const background_response = await get_background(name);
      if (background_response && background_response.ok) {
        if (
          background_response.headers.get("content-type") !== "application/json"
        ) {
          const blob = await background_response.blob();
          const file = new File([blob], name + ".jpg");
          setBGFile(file);
        }
      } else {
        throwToast();
        return;
      }

      setLoading(false);
    };

    setInitValues();
  }, [name]);

  return (
    <>
      <div
        className="flex flex-[1] w-1/2 self-center flex-col gap-4 overflow-y-scroll pb-10"
        style={{
          scrollbarWidth: "none",
        }}
      >
        <div className="flex flex-row items-center gap-10">
          <h1 className="text-5xl text-white">{name}</h1>
        </div>

        <FormControl>
          <FormLabel className="text-white">
            Upload Documents (.pdf, .txt)
          </FormLabel>
          <Input
            className="text-white items-center"
            type={"file"}
            border={"none"}
            p={0}
            title="Upload a file"
            accept=".pdf,.txt"
            ref={documentRef}
            multiple={true}
            onChange={(e) => {
              let files = e.target.files;
              if (files?.length) {
                setCanSubmit(true);
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

        {documents?.length > 0 && (
          <div className="flex flex-col gap-2">
            <h2 className="text-white">Documents</h2>

            {documents.map((doc, index) => (
              <DocumentListElement
                key={index}
                index={index}
                document={doc.name}
                removeDocumentListElement={removeDocumentListElement}
              />
            ))}
          </div>
        )}

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
            p={0}
            type={"file"}
            title="Upload a file"
            accept=".glb"
            ref={virtualAvatarRef}
            multiple={false}
            onChange={(e) => {
              let files = e.target.files;
              if (files?.length) {
                setCanSubmit(true);
                setVirtualAvatarFile(files[0]);
              }
            }}
            isDisabled={loading}
          />
        </FormControl>

        {virtualAvatarFile && (
          <FileElement
            document={virtualAvatarFile?.name ?? ""}
            clear={() => {
              setCanSubmit(true);
              setVirtualAvatarFile(undefined);
            }}
          />
        )}

        <FormControl>
          <FormLabel className="text-white">
            <div className="flex gap-2">
              Upload Avatar Background (Optionally | .jpg)
            </div>
          </FormLabel>
          <Input
            className="text-white items-center"
            border={"none"}
            p={0}
            type={"file"}
            title="Upload a file"
            accept=".jpg"
            ref={bgRef}
            multiple={false}
            onChange={(e) => {
              let files = e.target.files;
              if (files?.length) {
                setCanSubmit(true);
                setBGFile(files[0]);
              }
            }}
            isDisabled={loading}
          />
        </FormControl>

        {bgFile && (
          <FileElement
            document={bgFile?.name ?? ""}
            clear={() => {
              setCanSubmit(true);
              setBGFile(undefined);
            }}
          />
        )}

        <FormControl>
          <FormLabel className="text-white">
            Upload Voice (Optionally | .wav file)
          </FormLabel>
          <Input
            className="text-white items-center"
            border={"none"}
            p={0}
            type={"file"}
            title="Upload a file"
            ref={voiceRef}
            multiple={false}
            accept="audio/wav"
            onChange={(e) => {
              let files = e.target.files;
              if (files?.length) {
                setCanSubmit(true);
                setVoiceFile(files[0]);
              }
            }}
            isDisabled={loading}
          />
        </FormControl>

        {voiceFile && (
          <FileElement
            document={voiceFile?.name ?? ""}
            clear={() => {
              setCanSubmit(true);
              setVoiceFile(undefined);
            }}
          />
        )}

        <FormControl className="">
          <FormLabel className="text-white">
            Define Personality (optional)
          </FormLabel>
          <Textarea
            className="text-white items-center"
            border={"0.5px solid gray"}
            focusBorderColor="grey"
            value={singlePersonality[0] ?? ""}
            placeholder="Original Answer"
            onChange={(e) => handleInputChange(0, e.target.value)}
            isDisabled={loading}
          />

          <Textarea
            className="text-white items-center mt-2"
            border={"0.5px solid gray"}
            focusBorderColor="grey"
            value={singlePersonality[1] ?? ""}
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
        >
          <p>Add Personality</p>
        </Button>

        {personalityList?.length > 0 && (
          <div className="flex flex-col gap-2">
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
        isDisabled={loading || !documents.length || !canSubmit}
        onClick={() => {
          updatePerson();
        }}
        icon={<FontAwesomeIcon icon={faUpload} width="14" />}
      />
    </>
  );
};
