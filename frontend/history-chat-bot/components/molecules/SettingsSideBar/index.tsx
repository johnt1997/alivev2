"use client";
import { Button, IconButton, useToast } from "@chakra-ui/react";
import {
  faArrowLeft,
  faPlus,
  faTrash,
  faUser,
} from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import React, { FC, useEffect, useState } from "react";
import Link from "next/link";
import { delete_person, getAllPersons } from "@/app/utils/api";
import { Credits } from "@/components/atoms/Credits";

interface SettingsSideBarProps {
  persons: string[];
  loading: boolean;
  selectedPerson: string;
  setSelectedPerson: (person: string) => void;
  setAdd: (add: boolean) => void;
  setPersons: (persons: string[]) => void;
}

export const SettingsSideBar: FC<SettingsSideBarProps> = ({
  persons,
  loading,
  selectedPerson,
  setAdd,
  setSelectedPerson,
  setPersons,
}) => {
  const [showWarning, setShowWarning] = useState<boolean>(true);
  const warningId = "warning-toast";
  const toast = useToast();

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

  const deletePerson = async () => {
    const response = await delete_person(selectedPerson);
    if (!response?.ok) {
      toast({
        title: "Ups!",
        description: "Something went wrong, please try again later.",
        status: "error",
        duration: 5000,
        isClosable: true,
        position: "top",
      });
    }

    await fetchPersons();
    toast({
      title: "Success!",
      description: `${selectedPerson} Deleted!`,
      status: "success",
      duration: 5000,
      isClosable: true,
      position: "top",
    });
    setSelectedPerson("");
    setAdd(true);
  };

  useEffect(() => {
    if (!toast.isActive(warningId) && showWarning && selectedPerson) {
      setShowWarning(false);
      toast({
        id: warningId,
        title: "Warning",
        description:
          "Updating a Person will overwrite the old one, make sure to upload all necessary files again.",
        status: "warning",
        duration: 4000,
        isClosable: true,
        position: "top",
      });
    }
  }, [selectedPerson, showWarning]);

  return (
    <div className="flex flex-[0.2] flex-col h-screen bg-dark">
      <div
        className="flex flex-1 flex-col gap-4 overflow-y-scroll px-5 pt-5"
        style={{
          scrollbarWidth: "none",
        }}
      >
        {persons.map((person, index) => (
          <Button
            key={index}
            isDisabled={loading}
            onClick={() => {
              setSelectedPerson(person), setAdd(false);
            }}
            colorScheme={selectedPerson == person ? "gray" : "transparent"}
            size="sm"
            p={2}
            justifyContent={"space-between"}
            leftIcon={
              <div className="flex items-center">
                <p
                  key={index}
                  className={`${
                    selectedPerson == person ? "text-black" : "text-white"
                  }`}
                >
                  {person}
                </p>
              </div>
            }
            rightIcon={
              selectedPerson == person ? (
                <FontAwesomeIcon
                  onClick={() => {
                    deletePerson();
                  }}
                  icon={faTrash}
                  width="14"
                />
              ) : undefined
            }
          ></Button>
        ))}
      </div>

      <div className="mt-auto flex gap-2 p-5">
        <Link href="/">
          <IconButton
            size={"sm"}
            flex={0.5}
            aria-label="goBack"
            icon={<FontAwesomeIcon icon={faArrowLeft} width="14" />}
            isDisabled={loading}
          />
        </Link>

        <IconButton
          aria-label="addPerson"
          flex={1}
          size={"sm"}
          icon={<FontAwesomeIcon icon={faPlus} width="14" />}
          isDisabled={loading || !selectedPerson}
          onClick={() => {
            setAdd(true);
            setSelectedPerson("");
          }}
        />
      </div>
      <div className="px-5">
        <Credits />
      </div>
    </div>
  );
};
