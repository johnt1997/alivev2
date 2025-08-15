"use client";
import React, { useEffect, useState } from "react";
import { getAllPersons } from "../utils/api";
import { SettingsSideBar } from "@/components/molecules/SettingsSideBar";
import { PersonForm } from "@/components/molecules/PersonForm";
import { Button, useToast } from "@chakra-ui/react";
import { EditPersonForm } from "@/components/molecules/EditPersonForm";

const Settings = () => {
  const [persons, setPersons] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedPerson, setSelectedPerson] = useState<string>("");
  const [add, setAdd] = useState<boolean>(true);
  const toast = useToast();
  const loading_id = "loading-toast";

  useEffect(() => {
    if (loading && !toast.isActive(loading_id)) {
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
  }, [loading]);

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
    fetchPersons();
  }, []);

  return (
    <main className="flex bg-theme h-screen flex-row p-0">
      {/* <SideBar*/}
      <SettingsSideBar
        persons={persons}
        loading={loading}
        selectedPerson={selectedPerson}
        setAdd={setAdd}
        setSelectedPerson={setSelectedPerson}
        setPersons={setPersons}
      />

      {/** Add Person Section */}
      <div className="flex flex-[1] py-10 flex-col h-screen">
        {add && (
          <PersonForm
            loading={loading}
            persons={persons}
            setLoading={setLoading}
            setPersons={setPersons}
          />
        )}
        {selectedPerson && (
          <EditPersonForm
            name={selectedPerson}
            loading={loading}
            setLoading={setLoading}
            setPersons={setPersons}
            setSelectedPerson={setSelectedPerson}
            setAdd={setAdd}
          />
        )}
      </div>
    </main>
  );
};

export default Settings;
