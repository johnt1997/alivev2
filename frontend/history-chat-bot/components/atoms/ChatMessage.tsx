"use client";
import { Message } from "@/app/utils/types";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { Box, HStack, IconButton } from "@chakra-ui/react";
import { FaThumbsUp, FaThumbsDown } from "react-icons/fa";
import React, { FC, use, useEffect, useState } from "react";
import {
  faPlay,
  faFileAudio,
  faStop,
  faRobot,
  faUser,
  faComment,
} from "@fortawesome/free-solid-svg-icons";
import { getAudioResponse } from "@/app/utils/api";
import { Spinner, useToast } from "@chakra-ui/react";

interface INFChatMessage {
  message: Message;
  selected: string;
  hasVoice: boolean;
  globalDisabled: number;
  setGlobalDisabled: (value: number | ((prev: number) => number)) => void;
  audioPlaying: boolean;
  setAudioPlaying: (playing: boolean) => void;
  selectedVoice: string;
  selectedLanguage: string;
}

export const ChatMessage: FC<INFChatMessage> = ({
  message,
  selected,
  hasVoice,
  globalDisabled,
  setGlobalDisabled,
  audioPlaying,
  setAudioPlaying,
  selectedVoice,
  selectedLanguage,
}) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [uId, setUid] = useState<string>(selected + new Date().getTime());
  const [audioElement, setAudioElement] = useState<HTMLAudioElement>();
  const [vote, setVote] = useState<"up" | "down" | null>(null);
  const [audioPlayingLocal, setAudioPlayingLocal] = useState<boolean>(false);
  const toast = useToast();

  useEffect(() => {
    setAudioElement(undefined);
  }, [selectedVoice, selectedLanguage]);

  const handleToggleAudio = async () => {
    if (audioElement) {
      if (audioPlaying) {
        audioElement.pause();
        audioElement.currentTime = 0;
        setAudioPlaying(false);
        setAudioPlayingLocal(false);
      } else {
        const play = audioElement.play();
        if (play) {
          setAudioPlaying(true);
          setAudioPlayingLocal(true);

          await new Promise((resolve, reject) => {
            play
              .then(() => {
                audioElement.addEventListener("ended", resolve);
              })
              .catch(reject);
          });
        }
        setAudioPlaying(false);
        setAudioPlayingLocal(false);
      }
    }
  };

  const getAudio = async () => {
    setLoading(true);
    setGlobalDisabled((value) => Math.max(value + 1, 0));
    const response = await getAudioResponse(
      selected,
      uId,
      message.text,
      selectedVoice,
      selectedLanguage
    );
    if (response && response.ok) {
      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      const audioElement = new Audio(audioUrl);
      setAudioElement(audioElement);
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
    setLoading(false);
    setAudioPlaying(false);
  };

  return (
    <div className="flex flex-col mb-6 items-start">
      <div className="flex items-center gap-2">
        {message.isHuman ? (
          <FontAwesomeIcon
            width="14"
            icon={faComment}
            className="bg-white p-2 rounded-full"
          />
        ) : (
          <FontAwesomeIcon
            width="14"
            icon={faRobot}
            className="bg-white p-2 rounded-full"
          />
        )}

        <p className="font-extrabold text-white ">
          {message.isHuman ? "You" : selected}
        </p>
      </div>

      <div className="ml-10 flex flex-col gap-2">
        <p className="text-white">{message.text}</p>
        {!message.isHuman && loading && <Spinner size={"sm"} color="white" />}
        {/** State before loading audio -> get the audiofile */}
        {!message.isHuman && !loading && audioElement == undefined && (
          <FontAwesomeIcon
            color="white"
            width="14"
            onClick={() => {
              if (!globalDisabled && !audioPlaying) getAudio();
            }}
            className={`${
              (globalDisabled || audioPlaying) &&
              "opacity-60 cursor-not-allowed"
            }`}
            icon={faFileAudio}
          />
        )}

        {/** State with loaded file -> Play button */}
        {!message.isHuman && !loading && audioElement && !audioPlayingLocal && (
          <FontAwesomeIcon
            color="white"
            width="14"
            onClick={() => {
              if (!globalDisabled && !audioPlaying) handleToggleAudio();
            }}
            className={` ${
              (globalDisabled || audioPlaying) &&
              "opacity-60 cursor-not-allowed"
            }`}
            icon={faPlay}
          />
        )}

        {/** State while playing audio -> stop button */}
        {!message.isHuman && !loading && audioElement && audioPlayingLocal && (
          <FontAwesomeIcon
            color="white"
            width="14"
            onClick={() => {
              if (!globalDisabled) handleToggleAudio();
            }}
            className={`${globalDisabled && "opacity-60 cursor-not-allowed"}`}
            icon={faStop}
          />
        )}
        {!message.isHuman && (
          <HStack spacing={2} mt={2}>
            <IconButton
              aria-label="Upvote"
              icon={<FaThumbsUp />}
              size="sm"
              colorScheme={vote === "up" ? "green" : undefined}
              onClick={() => setVote(vote === "up" ? null : "up")}
            />
            <IconButton
              aria-label="Downvote"
              icon={<FaThumbsDown />}
              size="sm"
              colorScheme={vote === "down" ? "red" : undefined}
              onClick={() => setVote(vote === "down" ? null : "down")}
            />
          </HStack>
        )}
      </div>
    </div>
  );
};
