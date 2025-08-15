"use client";
// SOURCE: Code Adapted from https://github.com/wass08/r3f-lipsync-tutorial/tree/main
import React, { useEffect, useRef, useState } from "react";
import { useAnimations, useFBX, useGLTF, useTexture } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import { get_background } from "@/app/utils/api";

interface AvatarProps {
  position: { x: number; y: number; z: number };
  audioPlaying: boolean;
  setAudioPlaying: (playing: boolean) => void;
  modelUrl: string;
  backgroundUrl: string;
}

export function Avatar(props: AvatarProps) {
  const model = useGLTF(props.modelUrl);
  const texture = useTexture(props.backgroundUrl);

  const { animations: idleAnimation } = useFBX("/animations/Idle.fbx");
  const { animations: talkingAnimation } = useFBX("/animations/Talking.fbx");

  const viewport = useThree((state) => state.viewport);

  idleAnimation[0].name = "Idle";
  talkingAnimation[0].name = "Talking";

  const group = useRef<any>();
  const [animation, setAnimation] = useState<string>("Idle");
  const { actions } = useAnimations(
    [idleAnimation[0], talkingAnimation[0]],
    group
  );

  useEffect(() => {
    const selectedAction = actions[animation];

    if (selectedAction) {
      selectedAction?.reset()?.fadeIn(0.5)?.play();
      return () => {
        selectedAction?.fadeOut(0.5);
      };
    }
  }, [animation]);

  useEffect(() => {
    if (model.scene) {
      model.scene.position.set(
        props.position.x,
        props.position.y,
        props.position.z
      );
    }
  }, [props.modelUrl]);

  useEffect(() => {
    if (props.audioPlaying) {
      setAnimation("Talking");
    } else {
      setAnimation("Idle");
    }
  }, [props.audioPlaying]);

  return (
    <>
      <group ref={group} dispose={null}>
        <primitive object={model.scene} />
      </group>
      <mesh>
        <planeGeometry args={[viewport.width, viewport.height]} />
        <meshBasicMaterial map={texture} />
      </mesh>
    </>
  );
}
