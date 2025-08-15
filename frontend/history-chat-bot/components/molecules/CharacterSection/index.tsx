"use client";
import { Avatar } from "@/components/Avatar";
import { Environment, OrbitControls, useTexture } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import React, { FC, Suspense, memo } from "react";

interface CharacterSectionProps {
  audioPlaying: boolean;
  setAudioPlaying: (playing: boolean) => void;
  modelUrl: string;
  backgroundUrl: string;
}

export const CharacterSection: FC<CharacterSectionProps> = memo(
  ({ audioPlaying, setAudioPlaying, modelUrl, backgroundUrl }) => {
    return (
      <div className="flex flex-1 h-screen bg-theme p-5">
        <div className="flex flex-1 rounded-md items-center justify-center">
          <Canvas
            camera={{ position: [0, 0, 4], fov: 35 }}
            className={`bg-theme rounded-md`}
          >
            <ambientLight intensity={1.25} />
            <ambientLight intensity={0.1} />
            <directionalLight intensity={0.4} />
            <Suspense fallback={null}>
              <Avatar
                setAudioPlaying={setAudioPlaying}
                audioPlaying={audioPlaying}
                position={{ x: 0, y: -1.5, z: 2 }}
                modelUrl={modelUrl}
                backgroundUrl={backgroundUrl}
              />
            </Suspense>
          </Canvas>
        </div>
      </div>
    );
  }
);
