import { faPlay, faTrash } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import React, { FC } from "react";

interface PersonalityListElementProps {
  personality: string[];
  index: number;
  removePersonalityElement: (index: number) => void;
}

export const PersonalityListElement: FC<PersonalityListElementProps> = ({
  personality,
  index,
  removePersonalityElement,
}) => {
  return (
    <div className="flex flex-row justify-between items-center text-white bg-dark p-4 rounded-md">
      <div className="flex flex-col gap-2">
        <div>
          <p className="font-bold">Original Answer:</p>
          <p>{personality[0]}</p>
        </div>
        <div>
          <p className="font-bold">Reformulated Answer:</p>
          <p>{personality[1]}</p>
        </div>
      </div>
      <FontAwesomeIcon
        width="14"
        onClick={() => {
          removePersonalityElement(index);
        }}
        className={`p-4`}
        icon={faTrash}
      />
    </div>
  );
};
