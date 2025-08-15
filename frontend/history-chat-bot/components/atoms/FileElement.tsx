import { faPlay, faTrash } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import React, { FC } from "react";

interface FileElementProps {
  document: string;
  clear: () => void;
}

export const FileElement: FC<FileElementProps> = ({ document, clear }) => {
  return (
    <div className="flex flex-row justify-between items-center text-white bg-dark p-2 rounded-md">
      <div className="flex flex-col gap-2">
        <p>{document}</p>
      </div>
      <FontAwesomeIcon
        width="14"
        onClick={() => {
          clear();
        }}
        className={`p-2`}
        icon={faTrash}
      />
    </div>
  );
};
