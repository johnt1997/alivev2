import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGithub } from "@fortawesome/free-brands-svg-icons";

import React from "react";
import Link from "next/link";
export const Credits = () => {
  return (
    <div className="pb-2 flex-row flex gap-4 items-center">
      <Link href="https://github.com/thenextmz/" target="_blank">
        <FontAwesomeIcon icon={faGithub} color="white" width="14" />
      </Link>
      <p className="text-white font-bold">Mario Comanici, John Tusha 2025</p>
    </div>
  );
};
