"use client";
import { RelevantDocuments, TransformationSteps } from "@/app/utils/types";
import {
  Accordion,
  AccordionButton,
  AccordionIcon,
  AccordionItem,
  AccordionPanel,
  Box,
} from "@chakra-ui/react";
import React, { FC } from "react";
import { stopwords } from "@/app/utils/stopwords";
interface INFAccordionSection {
  relevantDocuments: RelevantDocuments[];
  transformationSteps: TransformationSteps | undefined;
  answer: string;
}

export const AccordionSection: FC<INFAccordionSection> = ({
  relevantDocuments,
  transformationSteps,
  answer,
}) => {
  return (
    <Accordion
      className="mb-4 bg-white rounded-xl"
      //defaultIndex={[0]}
      allowMultiple={false}
      allowToggle={true}
    >
      {relevantDocuments.map((document, index) => (
        <AccordionItem key={index} className="rounded-xl">
          <h2>
            <AccordionButton>
              <Box as="span" flex="1" textAlign="left">
                {document.source +
                  " - " +
                  "Page: " +
                  document.page +
                  " - " +
                  document.score.toFixed(2)}
              </Box>
              <AccordionIcon />
            </AccordionButton>
          </h2>
          <AccordionPanel pb={4}>
            {document.page_content
              .split(/[\s,.!?]+/)
              .map((word: string, index: number) => (
                <span
                  key={index}
                  style={{
                    backgroundColor:
                      answer.includes(word) && !stopwords.includes(word)
                        ? "yellow"
                        : "transparent",
                  }}
                >
                  {word}{" "}
                </span>
              ))}
          </AccordionPanel>
        </AccordionItem>
      ))}

      {transformationSteps && relevantDocuments.length > 0 && (
        <AccordionItem
          key={relevantDocuments.length + 1}
          className="rounded-xl"
        >
          <h2>
            <AccordionButton>
              <Box as="span" flex="1" textAlign="left">
                Transformation Steps
              </Box>
              <AccordionIcon />
            </AccordionButton>
          </h2>
          <AccordionPanel pb={4}>
            <p>
              <p className="font-bold">Original Question:</p>{" "}
              {transformationSteps.original_query}
            </p>
            <hr />
            <p>
              <p className="font-bold">Follow Up Question:</p>{" "}
              {transformationSteps.follow_up_question}{" "}
            </p>
            <hr />
            <p>
              <p className="font-bold">Third Person Question:</p>
              {transformationSteps.third_person_query}
            </p>
            <hr />
            <hr />
            <hr />
            <p>
              <p className="font-bold">Original Answer:</p>{" "}
              {transformationSteps.original_answer}
            </p>
            <hr />
            <p>
              <p className="font-bold">Transformed Answer: </p>
              {transformationSteps.transformed_answer}
            </p>
            <hr />
          </AccordionPanel>
        </AccordionItem>
      )}
    </Accordion>
  );
};
