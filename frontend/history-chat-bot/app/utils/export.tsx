// utils/export.ts
import { saveAs } from "file-saver";
import { Message } from "@/app/utils/types"; // ← das fehlt

import jsPDF from "jspdf";

export function exportChatAsMarkdown(chat: Message[]) {
  const md = chat
    .map((m) =>
      m.isHuman ? `**You:** ${m.text}` : `**Bot** ${m.text}:** ${m.text}`
    )
    .join("\n\n---\n\n");
  const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
  saveAs(blob, "chat.md");
}

export function exportChatAsPDF(chat: Message[]) {
  const doc = new jsPDF({ unit: "pt", format: "letter" });
  let y = 40;
  doc.setFontSize(12);
  chat.forEach((m) => {
    const prefix = m.isHuman ? "You: " : "Bot";
    const lines = doc.splitTextToSize(prefix + m.text, 500);
    doc.text(lines, 40, y);
    y += lines.length * 14 + 10;
    if (y > 750) {
      doc.addPage();
      y = 40;
    }
  });
  doc.save("chat.pdf");
}
