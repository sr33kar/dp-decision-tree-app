import React from "react";
import OptionButton from "./OptionButton";

interface Node {
  id?: string;
  question?: string;
  options?: { label: string; target: string }[];
  pattern?: string;
  description?: string;
  pseudocode?: string;
}

interface NodeCardProps {
  node: Node;
  onOptionClick: (target: string) => void;
}

export default function NodeCard({ node, onOptionClick }: NodeCardProps) {
  if (node.pattern) {
    return (
      <div className="max-w-3xl p-6 bg-white rounded-2xl shadow-lg border border-gray-200">
        <h1 className="text-2xl font-bold mb-4">{node.pattern}</h1>
        <p className="text-gray-700 mb-4">{node.description}</p>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto">
          {node.pseudocode}
        </pre>
      </div>
    );
  }

  return (
    <div className="max-w-2xl p-6 bg-white rounded-2xl shadow-lg border border-gray-200">
      <h2 className="text-xl font-semibold mb-6 text-gray-800">{node.question}</h2>
      <div className="grid grid-cols-1 gap-4">
        {node.options?.map((opt, idx) => (
          <OptionButton key={idx} label={opt.label} onClick={() => onOptionClick(opt.target)} />
        ))}
      </div>
    </div>
  );
}
