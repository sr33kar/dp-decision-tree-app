import React, { useState } from "react";
import NodeCard from "./NodeCard";
import TreeData from "../data/TreeData";

interface TreeNavigatorProps {
  treeData: typeof TreeData;
}

export default function TreeNavigator({ treeData }: TreeNavigatorProps) {
  const [history, setHistory] = useState(["start"]);

  const currentNodeId = history[history.length - 1];
  const currentNode = treeData[currentNodeId];

  const handleOptionClick = (target: string) => {
    setHistory([...history, target]);
  };

  const handleBack = () => {
    if (history.length > 1) {
      setHistory(history.slice(0, -1));
    }
  };

  const handleRestart = () => {
    setHistory(["start"]);
  };

  return (
    <div className="flex flex-col items-center min-h-screen bg-gray-50 p-4">
      <div className="w-full max-w-3xl mb-4 flex justify-between">
        <button
          onClick={handleBack}
          className="px-4 py-2 bg-gray-300 text-gray-800 rounded hover:bg-gray-400"
          disabled={history.length <= 1}
        >
          ◀ Back
        </button>
        <button
          onClick={handleRestart}
          className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          🔄 Restart
        </button>
      </div>
      <NodeCard node={currentNode} onOptionClick={handleOptionClick} />
    </div>
  );
}
