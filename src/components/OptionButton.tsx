import React from "react";

interface OptionButtonProps {
  label: string;
  onClick: () => void;
}

export default function OptionButton({ label, onClick }: OptionButtonProps) {
  return (
    <button
      onClick={onClick}
      className="w-full px-4 py-2 bg-blue-600 text-white font-medium rounded-xl hover:bg-blue-700 transition"
    >
      {label}
    </button>
  );
}
