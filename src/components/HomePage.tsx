import React from "react";
import { useNavigate } from "react-router-dom";
import image from "../assets/image.jpg";

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-indigo-100 to-blue-50 p-6">
      <h1 className="text-4xl font-bold text-gray-800 mb-4 text-center">
        🧠 Dynamic Programming Decision Helper
      </h1>
      <p className="text-lg text-gray-600 mb-8 text-center max-w-xl">
        Click through simple questions to find the right DP pattern, pseudocode, and explanation for your problem—without typing a word.
      </p>
      <button
        onClick={() => navigate("/tree")}
        className="px-6 py-3 bg-blue-600 text-white rounded-xl text-lg hover:bg-blue-700 transition"
      >
        🚀 Start Decision Tree
      </button><br></br>
      <img src={image} width="300" height="400"></img>
    </div>
  );
}
