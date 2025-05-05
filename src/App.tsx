import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import HomePage from "./components/HomePage";
import TreeNavigator from "./components/TreeNavigator";
import TreeData from "./data/TreeData";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/tree" element={<TreeNavigator treeData={TreeData} />} />
      </Routes>
    </Router>
  );
}
