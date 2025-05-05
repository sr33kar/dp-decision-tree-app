# DP Pattern Decision Tree App

This web application allows you to navigate through a dynamic programming decision tree to identify the correct DP pattern, understand its logic, and see pseudocode with detailed explanation — **without needing to enter a full problem description**.

## 🚀 Features

- Decision tree-based navigation to find suitable DP pattern
- Each leaf shows:
  - Pattern name
  - Problem description
  - Long pseudocode with inline comments
- Back and Restart navigation
- Easily extendable with more patterns

## 📦 Installation

### 1. Clone the repo or download files

```bash
git clone <your-repo-url>
cd <project-folder>
```

Or unzip the code from the provided archive.

### 2. Install dependencies

Make sure you have Node.js and npm installed.

```bash
npm install
```

### 3. Run the app

```bash
npm run dev
```

This will start the development server at:

```
http://localhost:5173
```

## 📁 Project Structure

```
src/
  components/         # React components like TreeNavigator
  data/TreeData.ts    # DP tree logic and pseudocode
  App.tsx             # Root of the app
  main.tsx            # Entry point
```

## 🧠 How to Use

1. Choose the structure of your DP problem (e.g., array, tree, graph).
2. Follow the on-screen questions to refine the type.
3. At the end, you'll reach a detailed pattern node with:
   - Problem category
   - Explanation
   - Pseudocode

## 📜 License

MIT

---

Built for DP interview mastery.
