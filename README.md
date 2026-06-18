# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.

## How to Build and Run the Project

### Prerequisites
- Node.js (for the React/Vite frontend)
- Python 3.8+ (for the FastAPI backend)

### 1. Start the Backend (FastAPI)
Open a terminal, navigate to the `backend` directory, install the Python dependencies, and run the server:

```bash
cd backend
pip install -r requirements.txt
python server.py
```
*The backend server will start and listen for incoming requests (usually on `http://localhost:8000` or `http://0.0.0.0:8000`).*

### 2. Start the Frontend (React + Vite)
Open a new terminal, navigate to the root directory of the project, install the Node dependencies, and start the development server:

```bash
npm install
npm run dev
```
*The frontend application will be available at `http://localhost:5173` (or another port specified in the console).*

### 3. Build Frontend for Production
To build the frontend application for production, run:

```bash
npm run build
```

You can preview the built application locally by running:

```bash
npm run preview
```
