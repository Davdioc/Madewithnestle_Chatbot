import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {createRoot} from 'react-dom/client';
import NestlePage from './NestlePage';
import ChatWidget from './ChatWidget';
//import './App.css'
function App() {
  return (
    <div className="app">
      <NestlePage />
      <ChatWidget />
    </div>
    
  );
}

export default App;

