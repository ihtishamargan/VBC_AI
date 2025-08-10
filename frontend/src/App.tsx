import React from 'react';
import { DocumentUpload } from './components/DocumentUpload';
import { SimpleChatInterface } from './components/SimpleChatInterface';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="app-header">
        <h1>VBC AI - Value-Based Care Contract Assistant</h1>
        <p>Upload VBC contracts and ask questions about them</p>
      </header>
      
      <div className="app-container">
        <div className="upload-section">
          <DocumentUpload />
        </div>
        
        <div className="chat-section">
          <SimpleChatInterface />
        </div>
      </div>
    </div>
  );
}

export default App;
