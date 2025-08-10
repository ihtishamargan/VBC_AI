/**
 * SimpleChatInterface Component - Custom chat interface for VBC AI
 */
import React, { useState, useRef, useEffect } from 'react';
import { Send, Copy, RotateCcw, MessageCircle } from 'lucide-react';
import { VBCApiService } from '../services/VBCApiService';
import './SimpleChatInterface.css';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: Array<{
    document_id: string;
    chunk_id: string;
    content: string;
    score: number;
  }>;
}

export const SimpleChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [inputValue]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await VBCApiService.sendChatMessage(userMessage.content);
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        sources: response.sources,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error: any) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error.message}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
  };

  const retryLastMessage = async () => {
    if (messages.length < 2) return;
    
    const lastUserMessage = [...messages].reverse().find(m => m.role === 'user');
    if (lastUserMessage) {
      // Remove the last assistant message and retry
      setMessages(prev => prev.slice(0, -1));
      setIsLoading(true);

      try {
        const response = await VBCApiService.sendChatMessage(lastUserMessage.content);
        
        const assistantMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: response.answer,
          timestamp: new Date(),
          sources: response.sources,
        };

        setMessages(prev => [...prev, assistantMessage]);
      } catch (error: any) {
        const errorMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: `Sorry, I encountered an error: ${error.message}. Please try again.`,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, errorMessage]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="simple-chat-interface">
      {/* Chat Header */}
      <div className="chat-header">
        <div className="chat-title">
          <MessageCircle size={20} />
          <span>VBC AI Assistant</span>
        </div>
        <div className="chat-status">
          {messages.length > 0 ? `${messages.length} messages` : 'Ready to chat'}
        </div>
      </div>

      {/* Messages Container */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <div className="empty-title">Start a VBC conversation</div>
            <div className="empty-description">
              Ask questions about your Value-Based Care contracts
            </div>
            <div className="empty-suggestions">
              <div className="suggestion" onClick={() => setInputValue("What are the key terms in this contract?")}>
                💡 "What are the key terms in this contract?"
              </div>
              <div className="suggestion" onClick={() => setInputValue("What outcome metrics are defined?")}>
                📊 "What outcome metrics are defined?"
              </div>
              <div className="suggestion" onClick={() => setInputValue("What is the payment model?")}>
                💰 "What is the payment model?"
              </div>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.role}`}>
                <div className="message-avatar">
                  {message.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-content">
                  <div className="message-text">
                    {message.content}
                  </div>
                  
                  {/* Sources */}
                  {message.sources && message.sources.length > 0 && (
                    <div className="message-sources">
                      <div className="sources-header">📄 Sources:</div>
                      <div className="sources-list">
                        {message.sources.map((source, idx) => (
                          <div key={idx} className="source-item">
                            <div className="source-content">
                              {source.content.substring(0, 100)}...
                            </div>
                            <div className="source-meta">
                              Score: {(source.score * 100).toFixed(0)}%
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Message Actions */}
                  {message.role === 'assistant' && (
                    <div className="message-actions">
                      <button 
                        className="action-button"
                        onClick={() => copyMessage(message.content)}
                        title="Copy message"
                      >
                        <Copy size={14} />
                      </button>
                      <button 
                        className="action-button"
                        onClick={retryLastMessage}
                        title="Retry message"
                        disabled={isLoading}
                      >
                        <RotateCcw size={14} />
                      </button>
                    </div>
                  )}
                  
                  <div className="message-timestamp">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
            
            {/* Loading Message */}
            {isLoading && (
              <div className="message assistant loading">
                <div className="message-avatar">🤖</div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Container */}
      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about your VBC contracts..."
            className="message-input"
            rows={1}
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="send-button"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
};
