import React, { useState, useEffect, useRef } from 'react';
import './ChatWidget.css';
import icon1 from './assets/icon1.png';
import icon2 from './assets/icon2.png';
import icon3 from './assets/icon3.png';
import icon4 from './assets/icon4.png';
import icon5 from './assets/icon5.png';
import dropdownIcon from './assets/dropdown.png';
import closeIcon from './assets/close.png';
import userIcon from './assets/usericon.png';
import sendMessageIcon from './assets/send.png';
import axios from 'axios';
import Markdown from 'react-markdown';


function ChatWidget({ label = 'Quicky' }) {
    //Icons
  const presetIcons = [icon1, icon2, icon3, icon4, icon5];

  const [isOpen, setIsOpen] = useState(false);
  const welcomeText = `Hi! I'm ${label}, your personal MadeWithNestlé AI assistant. Ask me anything, and I'll quickly search the entire site to find the answers you need.`;
  const typedWelcome = useTypingEffect(welcomeText, 25);
  const [showPreview, setShowPreview] = useState(true);
  

  const [messages, setMessages] = useState([
    {
      sender: 'bot',
      text: '', // will be updated after first render
      timestamp: new Date().toISOString()
    }
  ]);

  useEffect(() => {
    // update the welcome message after typing finishes
    setMessages(prev => {
      const updated = [...prev];
      updated[0].text = typedWelcome;
      return updated;
    });
  }, [typedWelcome]);

  const [input, setInput] = useState('');

  //Saved values
  const [botName, setBotName] = useState(label);
  const [botIcon, setBotIcon] = useState(presetIcons[0]);

  // editable values
  const [draftName, setDraftName] = useState(botName);
  const [draftIcon, setDraftIcon] = useState(botIcon);

  const [showDropdown, setShowDropdown] = useState(false);

  //Typing animation
  const [isTyping, setIsTyping] = useState(false);

  //scroll to bottom
  const chatBodyRef = useRef(null);
  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
    }
  }, [messages, isTyping]); 

  const toggleChat = () => {
    setIsOpen(!isOpen);
    setShowPreview(false);
  };


  const toggleDropdown = () => {
    setShowDropdown(!showDropdown);
    setDraftName(botName);
    setDraftIcon(botIcon);
  };

  const handleSave = () => {
    setBotName(draftName);
    setBotIcon(draftIcon);
    setShowDropdown(false);
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input,  timestamp: new Date().toISOString()};
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true); // start typing animation

    //retrieve user's location
    let coords = {};
    if (navigator.geolocation) {
      try {
        coords = await new Promise((resolve, reject) => {
          navigator.geolocation.getCurrentPosition(
            (position) => resolve({
              lat: position.coords.latitude,
              lng: position.coords.longitude
            }),
            () => resolve({})  // fail silently
          );
        });
      } catch (err) {
        console.warn('Location access failed');
      }
    }

    try {
      const response = await axios.post('http://localhost:8000/api/chat', {
        question: input,
        name: botName,
        lat: coords.lat || null,
        lng: coords.lng || null,
      });

      const botReply = {
        sender: 'bot',
        text: response.data.answer,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, botReply]);

    } catch (error) {
      console.error('Error fetching bot response:', error);
      setMessages(prev => [
        ...prev,
        {
          sender: 'bot',
          text: 'Something went wrong. Please try again.',
          timestamp: new Date().toISOString()
        }
      ]);

    } finally {
      setIsTyping(false); // stop typing animation
    }
  };
  return (
    <>
      {isOpen && <div className="chat-overlay" onClick={toggleChat}></div>}

      {isOpen && (
        <div className="chat-modal">
          <div className="chat-header">
            <div className="chat-header-left">
              <img src={botIcon} alt="Bot Icon" className="bot-icon" />
              <span>{botName}</span>
            </div>
            <div className="chat-header-right">
              <button className="dropdown-toggle" onClick={toggleDropdown}>
                <img src={dropdownIcon} alt="Dropdown" style={{ width: '20px', height: '20px' }} />
              </button>
              <button className="close-btn" onClick={toggleChat}>
                <img src={closeIcon} alt="Close" style={{ width: '18px', height: '18px' }} />
              </button>
            </div>

            {showDropdown && (
              <div className="chat-dropdown">
                <label>
                  Name:
                  <input
                    type="text"
                    value={draftName}
                    onChange={(e) => setDraftName(e.target.value)}
                  />
                </label>
                <div className="icon-options">
                  {presetIcons.map((icon, idx) => (
                    <img
                      key={idx}
                      src={icon}
                      alt={`icon-${idx + 1}`}
                      className={`icon-option ${icon === draftIcon ? 'selected' : ''}`}
                      onClick={() => setDraftIcon(icon)}
                    />
                  ))}
                </div>
                <button className="save-settings" onClick={handleSave}>Save</button>
              </div>
            )}
          </div>

          <div className="chat-body" ref = {chatBodyRef}>
            <div className="chat-welcome-section">
              <img src={botIcon} alt="Bot" className="chat-welcome-icon" />
              <div className="chat-welcome-name">{botName}</div>
              <br />
              <div className="chat-welcome-time">{new Date(messages[0].timestamp).toLocaleString()}</div>
            </div>
            {messages.map((msg, idx) => (
              <div key={idx} className={`chat-msg ${msg.sender}`}>
                <div className={`msg-wrapper ${msg.sender}`}>
                  <img src={msg.sender === 'bot' ? botIcon : userIcon} alt={msg.sender} className={`msg-avatar ${msg.sender === 'bot' ? 'left' : 'right'}`} />
                  <div>
                    <div className={`msg-bubble ${msg.sender === 'bot' ? 'bot-bubble-custom' : 'user-bubble-custom'}`}>
                      <Markdown>{msg.text}</Markdown>
                      <div className="msg-timestamp-inside">
                        {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="chat-msg bot">
                <div className="msg-wrapper bot">
                  <img src={botIcon} alt="bot" className="msg-avatar left" />
                  <div className="msg-bubble bot-bubble typing">
                    <span className="dot"></span>
                    <span className="dot"></span>
                    <span className="dot"></span>
                  </div>
                </div>
              </div>
          )}

          </div>

          <div className="chat-input">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask me anything..."
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleSend();
                }
              }}
            />
            <button className="send-button" onClick={handleSend}>
                <img src={sendMessageIcon} className="send-icon" />
            </button>

          </div>
        </div>
      )}
      {showPreview && !isOpen && (
        <div className="preview-wrapper">
          <div className="preview-bubble" onClick={toggleChat}>
            {typedWelcome}
          </div>
          <span className="preview-close-outside" onClick={() => setShowPreview(false)}>×</span>
        </div>
      )}
      <button className={`chat-launcher slide-in`} onClick={toggleChat}>
        {isOpen ? <img src={closeIcon} id = "idk"/> : <img src={botIcon} alt="Chat Icon" />}
      </button>
    </>
  );
}

function useTypingEffect(text, speed = 30) {
  const [displayedText, setDisplayedText] = useState('');

  useEffect(() => {
    let currentIndex = -1;
    setDisplayedText('');

    const intervalId = setInterval(() => {
      if (currentIndex < text.length-1) {
        setDisplayedText((prev) => prev + text[currentIndex]);
        currentIndex += 1;
      } else {
        clearInterval(intervalId);
      }
    }, speed);

    return () => clearInterval(intervalId);
  }, [text, speed]);

  return displayedText;
}

export default ChatWidget;