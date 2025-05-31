import React, { useEffect, useState } from 'react';
import ChatWidget from './ChatWidget';
import NestlePage from './NestlePage';
import NestlePage2 from './NestlePage2';

function App() {
  const [useAltPage, setUseAltPage] = useState(false);

  useEffect(() => {
    const handleResize = () => {
      setUseAltPage(window.innerWidth < 1025);
    };

    handleResize(); //run on initial load
    window.addEventListener('resize', handleResize); //Update on window resize
    return () => window.removeEventListener('resize', handleResize); //clean up the event listener
  }, []);

  return (
    <div>
      {useAltPage ? <NestlePage2 /> : <NestlePage />}
      <ChatWidget />
    </div>
  );
}

export default App;

