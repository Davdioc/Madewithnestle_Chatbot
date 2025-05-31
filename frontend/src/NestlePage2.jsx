import React, { useEffect } from 'react';

function NestlePage2() {
  return (
    <iframe
      src="/NP2.html"
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        border: 'none',
        zIndex: 0,
      }}
      title="Nestlé Page"
    />
  );
}
export default NestlePage2;