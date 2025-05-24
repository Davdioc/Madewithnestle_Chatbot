import React from 'react';
import { Helmet } from 'react-helmet';

function NestlePage() {
  return (
    <>
      <iframe
          src="/NP.html"
          style={{
            width: '100%',
            height: '100vh',
            border: 'none',
            overflow: 'hidden',
          }}
          title="Nestlé Clone"
        />
    </>
  );
}

export default NestlePage;