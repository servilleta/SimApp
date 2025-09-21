import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';

function useMarkdown(url) {
  const [content, setContent] = useState('Loading...');
  useEffect(() => {
    fetch(url)
      .then(r => r.text())
      .then(setContent)
      .catch(() => setContent('Failed to load.'));
  }, [url]);
  return content;
}

function TermsOfServicePage() {
  const termsText = useMarkdown('/legal/TERMS_OF_SERVICE.md');

  return (
    <div className="container" style={{ padding: '2rem' }}>
      <ReactMarkdown>{termsText}</ReactMarkdown>
    </div>
  );
}

export default TermsOfServicePage;
