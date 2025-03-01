import React, { useState } from 'react';
import HomePage from './components/HomePage';
import LoginModal from './components/LoginModal';
import PredictionModal from './components/PredictionModal';
import InfoModal from './components/InfoModal';

function App() {
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [showPredictionModal, setShowPredictionModal] = useState(false);
  const [showInfoModal, setShowInfoModal] = useState(false);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <HomePage 
        onLoginClick={() => setShowLoginModal(true)}
        onPredictionClick={() => setShowPredictionModal(true)}
        onLearnMoreClick={() => setShowInfoModal(true)}
      />
      
      {showLoginModal && (
        <LoginModal onClose={() => setShowLoginModal(false)} />
      )}
      
      {showPredictionModal && (
        <PredictionModal onClose={() => setShowPredictionModal(false)} />
      )}
      
      {showInfoModal && (
        <InfoModal onClose={() => setShowInfoModal(false)} />
      )}
    </div>
  );
}

export default App;