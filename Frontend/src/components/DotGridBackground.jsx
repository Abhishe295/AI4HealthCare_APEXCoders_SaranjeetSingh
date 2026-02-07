import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import DotGrid from './DotGrid.jsx';

const BackgroundContainer = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 0;
  pointer-events: none;
`;

const DotGridBackground = () => {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Don't render animated dot background on mobile devices
  if (isMobile) {
    return (
      <BackgroundContainer />
    );
  }

  return (
    <BackgroundContainer>
      <DotGrid
        dotSize={3}
        gap={20}
        baseColor="#570DF8"
        activeColor="#F000B8"
        proximity={120}
        shockRadius={150}
        shockStrength={4}
        resistance={850}
        returnDuration={1.5}
      />
    </BackgroundContainer>
  );
};

export default DotGridBackground;